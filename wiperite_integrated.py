#!/usr/bin/env python3
"""
WipeRite Integrated Controller
================================
Combines:
  1. whiteboardDetect.py  — detects the whiteboard quad and computes a traversal grid
  2. cameraColor.py       — EAST text detection + Tesseract OCR within the board bounds
  3. wiperite_client      — generates @WPE1 wire commands and sends them over TCP

Pipeline
--------
  Camera → WhiteboardDetect → grid cells
         → EAST/OCR          → text box regions (in board-relative UV coords)
         → CommandGen         → @WPE1 lines → TCP → TM4C/CC3100 robot

Usage
-----
  # View-only (browser stream, no robot connected):
  python3 wiperite_integrated.py

  # With robot:
  python3 wiperite_integrated.py --robot-ip 192.168.1.50

  # Tune grid density:
  python3 wiperite_integrated.py --robot-ip 192.168.1.50 --grid-cols 4 --grid-rows 3

Browser stream: http://<jetson-ip>:8080
"""

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

try:
    from flask import Flask, Response, jsonify
except ImportError:
    print("Flask not installed. Run: pip3 install flask --break-system-packages")
    sys.exit(1)

# ─────────────────────────── Constants ────────────────────────────────────────

FONT            = cv2.FONT_HERSHEY_SIMPLEX
WIDTH           = 1280
HEIGHT          = 720
PORT            = 8080
CAMERA_INDEXES  = [0, 1]
JPEG_QUALITY    = 80

# EAST text-detection model path. Resolved in this order:
#   1. --east-model command-line flag
#   2. EAST_MODEL environment variable
#   3. the default below: a 'models/' folder next to this script
# The .pb file is NOT committed to git (see .gitignore); download it and place
# it accordingly, or pass --east-model. See README for the download link.
_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EAST_MODEL = os.environ.get(
    'EAST_MODEL',
    os.path.join(_SCRIPT_DIR, 'models', 'frozen_east_text_detection.pb'),
)

MIN_AREA_RATIO  = 0.08
MAX_AREA_RATIO  = 0.95

ROBOT_PORT      = 5000
RECONNECT_DELAY = 3.0       # seconds between TCP reconnect attempts

# ─────────────────────────── Thread-safe frame buffer ─────────────────────────

class FrameBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._jpeg  = None
        self._error = ''

    def set(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg  = jpeg
            self._error = ''

    def set_error(self, err: str) -> None:
        with self._lock:
            self._error = err

    def get(self):
        with self._lock:
            return self._jpeg, self._error


# ─────────────────────────── Shared state ─────────────────────────────────────

class SharedState:
    """
    Carries all data produced by the vision pipeline so the command generator
    and the web UI can read it without locks in hot paths.
    """
    def __init__(self):
        self._lock = threading.Lock()

        # Whiteboard geometry (pixel coords, ordered TL/TR/BR/BL)
        self.board_quad   = None    # np.array shape (4,2) or None
        self.board_w_px   = 0
        self.board_h_px   = 0

        # Board physical size hint (mm) — set from CLI or measured by robot
        self.board_w_mm   = 1200   # default 120 cm
        self.board_h_mm   = 900    # default 90 cm

        # Grid parameters
        self.grid_cols    = 4
        self.grid_rows    = 3

        # OCR results: list of (u_norm, v_norm, w_norm, h_norm, text)
        # u,v are centre coords in [0,1] relative to board; w,h are extents
        self.text_boxes   = []

        # Command queue: list of @WPE1 wire strings ready to send
        self.cmd_queue    = []

        # Status
        self.board_found  = False
        self.last_cmd_sent = ''

    # ── helpers ──────────────────────────────────────────────────────────────

    def update_board(self, quad, w_px, h_px):
        with self._lock:
            self.board_quad  = quad
            self.board_w_px  = int(w_px)
            self.board_h_px  = int(h_px)
            self.board_found = quad is not None

    def update_text_boxes(self, boxes):
        with self._lock:
            self.text_boxes = boxes

    def enqueue_cmd(self, wire: str):
        with self._lock:
            self.cmd_queue.append(wire)

    def dequeue_cmd(self):
        with self._lock:
            if self.cmd_queue:
                return self.cmd_queue.pop(0)
            return None

    def snapshot(self):
        with self._lock:
            return {
                'board_found'  : self.board_found,
                'board_w_px'   : self.board_w_px,
                'board_h_px'   : self.board_h_px,
                'board_w_mm'   : self.board_w_mm,
                'board_h_mm'   : self.board_h_mm,
                'grid_cols'    : self.grid_cols,
                'grid_rows'    : self.grid_rows,
                'text_boxes'   : list(self.text_boxes),
                'last_cmd_sent': self.last_cmd_sent,
                'queue_depth'  : len(self.cmd_queue),
            }


# ─────────────────────────── Geometry helpers ─────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype=np.float32)


def px_dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def pixel_to_board_uv(px: float, py: float, quad: np.ndarray,
                       board_w_px: int, board_h_px: int):
    """
    Map a pixel coordinate inside the board quad to normalised (u,v) ∈ [0,1].
    Uses a simple perspective-corrected affine via getPerspectiveTransform.
    Returns (u, v) or None if transform fails.
    """
    if quad is None or board_w_px == 0 or board_h_px == 0:
        return None
    dst = np.array([
        [0,           0          ],
        [board_w_px,  0          ],
        [board_w_px,  board_h_px ],
        [0,           board_h_px ],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    pt = np.array([[[px, py]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, M)
    u = float(out[0, 0, 0]) / board_w_px
    v = float(out[0, 0, 1]) / board_h_px
    return u, v


# ─────────────────────────── Whiteboard detection ─────────────────────────────

def preprocess_for_board(frame: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  kernel, iterations=1)
    edges  = cv2.Canny(blur, 50, 150)
    combined = cv2.bitwise_or(bright, edges)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


def find_whiteboard(frame: np.ndarray):
    """
    Returns (ordered_quad_or_None, (w_px, h_px)).
    """
    mask = preprocess_for_board(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w      = frame.shape[:2]
    frame_area = h * w
    min_area  = frame_area * MIN_AREA_RATIO
    max_area  = frame_area * MAX_AREA_RATIO

    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            rect  = cv2.minAreaRect(cnt)
            box   = cv2.boxPoints(rect)
            approx = np.int32(box).reshape(-1, 1, 2)

        pts = approx.reshape(-1, 2)
        if len(pts) != 4:
            continue

        ordered = order_points(pts)
        w_top  = px_dist(ordered[0], ordered[1])
        w_bot  = px_dist(ordered[3], ordered[2])
        h_left = px_dist(ordered[0], ordered[3])
        h_right= px_dist(ordered[1], ordered[2])
        w_px   = (w_top + w_bot) / 2.0
        h_px   = (h_left + h_right) / 2.0

        if w_px < 100 or h_px < 100:
            continue
        aspect = w_px / max(h_px, 1.0)
        if aspect < 0.5 or aspect > 3.5:
            continue

        if area > best_area:
            best_area = area
            best_quad = ordered

    if best_quad is None:
        return None, (0, 0)

    tl, tr, br, bl = best_quad
    w_px = int((px_dist(tl, tr) + px_dist(bl, br)) / 2)
    h_px = int((px_dist(tl, bl) + px_dist(tr, br)) / 2)
    return best_quad, (w_px, h_px)


# ─────────────────────────── OCR helpers ──────────────────────────────────────

def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def run_tesseract(img: np.ndarray) -> str:
    if shutil.which('tesseract') is None:
        return ''
    tmp = tempfile.mkdtemp()
    try:
        in_path  = os.path.join(tmp, 'in.png')
        out_base = os.path.join(tmp, 'out')
        cv2.imwrite(in_path, img)
        subprocess.run(
            ['tesseract', in_path, out_base, '--psm', '7', '--oem', '3'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        txt = out_base + '.txt'
        if os.path.exists(txt):
            with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        return ''
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def detect_text_boxes(frame: np.ndarray, net):
    """EAST text detection → list of (x1,y1,x2,y2) pixel boxes."""
    H, W  = frame.shape[:2]
    rW, rH = W / 320.0, H / 320.0
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (320, 320)), 1.0, (320, 320),
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        'feature_fusion/Conv_7/Sigmoid',
        'feature_fusion/concat_3'
    ])

    rects, confs = [], []
    for y in range(scores.shape[2]):
        for x in range(scores.shape[3]):
            score = float(scores[0, 0, y, x])
            if score < 0.5:
                continue
            ox, oy = x * 4.0, y * 4.0
            angle  = geometry[0, 4, y, x]
            cos_, sin_ = np.cos(angle), np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            end_x  = int(ox + cos_ * geometry[0, 1, y, x] + sin_ * geometry[0, 2, y, x])
            end_y  = int(oy - sin_ * geometry[0, 1, y, x] + cos_ * geometry[0, 2, y, x])
            rects.append((int(end_x - w), int(end_y - h), end_x, end_y))
            confs.append(score)

    if not rects:
        return []

    boxes = cv2.dnn.NMSBoxes(
        [(r[0], r[1], r[2]-r[0], r[3]-r[1]) for r in rects],
        confs, 0.5, 0.3
    )
    results = []
    if len(boxes) > 0:
        for i in boxes.flatten():
            x1, y1, x2, y2 = rects[i]
            results.append((
                max(0, int(x1 * rW)), max(0, int(y1 * rH)),
                max(0, int(x2 * rW)), max(0, int(y2 * rH))
            ))
    return results


# ─────────────────────────── Grid command generation ──────────────────────────

def build_wpe1_command(board_mode: int, u_milli: int, v_milli: int,
                        x_mm: int, y_mm: int, w_mm: int, h_mm: int) -> str:
    """
    Construct a @WPE1 wire line understood by wiperite_client / TM4C firmware.

    Format: @WPE1,<board_mode>,<u_milli>,<v_milli>,<x_mm>,<y_mm>,<w_mm>,<h_mm>,E\\n

    board_mode : 0 = absolute mm coords, 1 = normalised UV (u/v in 0-1000)
    u_milli    : horizontal position in 0–1000 (only used when board_mode=1)
    v_milli    : vertical position in 0–1000   (only used when board_mode=1)
    x_mm, y_mm : top-left of board in mm       (only used when board_mode=0)
    w_mm, h_mm : board dimensions in mm
    """
    return (f"@WPE1,{board_mode},{u_milli},{v_milli},"
            f"{x_mm},{y_mm},{w_mm},{h_mm},E\n")


def generate_grid_sweep_commands(state: SharedState) -> list:
    """
    Generate a serpentine (boustrophedon) sweep of the board area,
    skipping cells that already contain detected text (to preserve writing).
    Returns list of @WPE1 wire strings.
    """
    snap = state.snapshot()
    if not snap['board_found']:
        return []

    cols     = snap['grid_cols']
    rows     = snap['grid_rows']
    w_mm     = snap['board_w_mm']
    h_mm     = snap['board_h_mm']
    boxes    = snap['text_boxes']

    cell_w_mm = w_mm / cols
    cell_h_mm = h_mm / rows

    # Build a set of occupied cells from OCR text boxes
    occupied = set()
    for (u, v, uw, vh, text) in boxes:
        if not text:
            continue
        # centre of text box in grid coords
        col = int(u * cols)
        row = int(v * rows)
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        occupied.add((row, col))

    cmds = []
    for row in range(rows):
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_range:
            if (row, col) in occupied:
                continue   # skip — text is here, robot should avoid
            # Centre of this cell in UV milli
            u_milli = int((col + 0.5) / cols * 1000)
            v_milli = int((row + 0.5) / rows * 1000)
            # Absolute mm coords of cell top-left
            x_mm = int(col * cell_w_mm)
            y_mm = int(row * cell_h_mm)
            cmds.append(build_wpe1_command(
                board_mode=1,
                u_milli=u_milli,
                v_milli=v_milli,
                x_mm=x_mm,
                y_mm=y_mm,
                w_mm=int(cell_w_mm),
                h_mm=int(cell_h_mm),
            ))
    return cmds


def generate_erase_text_commands(state: SharedState) -> list:
    """
    For each detected text region, produce a targeted @WPE1 erase command
    that sends the robot to that exact cell.
    """
    snap  = state.snapshot()
    boxes = snap['text_boxes']
    w_mm  = snap['board_w_mm']
    h_mm  = snap['board_h_mm']
    cmds  = []
    for (u, v, uw, vh, text) in boxes:
        if not text:
            continue
        u_milli = int(u * 1000)
        v_milli = int(v * 1000)
        x_mm    = int((u - uw / 2) * w_mm)
        y_mm    = int((v - vh / 2) * h_mm)
        bw_mm   = int(uw * w_mm)
        bh_mm   = int(vh * h_mm)
        cmds.append(build_wpe1_command(
            board_mode=1,
            u_milli=u_milli,
            v_milli=v_milli,
            x_mm=max(0, x_mm),
            y_mm=max(0, y_mm),
            w_mm=max(10, bw_mm),
            h_mm=max(10, bh_mm),
        ))
    return cmds


# ─────────────────────────── TCP robot sender ─────────────────────────────────

class RobotSender(threading.Thread):
    """
    Background thread that drains SharedState.cmd_queue and sends each
    @WPE1 line over TCP to the TM4C/CC3100 robot (wiperite_client protocol).
    Auto-reconnects on drop.
    """
    def __init__(self, ip: str, port: int, state: SharedState, running):
        super().__init__(daemon=True, name='RobotSender')
        self.ip      = ip
        self.port    = port
        self.state   = state
        self.running = running
        self.sock    = None

    def _connect(self) -> bool:
        try:
            s = socket.create_connection((self.ip, self.port), timeout=5)
            s.settimeout(2.0)
            self.sock = s
            print(f'[RobotSender] Connected to {self.ip}:{self.port}')
            return True
        except OSError as e:
            print(f'[RobotSender] Connect failed: {e}')
            self.sock = None
            return False

    def _send(self, wire: str) -> bool:
        if self.sock is None:
            return False
        try:
            self.sock.sendall(wire.encode('ascii'))
            with self.state._lock:
                self.state.last_cmd_sent = wire.strip()
            print(f'[RobotSender] Sent: {wire.strip()}')
            return True
        except OSError as e:
            print(f'[RobotSender] Send error: {e}')
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            return False

    def run(self):
        while self.running[0]:
            if self.sock is None:
                if not self._connect():
                    time.sleep(RECONNECT_DELAY)
                    continue

            cmd = self.state.dequeue_cmd()
            if cmd:
                if not self._send(cmd):
                    # Re-queue if send failed
                    self.state.enqueue_cmd(cmd)
                    time.sleep(0.5)
            else:
                time.sleep(0.05)


# ─────────────────────────── Camera + vision thread ───────────────────────────

def camera_thread(buf: FrameBuffer, state: SharedState, running,
                  east_model_path=DEFAULT_EAST_MODEL):
    # Load EAST if available
    east_net = None
    if os.path.exists(east_model_path):
        try:
            east_net = cv2.dnn.readNet(east_model_path)
            print(f'[Vision] EAST model loaded from {east_model_path}')
        except Exception as e:
            print(f'[Vision] EAST load failed: {e}')
    else:
        print(f'[Vision] EAST model not found at {east_model_path} — text detection disabled')

    # Open camera
    cap, cam_idx = None, None
    for idx in CAMERA_INDEXES:
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            c.set(cv2.CAP_PROP_FPS, 30)
            aw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'[Vision] Camera /dev/video{idx} at {aw}x{ah}')
            cap, cam_idx = c, idx
            break
        c.release()

    if cap is None:
        buf.set_error('No camera found on /dev/video0 or /dev/video1')
        return

    frame_idx      = 0
    detect_interval = 15        # run EAST every N frames
    board_interval  = 5         # run board detect every N frames
    last_text_boxes = []        # cached OCR results
    last_quad       = None

    fps_frames, fps_time, fps_val = 0, time.time(), 0.0

    try:
        while running[0]:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_idx  += 1
            fps_frames += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val    = fps_frames / (now - fps_time)
                fps_frames = 0
                fps_time   = now

            # ── Board detection ───────────────────────────────────────────
            if frame_idx % board_interval == 0:
                quad, (w_px, h_px) = find_whiteboard(frame)
                last_quad = quad
                state.update_board(quad, w_px, h_px)

            # ── EAST + OCR (only inside board bounds) ─────────────────────
            if east_net is not None and frame_idx % detect_interval == 0:
                detect_frame = frame
                if last_quad is not None:
                    # Crop to board bounding rect for faster/more accurate OCR
                    pts  = last_quad.astype(int)
                    x, y, bw, bh = cv2.boundingRect(pts)
                    x  = max(0, x); y = max(0, y)
                    bw = min(frame.shape[1] - x, bw)
                    bh = min(frame.shape[0] - y, bh)
                    if bw > 50 and bh > 50:
                        detect_frame = frame[y:y+bh, x:x+bw]

                raw_boxes = detect_text_boxes(detect_frame, east_net)
                snap = state.snapshot()
                new_boxes = []

                for (rx1, ry1, rx2, ry2) in raw_boxes:
                    # Map back to full-frame coords if we cropped
                    if detect_frame is not frame and last_quad is not None:
                        pts = last_quad.astype(int)
                        ox, oy, _, _ = cv2.boundingRect(pts)
                        rx1 += ox; ry1 += oy; rx2 += ox; ry2 += oy

                    # Expand ROI slightly
                    px1 = max(0, rx1 - 5); py1 = max(0, ry1 - 5)
                    px2 = min(frame.shape[1], rx2 + 5)
                    py2 = min(frame.shape[0], ry2 + 5)
                    roi = frame[py1:py2, px1:px2]
                    if roi.size == 0:
                        continue

                    text = run_tesseract(preprocess_for_ocr(roi)).strip()

                    # Compute board-relative UV for box centre
                    cx_px = (rx1 + rx2) / 2.0
                    cy_px = (ry1 + ry2) / 2.0
                    uv = None
                    if last_quad is not None and snap['board_w_px'] and snap['board_h_px']:
                        uv = pixel_to_board_uv(
                            cx_px, cy_px,
                            last_quad, snap['board_w_px'], snap['board_h_px']
                        )
                    if uv is None:
                        uv = (cx_px / frame.shape[1], cy_px / frame.shape[0])

                    # Normalised width/height relative to board
                    bw_px = snap['board_w_px'] or frame.shape[1]
                    bh_px = snap['board_h_px'] or frame.shape[0]
                    uw = (rx2 - rx1) / bw_px
                    vh = (ry2 - ry1) / bh_px
                    new_boxes.append((uv[0], uv[1], uw, vh, text))

                last_text_boxes = new_boxes
                state.update_text_boxes(new_boxes)

            # ── Draw overlay ──────────────────────────────────────────────
            overlay = frame.copy()

            # Board outline + grid
            if last_quad is not None:
                pts = last_quad.astype(int)
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)

                snap = state.snapshot()
                cols = snap['grid_cols']
                rows = snap['grid_rows']

                # Perspective-warp grid lines onto the frame
                tl, tr, br, bl = last_quad
                for c in range(1, cols):
                    frac = c / cols
                    top_pt  = (tl + (tr - tl) * frac).astype(int)
                    bot_pt  = (bl + (br - bl) * frac).astype(int)
                    cv2.line(overlay, tuple(top_pt), tuple(bot_pt), (0, 180, 255), 1)
                for r in range(1, rows):
                    frac = r / rows
                    left_pt  = (tl + (bl - tl) * frac).astype(int)
                    right_pt = (tr + (br - tr) * frac).astype(int)
                    cv2.line(overlay, tuple(left_pt), tuple(right_pt), (0, 180, 255), 1)

                # Label corners
                labels = ['TL', 'TR', 'BR', 'BL']
                for i, p in enumerate(pts):
                    cv2.circle(overlay, tuple(p), 7, (0, 255, 255), -1)
                    cv2.putText(overlay, labels[i], (p[0]+8, p[1]-8),
                                FONT, 0.55, (255, 255, 0), 2)

                cv2.putText(overlay, 'Board detected',
                            (20, 35), FONT, 0.9, (0, 255, 0), 2)
                cv2.putText(overlay,
                            f'Board: {snap["board_w_px"]}x{snap["board_h_px"]} px  '
                            f'Grid: {cols}x{rows}',
                            (20, 70), FONT, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(overlay, 'Board: not detected',
                            (20, 35), FONT, 0.9, (0, 0, 255), 2)

            # OCR boxes
            for (u, v, uw, vh, text) in last_text_boxes:
                # Convert UV back to pixel for display
                h_f, w_f = frame.shape[:2]
                if last_quad is not None:
                    snap = state.snapshot()
                    bw_px = snap['board_w_px'] or w_f
                    bh_px = snap['board_h_px'] or h_f
                    dst = np.array([
                        [0, 0], [bw_px, 0], [bw_px, bh_px], [0, bh_px]
                    ], dtype=np.float32)
                    M_inv = cv2.getPerspectiveTransform(dst, last_quad.astype(np.float32))
                    cx_b  = u * bw_px
                    cy_b  = v * bh_px
                    half_w = uw * bw_px / 2
                    half_h = vh * bh_px / 2
                    corners_b = np.array([[
                        [cx_b - half_w, cy_b - half_h],
                        [cx_b + half_w, cy_b - half_h],
                        [cx_b + half_w, cy_b + half_h],
                        [cx_b - half_w, cy_b + half_h],
                    ]], dtype=np.float32)
                    corners_f = cv2.perspectiveTransform(corners_b, M_inv)
                    box_pts   = corners_f[0].astype(int)
                    cv2.polylines(overlay, [box_pts], True, (255, 0, 255), 2)
                    ty = box_pts[0, 1] - 8
                    tx = box_pts[0, 0]
                else:
                    cx = int(u * w_f); cy = int(v * h_f)
                    hw = int(uw * w_f / 2); hh = int(vh * h_f / 2)
                    cv2.rectangle(overlay, (cx-hw, cy-hh), (cx+hw, cy+hh), (255, 0, 255), 2)
                    tx, ty = cx - hw, cy - hh - 8

                if text:
                    cv2.putText(overlay, text, (tx, ty if ty > 12 else ty + 24),
                                FONT, 0.5, (0, 0, 0), 3)
                    cv2.putText(overlay, text, (tx, ty if ty > 12 else ty + 24),
                                FONT, 0.5, (255, 0, 255), 1)

            # HUD
            snap = state.snapshot()
            hud  = (f"FPS:{fps_val:.1f}  Cam:/dev/video{cam_idx}  "
                    f"OCR boxes:{len(last_text_boxes)}  "
                    f"Queue:{snap['queue_depth']}  "
                    f"Last:{snap['last_cmd_sent'][:30]}")
            cv2.putText(overlay, hud, (8, frame.shape[0] - 10),
                        FONT, 0.45, (200, 200, 200), 1)

            ok, enc = cv2.imencode('.jpg', overlay,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                buf.set(enc.tobytes())

    except Exception as e:
        buf.set_error(f'Vision thread error: {e}')
        print(f'[Vision] ERROR: {e}')
    finally:
        cap.release()
        print('[Vision] Camera released')


# ─────────────────────────── Flask web app ────────────────────────────────────

def create_app(buf: FrameBuffer, state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.route('/')
    def index():
        return '''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WipeRite Integrated Controller</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background:#111; color:#eee; font-family:Arial,sans-serif; padding:12px; }
h2   { text-align:center; margin-bottom:10px; font-size:1.2em; }
.layout { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
.stream { flex:1; min-width:300px; }
.stream img { width:100%; border:2px solid #444; display:block; }
.panel { width:300px; display:flex; flex-direction:column; gap:10px; }
.card  { background:#1e1e1e; border:1px solid #333; border-radius:6px; padding:10px; }
.card h3 { font-size:.85em; color:#aaa; margin-bottom:6px; text-transform:uppercase; }
.btn { width:100%; padding:8px; border:none; border-radius:4px;
       color:#fff; cursor:pointer; font-size:.9em; margin-top:4px; }
.btn-sweep  { background:#0066cc; }
.btn-erase  { background:#cc4400; }
.btn-clear  { background:#444; }
.btn:hover  { opacity:.85; }
#status { font-size:.78em; color:#8f8; word-break:break-all; }
#boxes  { font-size:.75em; color:#ccc; max-height:120px; overflow-y:auto; }
label   { font-size:.82em; color:#aaa; }
input[type=number] { width:60px; background:#2a2a2a; border:1px solid #555;
                     color:#eee; padding:3px 5px; border-radius:3px; }
</style>
</head>
<body>
<h2>WipeRite Integrated Controller</h2>
<div class="layout">
  <div class="stream">
    <img src="/stream.mjpg" alt="live stream">
  </div>
  <div class="panel">
    <div class="card">
      <h3>Board dimensions (mm)</h3>
      <label>Width <input id="wmm" type="number" value="1200" min="100" max="5000"></label>
      &nbsp;
      <label>Height <input id="hmm" type="number" value="900"  min="100" max="5000"></label>
    </div>
    <div class="card">
      <h3>Grid</h3>
      <label>Cols <input id="gcols" type="number" value="4" min="1" max="20"></label>
      &nbsp;
      <label>Rows <input id="grows" type="number" value="3" min="1" max="20"></label>
    </div>
    <div class="card">
      <h3>Commands</h3>
      <button class="btn btn-sweep"  onclick="sweep()">&#9654; Full Grid Sweep</button>
      <button class="btn btn-erase"  onclick="erase()">&#10005; Erase Text Regions</button>
      <button class="btn btn-clear"  onclick="clearQueue()">&#9632; Clear Queue</button>
    </div>
    <div class="card">
      <h3>Status</h3>
      <div id="status">—</div>
    </div>
    <div class="card">
      <h3>Detected Text Boxes</h3>
      <div id="boxes">—</div>
    </div>
  </div>
</div>
<script>
async function applySettings() {
  const wmm  = document.getElementById('wmm').value;
  const hmm  = document.getElementById('hmm').value;
  const cols = document.getElementById('gcols').value;
  const rows = document.getElementById('grows').value;
  await fetch('/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({board_w_mm:+wmm, board_h_mm:+hmm,
                          grid_cols:+cols, grid_rows:+rows})
  });
}
async function sweep() {
  await applySettings();
  const r = await fetch('/cmd/sweep', {method:'POST'});
  const j = await r.json();
  document.getElementById('status').textContent = j.message;
}
async function erase() {
  await applySettings();
  const r = await fetch('/cmd/erase', {method:'POST'});
  const j = await r.json();
  document.getElementById('status').textContent = j.message;
}
async function clearQueue() {
  const r = await fetch('/cmd/clear', {method:'POST'});
  const j = await r.json();
  document.getElementById('status').textContent = j.message;
}
async function poll() {
  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    const s = document.getElementById('status');
    s.textContent = (j.board_found ? '✓ Board ' + j.board_w_px + 'x' + j.board_h_px + ' px' :
                     '✗ Board not detected') +
                    ' | Queue: ' + j.queue_depth +
                    ' | Last: ' + (j.last_cmd_sent || '—');
    const b = document.getElementById('boxes');
    if (j.text_boxes && j.text_boxes.length) {
      b.innerHTML = j.text_boxes.map(function(tb) {
        return '<div>u=' + tb[0].toFixed(2) + ' v=' + tb[1].toFixed(2) +
               ' "' + (tb[4] || '(no text)') + '"</div>';
      }).join('');
    } else {
      b.textContent = 'None detected';
    }
  } catch(e) {}
}
setInterval(poll, 1000);
poll();
</script>
</body>
</html>'''

    @app.route('/stream.mjpg')
    def stream():
        def generate():
            while True:
                jpeg, err = buf.get()
                if err:
                    time.sleep(0.5)
                    continue
                if jpeg is None:
                    time.sleep(0.03)
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                time.sleep(0.033)
        return Response(generate(),
                        content_type='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/state')
    def api_state():
        return jsonify(state.snapshot())

    @app.route('/settings', methods=['POST'])
    def settings():
        from flask import request
        data = request.get_json(silent=True) or {}
        with state._lock:
            if 'board_w_mm' in data:
                state.board_w_mm = int(data['board_w_mm'])
            if 'board_h_mm' in data:
                state.board_h_mm = int(data['board_h_mm'])
            if 'grid_cols' in data:
                state.grid_cols = int(data['grid_cols'])
            if 'grid_rows' in data:
                state.grid_rows = int(data['grid_rows'])
        return jsonify({'ok': True})

    @app.route('/cmd/sweep', methods=['POST'])
    def cmd_sweep():
        if not state.snapshot()['board_found']:
            return jsonify({'ok': False, 'message': 'Board not detected yet — cannot plan sweep.'})
        cmds = generate_grid_sweep_commands(state)
        for c in cmds:
            state.enqueue_cmd(c)
        return jsonify({'ok': True, 'message': f'Queued {len(cmds)} sweep commands.',
                        'commands': cmds})

    @app.route('/cmd/erase', methods=['POST'])
    def cmd_erase():
        boxes = state.snapshot()['text_boxes']
        if not boxes:
            return jsonify({'ok': False, 'message': 'No text boxes detected to erase.'})
        cmds = generate_erase_text_commands(state)
        for c in cmds:
            state.enqueue_cmd(c)
        return jsonify({'ok': True, 'message': f'Queued {len(cmds)} erase commands.',
                        'commands': cmds})

    @app.route('/cmd/clear', methods=['POST'])
    def cmd_clear():
        with state._lock:
            n = len(state.cmd_queue)
            state.cmd_queue.clear()
        return jsonify({'ok': True, 'message': f'Cleared {n} commands from queue.'})

    @app.route('/healthz')
    def healthz():
        jpeg, err = buf.get()
        return f'ok\nerror={err}\nframe={"yes" if jpeg else "no"}\n'

    return app


# ─────────────────────────── Entry point ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WipeRite Integrated Controller')
    parser.add_argument('--robot-ip',  default='',    help='TM4C/CC3100 IP address')
    parser.add_argument('--robot-port',type=int, default=ROBOT_PORT, help='TCP port (default 5000)')
    parser.add_argument('--port',      type=int, default=PORT,       help='Web server port (default 8080)')
    parser.add_argument('--grid-cols', type=int, default=4,          help='Grid columns (default 4)')
    parser.add_argument('--grid-rows', type=int, default=3,          help='Grid rows (default 3)')
    parser.add_argument('--board-w',   type=int, default=1200,       help='Board width mm (default 1200)')
    parser.add_argument('--board-h',   type=int, default=900,        help='Board height mm (default 900)')
    parser.add_argument('--east-model', default=DEFAULT_EAST_MODEL,
                        help='Path to EAST .pb model (env: EAST_MODEL; '
                             'default: models/ next to this script)')
    args = parser.parse_args()

    print('[Main] WipeRite Integrated Controller starting...')
    if args.robot_ip:
        print(f'[Main] Robot target: {args.robot_ip}:{args.robot_port}')
    else:
        print('[Main] No --robot-ip given — vision/web only, no TCP sending.')
    print(f'[Main] Browser: http://<jetson-ip>:{args.port}')

    buf     = FrameBuffer()
    state   = SharedState()
    running = [True]

    with state._lock:
        state.grid_cols  = args.grid_cols
        state.grid_rows  = args.grid_rows
        state.board_w_mm = args.board_w
        state.board_h_mm = args.board_h

    # Vision thread
    t_vis = threading.Thread(
        target=camera_thread, args=(buf, state, running, args.east_model),
        daemon=True, name='VisionThread'
    )
    t_vis.start()

    # Robot sender thread (only if IP provided)
    t_robot = None
    if args.robot_ip:
        t_robot = RobotSender(args.robot_ip, args.robot_port, state, running)
        t_robot.start()

    app = create_app(buf, state)
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        t_vis.join(2.0)
        if t_robot:
            t_robot.join(2.0)
        print('[Main] Stopped.')


if __name__ == '__main__':
    main()
