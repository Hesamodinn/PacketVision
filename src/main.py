"""
ROI is ONLY a trigger.
Capture FULL-RES reference immediately after ROI confirm.
When ROI changes -> start countdown.
During countdown -> run AF single-shot + wait (ONCE at countdown start), then capture BEST frame from a burst to avoid blur.
Then capture FULL-RES new image.

FINAL STEP:
- Build RED mask on the NEW full-res image (fast on resized copy)
- Find biggest red contour
- Get minAreaRect on that contour
- Warp/Crop from ORIGINAL full-res image (deskewed)
- Empty pixels from warp filled with GREEN (0,255,0)

UPDATED IN THIS VERSION:
✅ Gemini key is ONLY from env var: GEMINI_API_KEY  (no hardcoded key)
✅ Enforce Gemini quotas in code:
   - RPM=2  -> cooldown 35s
   - RPD=9  -> daily cap 9 calls
   Persisted in /home/droplab/ocr_rate_state.json
✅ Retry w/ exponential backoff on 429
✅ OCR payload control:
   - downscale long edge and/or max pixels
   - JPEG compress to keep bytes under target (default <900KB)
✅ Gemini function returns ONLY the clean ID (string) or None
✅ Google Sheet update:
   - Find row where Column A == ID
   - If not found -> message
   - If Column D == "done" -> "packet is done already"
   - If Column D == "receipt" -> update:
       D = content
       H = machine
       I = machine
       O = current Toronto datetime like: 11/4/2025, 4:47:03 PM
✅ Shows a message box in the OpenCV window with ID + sheet result
"""

import os
import time
import cv2
import numpy as np
import textwrap
import json
import re
import random
from datetime import datetime
from zoneinfo import ZoneInfo
from picamera2 import Picamera2
from typing import Optional, Tuple
import google.generativeai as genai
import gspread
from app_config import AppConfig
import threading
from ui_bus import EventBus
import google.api_core.exceptions


###
#HEADLESS = not bool(os.environ.get("DISPLAY"))
#print("[ENV] DISPLAY=", os.environ.get("DISPLAY"), "HEADLESS=", HEADLESS, flush=True)
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#SAVE_DIR = os.path.join(SCRIPT_DIR, "captures")



CFG = AppConfig.load(__file__)  # defaults + config.json override + runtime paths

from zoneinfo import ZoneInfo
import os

from app_config import AppConfig

CFG = AppConfig.load(__file__)  # defaults + config.json override + runtime paths

# --- runtime / env ---
HEADLESS = CFG.headless
print("[ENV] DISPLAY=", os.environ.get("DISPLAY"), "HEADLESS=", HEADLESS, flush=True)

SCRIPT_DIR = CFG.runtime.script_dir
SAVE_DIR   = CFG.runtime.save_dir

os.makedirs(SAVE_DIR, exist_ok=True)

# --- preview (lores) ---
PREVIEW_W  = int(CFG.preview.w)
PREVIEW_H  = int(CFG.preview.h)
TARGET_FPS = int(CFG.preview.fps)

# --- capture (main) ---
CAPTURE_W = int(CFG.capture.w)
CAPTURE_H = int(CFG.capture.h)

# --- fill color ---
FILL_GREEN = tuple(CFG.fill_green_bgr)  # (0,255,0)

# --- ROI trigger ---
DIFF_THRESHOLD_PREVIEW_L  = int(CFG.roi_trigger.diff_threshold_l)
DIFF_THRESHOLD_PREVIEW_AB = int(CFG.roi_trigger.diff_threshold_ab)
ROI_RATIO_ON              = float(CFG.roi_trigger.roi_ratio_on)
ROI_RATIO_OFF             = float(CFG.roi_trigger.roi_ratio_off)
PRESENT_FRAMES_N          = int(CFG.roi_trigger.present_frames_n)
EMPTY_FRAMES_M            = int(CFG.roi_trigger.empty_frames_m)

# --- countdown ---
CAPTURE_COUNTDOWN_SEC = float(CFG.countdown_sec)

# --- AF ---
AF_FORCE_SINGLE_SHOT_BEFORE_CAPTURE = bool(CFG.af.force_single_shot_before_capture)
AF_WAIT_TIMEOUT_SEC   = float(CFG.af.wait_timeout_sec)
AF_POLL_INTERVAL_SEC  = float(CFG.af.poll_interval_sec)
AF_SETTLE_SEC         = float(CFG.af.settle_sec)
CAPTURE_BURST_COUNT   = int(CFG.af.burst_count)
CAPTURE_BURST_GAP_SEC = float(CFG.af.burst_gap_sec)

REFOCUS_ON_COUNTDOWN_START         = bool(CFG.af.refocus_on_countdown_start)
REFOCUS_EVERY_SEC_DURING_COUNTDOWN = float(CFG.af.refocus_every_sec_during_countdown)

DISCARD_MAIN_FRAMES_AFTER_AF = int(CFG.af.discard_main_frames_after_af)
DISCARD_MAIN_FRAME_GAP_SEC   = float(CFG.af.discard_main_frame_gap_sec)

AF_CONTINUOUS_MODE          = bool(CFG.af.continuous_mode)
LOCK_AE_AWB_AFTER_REFERENCE = bool(CFG.af.lock_ae_awb_after_reference)

# --- debug ---
SHOW_PREVIEW_DIFF = bool(CFG.debug.show_preview_diff)
DEBUG_MAX_W       = int(CFG.debug.max_w)

# --- red crop ---
RED_WORK_MAX_DIM  = int(CFG.red_crop.work_max_dim)
RED_PAD_RATIO     = float(CFG.red_crop.pad_ratio)
TRIM_GREEN_BORDER = bool(CFG.red_crop.trim_green_border)

# --- OCR ---
MIN_CROP_DIM_FOR_OCR      = int(CFG.ocr.min_crop_dim_for_ocr)
FULL_RES_UPSCALE_IF_SMALL = int(CFG.ocr.full_res_upscale_if_small)

OCR_MAX_DIM               = int(CFG.ocr.max_dim)
OCR_MAX_PIXELS            = int(CFG.ocr.max_pixels)
OCR_JPEG_TARGET_MAX_BYTES = int(CFG.ocr.jpeg_target_max_bytes)
OCR_JPEG_START_QUALITY    = int(CFG.ocr.jpeg_start_quality)
OCR_JPEG_MIN_QUALITY      = int(CFG.ocr.jpeg_min_quality)
SAVE_OCR_INPUT            = bool(CFG.ocr.save_ocr_input)

OCR_MIN_SECONDS_BETWEEN_CALLS = float(CFG.ocr.min_seconds_between_calls)
OCR_MAX_CALLS_PER_DAY         = int(CFG.ocr.max_calls_per_day)
RATE_STATE_PATH               = str(CFG.ocr.rate_state_path)
TZ                            = ZoneInfo(str(CFG.ocr.tz))

# --- Google Sheets ---
SERVICE_ACCOUNT_JSON = str(CFG.sheets.service_account_json)
SPREADSHEET_ID       = str(CFG.sheets.spreadsheet_id)
WORKSHEET_NAME       = str(CFG.sheets.worksheet_name)
DEFAULT_MACHINE      = str(CFG.sheets.default_machine)
DEFAULT_CONTENT      = str(CFG.sheets.default_content)

# --- persisted paths ---
ROI_STATE_PATH = CFG.paths.roi_state_path
REF_LAB_PATH   = str(CFG.raw.paths.ref_lab_path)


# =========================
# ROI Picker (OpenCV)
# =========================
drawing = False
ix, iy = -1, -1
ROI = None  # (x,y,w,h)
def save_roi_and_reference(rx, ry, rw, rh, ref_lab_preview: np.ndarray) -> None:
    state = {
        "preview_w": PREVIEW_W,
        "preview_h": PREVIEW_H,
        "rx": int(rx), "ry": int(ry), "rw": int(rw), "rh": int(rh),
        "saved_at": datetime.now(TZ).isoformat(),
    }
    tmp = ROI_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, ROI_STATE_PATH)

    np.save(REF_LAB_PATH, ref_lab_preview)  # saves as .npy
    print(f"[STATE] ROI+REF saved -> {ROI_STATE_PATH} and {REF_LAB_PATH}")


def load_roi_and_reference():
    if (not os.path.exists(ROI_STATE_PATH)) or (not os.path.exists(REF_LAB_PATH)):
        return None

    try:
        with open(ROI_STATE_PATH, "r") as f:
            st = json.load(f)

        # Safety: if resolution changed since last save, force ROI picker again
        if st.get("preview_w") != PREVIEW_W or st.get("preview_h") != PREVIEW_H:
            print("[STATE] Saved ROI was for different PREVIEW size -> need reselect ROI.")
            return None

        rx = int(st["rx"]); ry = int(st["ry"]); rw = int(st["rw"]); rh = int(st["rh"])
        ref_lab_preview = np.load(REF_LAB_PATH)

        # Basic validity checks
        if ref_lab_preview is None or getattr(ref_lab_preview, "size", 0) == 0:
            return None

        return (rx, ry, rw, rh, ref_lab_preview)

    except Exception as e:
        print(f"[STATE] Failed to load ROI/REF: {e}")
        return None


def clear_saved_roi_and_reference():
    for p in (ROI_STATE_PATH, REF_LAB_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    print("[STATE] Cleared saved ROI/REF.")

def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, ROI
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ROI = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ROI = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ROI = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))

def clamp_roi(roi, W, H):
    x, y, w, h = roi
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

# =========================
# UI helpers
# =========================
def draw_text_box(img, text, x=10, y=10, w=None, line_h=24, alpha=0.65, max_lines=7):
    if w is None:
        w = img.shape[1] - 20

    wrap_width = max(20, int(w / 16))
    lines = []
    for raw_line in str(text).splitlines():
        wrapped = textwrap.wrap(raw_line, width=wrap_width) or [""]
        lines.extend(wrapped)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["..."]

    hh = line_h * (len(lines) + 1) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + hh), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    yy = y + line_h
    for ln in lines:
        cv2.putText(out, ln, (x + 10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        yy += line_h
    return out

def show_crop_and_text_window(crop_bgr: np.ndarray, text: str, win="OCR_RESULT", max_w=1200):
    """Show image on top + text below in a single OpenCV window."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return

    disp = crop_bgr.copy()
    h, w = disp.shape[:2]
    if w > max_w:
        s = max_w / float(w)
        disp = cv2.resize(disp, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

    pad = 12
    line_h = 28
    panel_h = 260
    panel = np.full((panel_h, disp.shape[1], 3), (0, 0, 0), dtype=np.uint8)

    wrap_width = max(20, int(disp.shape[1] / 14))
    lines = []
    for raw in str(text).splitlines():
        lines += textwrap.wrap(raw, width=wrap_width) or [""]

    max_lines = (panel_h - 2 * pad) // line_h
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["..."]

    y = pad + line_h
    for ln in lines:
        cv2.putText(panel, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += line_h

    out = np.vstack([disp, panel])
    cv2.imshow(win, out)
    cv2.waitKey(1)

# =========================
# OCR payload helpers
# =========================
def downscale_for_ocr(gray: np.ndarray, max_dim: int, max_pixels: int) -> np.ndarray:
    if gray is None or getattr(gray, "size", 0) == 0:
        return gray
    h, w = gray.shape[:2]
    scale_dim = min(1.0, max_dim / float(max(h, w)))
    scale_pix = min(1.0, (max_pixels / float(h * w)) ** 0.5)
    scale = min(scale_dim, scale_pix)
    if scale >= 1.0:
        return gray
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

def encode_jpeg_under(gray: np.ndarray, max_bytes: int, start_q: int, min_q: int):
    if gray is None or getattr(gray, "size", 0) == 0:
        return None, None, 0

    q = int(start_q)
    best = None
    while q >= int(min_q):
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            break
        data = buf.tobytes()
        best = (data, q, len(data))
        if len(data) <= max_bytes:
            return best
        q -= 5

    return best if best else (None, None, 0)

def build_ocr_payload(gray: np.ndarray):
    if gray is None or getattr(gray, "size", 0) == 0:
        return None, None, None, None

    g = downscale_for_ocr(gray, OCR_MAX_DIM, OCR_MAX_PIXELS)
    jpeg_bytes, q, sz = encode_jpeg_under(g, OCR_JPEG_TARGET_MAX_BYTES, OCR_JPEG_START_QUALITY, OCR_JPEG_MIN_QUALITY)
    if jpeg_bytes is None:
        return None, None, None, None

    if sz > OCR_JPEG_TARGET_MAX_BYTES:
        g2 = downscale_for_ocr(g, max_dim=int(OCR_MAX_DIM * 0.80), max_pixels=int(OCR_MAX_PIXELS * 0.75))
        jpeg_bytes2, q2, sz2 = encode_jpeg_under(
            g2,
            OCR_JPEG_TARGET_MAX_BYTES,
            start_q=max(70, OCR_JPEG_START_QUALITY - 10),
            min_q=max(40, OCR_JPEG_MIN_QUALITY - 5),
        )
        if jpeg_bytes2 is not None and sz2 <= sz:
            g, jpeg_bytes, q, sz = g2, jpeg_bytes2, q2, sz2

    return jpeg_bytes, "image/jpeg", q, g

# =========================
# Gemini quota helpers
# =========================
def _load_rate_state():
    if not os.path.exists(RATE_STATE_PATH):
        return {"date": None, "count": 0, "last_ts": 0.0}
    try:
        with open(RATE_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "count": 0, "last_ts": 0.0}

def _save_rate_state(st):
    tmp = RATE_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, RATE_STATE_PATH)

def _can_call_gemini():
    st = _load_rate_state()
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if st.get("date") != today:
        st = {"date": today, "count": 0, "last_ts": 0.0}

    if st["count"] >= OCR_MAX_CALLS_PER_DAY:
        return False, f"Daily limit reached ({st['count']}/{OCR_MAX_CALLS_PER_DAY})"

    now = time.time()
    if now - st["last_ts"] < OCR_MIN_SECONDS_BETWEEN_CALLS:
        wait = OCR_MIN_SECONDS_BETWEEN_CALLS - (now - st["last_ts"])
        return False, f"Cooldown active. Wait {wait:.1f}s"

    return True, "OK"

def _mark_gemini_called():
    st = _load_rate_state()
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if st.get("date") != today:
        st = {"date": today, "count": 0, "last_ts": 0.0}
    st["count"] += 1
    st["last_ts"] = time.time()
    _save_rate_state(st)

def _gemini_call_with_backoff(model, parts, generation_config, max_retries=4):
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return model.generate_content(parts, generation_config=generation_config)
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg) or ("resource exhausted" in msg) or ("quota" in msg):
                if attempt == max_retries:
                    raise
                time.sleep(delay + random.uniform(0, 0.7))
                delay = min(delay * 2.0, 30.0)
                continue
            raise


# =========================
# Gemini OCR (returns ONLY ID)
# =========================
def extract_id_from_packet(img_gray_or_bgr: np.ndarray) -> Tuple[Optional[str], str]:
    """
    Returns (found_id, status)

    found_id:
      - str (clean ID) if found
      - None if not found or error

    status:
      - "OK"
      - "NOT_FOUND"
      - "ERROR: <message>"
    """

    # Recommended: keep your key in an env var instead of hardcoding
    api_key = str(CFG.api_key)
    if not api_key:
        return None, "ERROR: GEMINI_API_KEY not set."

    genai.configure(api_key=api_key)

    if img_gray_or_bgr is None or getattr(img_gray_or_bgr, "size", 0) == 0:
        return None, "ERROR: No image provided."

    # Ensure GRAY
    if len(img_gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(img_gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_gray_or_bgr

    # --- ADDED: RESIZE IMAGE TO SAVE TOKENS ---
    # High-res images consume massive TPM (Tokens Per Minute).
    # 1600px is a good sweet spot for OCR accuracy vs. token cost.
    h, w = gray.shape[:2]
    if max(h, w) > 2048:
        scale = 2048 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    payload_bytes, mime, used_q, used_gray = build_ocr_payload(gray)
    if payload_bytes is None:
        return None, "ERROR: Failed to build OCR payload."

    print(f"[OCR SEND] bytes={len(payload_bytes)} q={used_q} dim={used_gray.shape[1]}x{used_gray.shape[0]}")

    prompt = """
Extract the primary ID from this logistics document (shipping label or packing slip).

Logic:
1) Search for an Amazon Order ID (Format: 000-0000000-0000000).
2) If not found, search for a Tracking ID. Look for keywords like 'TRK#', 'Tracking',
   'Waybill', 'Shipment ID', or carrier-specific formats (e.g., UPS starting with '1Z').

Return a JSON object:
{
  "found": true/false,
  "type": "amazon_order_id" OR "tracking_number" OR "none",
  "id": "clean_id_value_here"
}

Rules:
- If nothing is found: "found": false and "id": null.
- Keep Amazon Order ID hyphens (000-0000000-0000000).
- For tracking numbers: remove spaces and punctuation, keep letters/numbers only.
"""

    # --- REPLACED: RETRY LOOP TO HANDLE 429 ERROR ---
    max_retries = 3
    raw = ""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            image_part = {"mime_type": mime, "data": payload_bytes}

            resp = model.generate_content(
                [prompt, image_part],
                generation_config={"response_mime_type": "application/json"},
            )

            raw = (resp.text or "").strip()
            break  # success -> exit retry loop

        except google.api_core.exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 10s, 20s, ...
                print(f"[OCR 429] Quota hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            return None, f"ERROR: Quota exhausted after {max_retries} attempts."
        except Exception as e:
            return None, f"ERROR: {e}"

    # --- PROCEED WITH YOUR JSON PARSING (your original logic) ---
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", raw, flags=re.DOTALL)
        if not m:
            return None, f"ERROR: Gemini returned non-JSON: {raw[:200]}"
        data = json.loads(m.group(0))

    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), None)
        if data is None:
            return None, f"ERROR: Gemini returned JSON list but no object: {raw[:200]}"

    if not isinstance(data, dict):
        return None, f"ERROR: Gemini returned unexpected JSON type: {type(data).__name__}"

    found = bool(data.get("found", False))
    if not found:
        return None, "NOT_FOUND"

    raw_id = data.get("id", None)
    typ = str(data.get("type", "none")).strip()

    if raw_id is None:
        return None, "NOT_FOUND"

    raw_id = str(raw_id).strip()

    if typ == "amazon_order_id":
        clean_id = re.sub(r"[^0-9\-]", "", raw_id)
    else:
        clean_id = re.sub(r"[^A-Za-z0-9]", "", raw_id)

    if not clean_id:
        return None, "NOT_FOUND"

    return clean_id, "OK"


def make_ocr_input_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# =========================
# Trigger helpers (lores ROI) - shadow suppressed
# =========================
def make_ref_lab_preview(roi_bgr):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    return lab

def roi_change_ratio_preview_shadow_suppressed(roi_bgr, ref_lab, thr_L, thr_AB):
    cur_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    cur_lab = cv2.GaussianBlur(cur_lab, (5, 5), 0)

    L0, A0, B0 = cv2.split(ref_lab)
    L1, A1, B1 = cv2.split(cur_lab)

    dL = cv2.absdiff(L1, L0)
    dA = cv2.absdiff(A1, A0)
    dB = cv2.absdiff(B1, B0)
    dAB = cv2.add(dA, dB)

    mask_L = (dL > thr_L).astype(np.uint8) * 255
    mask_AB = (dAB > thr_AB).astype(np.uint8) * 255

    e0 = cv2.Canny(L0, 40, 120)
    e1 = cv2.Canny(L1, 40, 120)
    e_diff = cv2.bitwise_xor(e0, e1)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    e_diff = cv2.dilate(e_diff, k, iterations=1)

    mask = cv2.bitwise_or(mask_AB, cv2.bitwise_and(mask_L, e_diff))
    ratio = cv2.countNonZero(mask) / float(mask.size)
    return ratio, mask

# =========================
# Camera (Picamera2)
# =========================
def init_camera():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (CAPTURE_W, CAPTURE_H), "format": "RGB888"},
        lores={"size": (PREVIEW_W, PREVIEW_H), "format": "RGB888"},
        controls={"FrameDurationLimits": (int(1e6 / TARGET_FPS), int(1e6 / TARGET_FPS))}
    )
    picam2.configure(config)
    apply_controls(picam2, af_enabled=AF_CONTINUOUS_MODE, ae_awb_lock=False, lock_values=None)
    picam2.start()
    time.sleep(0.25)
    return picam2

def apply_controls(picam2, af_enabled=True, ae_awb_lock=False, lock_values=None):
    controls = {}
    controls["AfMode"] = 2 if af_enabled else 0  # 2=continuous

    if ae_awb_lock and lock_values:
        controls["AeEnable"] = False
        controls["AwbEnable"] = False
        controls["ExposureTime"] = int(lock_values.get("ExposureTime", 10000))
        controls["AnalogueGain"] = float(lock_values.get("AnalogueGain", 2.0))
        if "ColourGains" in lock_values:
            controls["ColourGains"] = lock_values["ColourGains"]
    else:
        controls["AeEnable"] = True
        controls["AwbEnable"] = True

    try:
        picam2.set_controls(controls)
    except Exception as e:
        print(f"[WARN] Some camera controls unsupported: {e}")

def capture_lock_values(picam2):
    try:
        md = picam2.capture_metadata()
        lock_values = {
            "ExposureTime": md.get("ExposureTime", 10000),
            "AnalogueGain": md.get("AnalogueGain", 2.0),
        }
        if "ColourGains" in md:
            lock_values["ColourGains"] = md["ColourGains"]
        return lock_values
    except Exception as e:
        print(f"[WARN] Could not read metadata for AE/AWB lock: {e}")
        return None

def laplacian_sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def af_single_shot_and_wait(picam2, timeout_sec=3.0, poll_sec=0.05, settle_sec=0.2):
    t0 = time.time()
    ok_trigger = False

    for c in (
        {"AfMode": 1, "AfTrigger": 1},
        {"AfMode": 1, "AfTrigger": 0},
        {"AfMode": 1},
        {"AfTrigger": 1},
    ):
        try:
            picam2.set_controls(c)
            ok_trigger = True
            break
        except Exception:
            continue

    focused = False
    while (time.time() - t0) < timeout_sec:
        try:
            md = picam2.capture_metadata()
            if "AfState" in md:
                s = md["AfState"]
                if isinstance(s, int) and s == 2:
                    focused = True
                    break
                if isinstance(s, str) and ("focus" in s.lower() and "ed" in s.lower()):
                    focused = True
                    break
        except Exception:
            pass
        time.sleep(poll_sec)

    time.sleep(settle_sec)
    return ok_trigger, focused

def discard_main_frames(picam2, n=2, gap_sec=0.03):
    for _ in range(max(0, int(n))):
        _ = picam2.capture_array("main")
        if gap_sec and gap_sec > 0:
            time.sleep(gap_sec)

def capture_main_best_frame(picam2, burst_count=3, gap_sec=0.10):
    best = None
    best_score = -1.0
    for _ in range(max(1, int(burst_count))):
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        score = laplacian_sharpness(bgr)
        if score > best_score:
            best_score = score
            best = bgr
        if gap_sec and gap_sec > 0:
            time.sleep(gap_sec)
    return best, best_score

# =========================
# Biggest red rotated rectangle crop + bbox
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def resize_max_dim(img: np.ndarray, max_dim: int = 1200):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    scale = max_dim / float(m)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def red_mask_fast(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,   20,  10], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 20,  10], dtype=np.uint8)
    upper2 = np.array([179,255,255], dtype=np.uint8)

    mask_hsv = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    b, g, r = cv2.split(bgr)
    mask_rgb = ((r > g + 25) & (r > b + 25) & (r > 60)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask_hsv, mask_rgb)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def pad_box_points(box, img_w, img_h, pad_ratio=0.04):
    box = box.astype(np.float32)
    cx = np.mean(box[:, 0])
    cy = np.mean(box[:, 1])
    scale = 1.0 + pad_ratio
    box[:, 0] = (box[:, 0] - cx) * scale + cx
    box[:, 1] = (box[:, 1] - cy) * scale + cy
    box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
    return box

def warp_min_area_rect(img_bgr, box_pts, border_value=(0,255,0)):
    src = order_points(box_pts)
    (tl, tr, br, bl) = src

    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    maxW = max(maxW, 2)
    maxH = max(maxH, 2)

    dst = np.array([[0, 0],
                    [maxW - 1, 0],
                    [maxW - 1, maxH - 1],
                    [0, maxH - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img_bgr, M, (maxW, maxH),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped

def trim_fill_border(bgr: np.ndarray, fill_bgr=(0,255,0), tol=10):
    if bgr is None:
        return None
    diff = cv2.absdiff(bgr, np.array(fill_bgr, dtype=np.uint8))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > tol)
    if len(xs) == 0 or len(ys) == 0:
        return bgr
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return bgr[y0:y1+1, x0:x1+1]

def bbox_from_box_points(box_pts, W, H):
    xs = box_pts[:, 0]
    ys = box_pts[:, 1]
    x0 = int(max(0, np.floor(xs.min())))
    y0 = int(max(0, np.floor(ys.min())))
    x1 = int(min(W - 1, np.ceil(xs.max())))
    y1 = int(min(H - 1, np.ceil(ys.max())))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    return x0, y0, bw, bh

def crop_biggest_red_rotated_keep_full_rectangle(img_bgr, work_max_dim=1200, pad_ratio=0.04, trim=True):
    if img_bgr is None:
        return None, None, None

    small, scale = resize_max_dim(img_bgr, max_dim=work_max_dim)
    mask_red = red_mask_fast(small)

    cnts = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    if not contours:
        return None, None, None

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box_s = cv2.boxPoints(rect).astype(np.float32)
    box_full = box_s / scale

    H, W = img_bgr.shape[:2]
    box_full = pad_box_points(box_full, W, H, pad_ratio=pad_ratio)

    crop_warped = warp_min_area_rect(img_bgr, box_full, border_value=FILL_GREEN)
    crop_final = trim_fill_border(crop_warped, fill_bgr=FILL_GREEN) if trim else crop_warped

    bbox_xywh = bbox_from_box_points(box_full, W, H)
    return crop_final, box_full, bbox_xywh

def upscale2x_and_recrop_if_small(new_full_bgr: np.ndarray, crop_final_bgr: np.ndarray):
    if crop_final_bgr is None or getattr(crop_final_bgr, "size", 0) == 0:
        return None, None, False

    ch, cw = crop_final_bgr.shape[:2]
    if cw >= MIN_CROP_DIM_FOR_OCR and ch >= MIN_CROP_DIM_FOR_OCR:
        return crop_final_bgr, None, False

    up = int(FULL_RES_UPSCALE_IF_SMALL)
    H, W = new_full_bgr.shape[:2]
    new_full_2x = cv2.resize(new_full_bgr, (W * up, H * up), interpolation=cv2.INTER_CUBIC)

    crop2, box2, bbox2 = crop_biggest_red_rotated_keep_full_rectangle(
        new_full_2x,
        work_max_dim=RED_WORK_MAX_DIM,
        pad_ratio=RED_PAD_RATIO,
        trim=TRIM_GREEN_BORDER
    )

    if crop2 is None or crop2.size == 0 or bbox2 is None:
        return crop_final_bgr, None, False

    x2, y2, w2, h2 = bbox2
    x = int(round(x2 / up))
    y = int(round(y2 / up))
    w = int(round(w2 / up))
    h = int(round(h2 / up))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    bbox_orig = (x, y, w, h)

    return crop2, bbox_orig, True

def save_yolo_bbox(label_path, img_w, img_h, bbox_xywh, cls_id=0):
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / float(img_w)
    cy = (y + h / 2.0) / float(img_h)
    nw = w / float(img_w)
    nh = h / float(img_h)

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"ABS_XYWH {x} {y} {w} {h}\n")
        f.write(f"YOLO {cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
def ask_yes_no(title: str, message: str) -> bool:
    """
    Returns True if user clicks Yes, False if No.
    Uses Tkinter messagebox (real popup).
    """
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()          # hide main window
    root.attributes("-topmost", True)
    answer = messagebox.askyesno(title, message)
    root.destroy()
    return bool(answer)

# =========================
# Google Sheets helpers
# =========================
def normalize_id_for_db(s: str) -> str:
    # keep only letters/numbers, uppercase for consistency
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()
def now_toronto_str() -> str:
    dt = datetime.now(TZ)
    # Example: 11/4/2025, 4:47:03 PM  (no leading zeros)
    return f"{dt.month}/{dt.day}/{dt.year}, {dt.strftime('%I:%M:%S %p').lstrip('0')}"
def add_row_if_user_confirms(
    spreadsheet_id: str,
    worksheet_name: str,
    id_value_ui: str,   # keep original (with - or spaces) for UI
    id_value_db: str,   # normalized for sheet/db (example: 122312)
    content: str,
    machine: str,
    service_account_json: str,
) -> dict:
    """
    Appends a NEW row to the sheet (your 'db') when user clicks YES.

    Writes:
      A = id_value_db  (normalized)
      D = content
      H = machine
      I = machine
      O = current Toronto datetime

    Returns dict:
      ok, added, reason
    """
    try:
        gc = gspread.service_account(filename=service_account_json)
        ws = gc.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    except Exception as e:
        return {"ok": False, "added": False, "reason": f"Sheets open/auth error: {e}"}

    # Determine how many columns exist so we can place D/H/I/O correctly
    # Use header row length if exists; fallback to 15 columns (A..O)
    try:
        header = ws.row_values(1)
        total_cols = max(len(header), 15)
    except Exception:
        total_cols = 15

    row = [""] * total_cols

    # Columns: A=1, D=4, H=8, I=9, O=15
    row[0]  = str(normalize_id_for_db(id_value_db))         # A
    row[3]  = str(content)             # D
    row[7]  = str(machine)             # H
    row[8]  = str(machine)             # I
    row[14] = now_toronto_str()        # O

    try:
        ws.append_row(row, value_input_option="USER_ENTERED")
        return {
            "ok": True,
            "added": True,
            "reason": f"Added new row for ID '{id_value_ui}' (stored as '{id_value_db}')."
        }
    except Exception as e:
        return {"ok": False, "added": False, "reason": f"Append row failed: {e}"}
def read_row_by_id_and_update_if_receipt(
    spreadsheet_id: str,
    worksheet_name: str,
    id_value: str,
    content: str,
    machine: str,
    id_col: int = 1,
):
    """
    Returns dict with keys:
      ok: bool (operation success)
      found: bool (ID exists in sheet)
      updated: bool
      reason: str
      row: int|None
    """
    try:
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_JSON)
        ws = gc.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    except Exception as e:
        return {"ok": False, "found": False, "updated": False, "row": None,
                "reason": f"Sheets auth/open error: {e}"}

    # Search in column A (id_col=1 by default)
    col_values = ws.col_values(id_col)
    try:
        row_index = col_values.index(normalize_id_for_db(id_value)) + 1  # Sheets are 1-based
    except ValueError:
        return {"ok": True, "found": False, "updated": False, "row": None,
                "reason": f"ID not found: {id_value}"}

    # Now ID exists
    d_value_raw = (ws.cell(row_index, 4).value or "").strip()
    d_value = d_value_raw.lower()

    if d_value == "done":
        return {"ok": True, "found": True, "updated": False, "row": row_index,
                "reason": f"Packet is DONE already "}

    if d_value != "receipt":
        return {"ok": True, "found": True, "updated": False, "row": row_index,
                "reason": "This ID already exists"}

    delivery_time = now_toronto_str()

    # Update D/H/I/O in one call: range D..O (12 cols)
    cells = ws.range(row_index, 4, row_index, 15)  # D..O (12 cells)

    # cells list order is left->right: D,E,F,G,H,I,J,K,L,M,N,O
    cells[0].value  = content        # D
    cells[4].value  = machine        # H
    cells[5].value  = machine        # I
    cells[11].value = delivery_time  # O

    ws.update_cells(cells, value_input_option="USER_ENTERED")

    return {"ok": True, "found": True, "updated": True, "row": row_index,
            "reason": f"information updated  sucessfully."}


def show_loading(frame_bgr, rx, ry, rw, rh, text="LOADING... PLEASE WAIT"):
    """Draws a big loading overlay and refreshes the OpenCV window immediately."""
    show = frame_bgr.copy()

    # ROI rectangle
    cv2.rectangle(show, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 3)

    # Dark overlay bar
    h, w = show.shape[:2]
    overlay = show.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    show = cv2.addWeighted(overlay, 0.65, show, 0.35, 0)

    # Big text
    cv2.putText(show, text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("capture", show)
    cv2.waitKey(1)  # forces UI refresh

# =========================
# Main
# =========================
def main():
    global ROI

    picam2 = init_camera()
    af_enabled = AF_CONTINUOUS_MODE
    ae_awb_locked = False
    lock_values = None

    # Try load saved ROI + reference
    loaded = load_roi_and_reference()
    if loaded:
        rx, ry, rw, rh, ref_lab_preview = loaded
        print("[STATE] Loaded saved ROI + reference. Skipping ROI picker and reference capture.")
        cv2.namedWindow("capture")
    else:
        # ---- ROI Picker ----
        cv2.namedWindow("ROI Picker")
        cv2.setMouseCallback("ROI Picker", mouse_cb)
        print("Drag ROI. ENTER=confirm (capture FULL-RES reference). ESC=quit.")

        while True:
            rgb = picam2.capture_array("lores")
            lores = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            disp = lores.copy()
            cv2.putText(disp, "Drag ROI. ENTER=confirm. ESC=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if ROI is not None:
                x0, y0, w0, h0 = ROI
                cv2.rectangle(disp, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)

            cv2.imshow("ROI Picker", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                ROI = None
                break
            elif k == 13 and ROI is not None and ROI[2] > 10 and ROI[3] > 10:
                break

        cv2.destroyWindow("ROI Picker")
        if ROI is None:
            cv2.destroyAllWindows()
            picam2.stop()
            raise SystemExit("No ROI selected.")

        rx, ry, rw, rh = clamp_roi(ROI, PREVIEW_W, PREVIEW_H)
        print("ROI (preview):", (rx, ry, rw, rh))

        cv2.namedWindow("capture")

        # ---- Capture FULL-RES reference (not saved) ----
        print("[REF] Capturing FULL-RES reference (best-of-burst)...")
        if af_enabled and AF_FORCE_SINGLE_SHOT_BEFORE_CAPTURE:
            af_single_shot_and_wait(picam2, AF_WAIT_TIMEOUT_SEC, AF_POLL_INTERVAL_SEC, AF_SETTLE_SEC)
            try:
                picam2.set_controls({"AfMode": 2})
            except Exception:
                pass
            discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)

        ref_full, ref_score = capture_main_best_frame(picam2, CAPTURE_BURST_COUNT, CAPTURE_BURST_GAP_SEC)
        print("[REF] captured (not saved)", f"(sharpness={ref_score:.1f})")

        rgb_ref_lores = picam2.capture_array("lores")
        ref_lores = cv2.cvtColor(rgb_ref_lores, cv2.COLOR_RGB2BGR)
        ref_lab_preview = make_ref_lab_preview(ref_lores[ry:ry + rh, rx:rx + rw].copy())

        # Save ROI + reference so next run skips this
        save_roi_and_reference(rx, ry, rw, rh, ref_lab_preview)

        if LOCK_AE_AWB_AFTER_REFERENCE:
            lock_values = capture_lock_values(picam2)
            if lock_values:
                ae_awb_locked = True
                apply_controls(picam2, af_enabled=af_enabled, ae_awb_lock=True, lock_values=lock_values)
                print("[LOCK] AE/AWB locked after reference.")


    # ---- Trigger state ----
    armed = True
    occupied = False
    present_count = 0
    empty_count = 0
    countdown_active = False
    countdown_start = 0.0
    last_text = ""

    print("Running. ESC quit | r recapture ref | c change ROI/ref | f AF | l lock")

    while True:
        rgb = picam2.capture_array("lores")
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        roi_now = frame[ry:ry + rh, rx:rx + rw]
        ratio, diff_bin = roi_change_ratio_preview_shadow_suppressed(
            roi_now, ref_lab_preview, DIFF_THRESHOLD_PREVIEW_L, DIFF_THRESHOLD_PREVIEW_AB
        )

        if not occupied:
            occupied = ratio > ROI_RATIO_ON
        else:
            occupied = ratio > ROI_RATIO_OFF

        if occupied:
            present_count += 1
            empty_count = 0
        else:
            empty_count += 1
            present_count = 0

        # Start countdown
        if (not countdown_active) and armed and present_count >= PRESENT_FRAMES_N:
            countdown_active = True
            countdown_start = time.time()

            if af_enabled and REFOCUS_ON_COUNTDOWN_START:
                ok, focused = af_single_shot_and_wait(
                    picam2,
                    timeout_sec=AF_WAIT_TIMEOUT_SEC,
                    poll_sec=AF_POLL_INTERVAL_SEC,
                    settle_sec=AF_SETTLE_SEC
                )
                try:
                    picam2.set_controls({"AfMode": 2})
                except Exception:
                    pass
                discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)
                last_text = f"AF start ok={ok} focused={focused}"

        if countdown_active:
            remaining = CAPTURE_COUNTDOWN_SEC - (time.time() - countdown_start)

            if remaining <= 0:
                show_loading(frame, rx, ry, rw, rh, "CAPTURING / OCR... PLEASE WAIT")
                ts = int(time.time())
                print("[NEW] Capturing FULL-RES new image (AF wait + best-of-burst)...")

                if af_enabled and AF_FORCE_SINGLE_SHOT_BEFORE_CAPTURE:
                    ok, focused = af_single_shot_and_wait(
                        picam2,
                        timeout_sec=AF_WAIT_TIMEOUT_SEC,
                        poll_sec=AF_POLL_INTERVAL_SEC,
                        settle_sec=AF_SETTLE_SEC
                    )
                    try:
                        picam2.set_controls({"AfMode": 2})
                    except Exception:
                        pass
                    discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)
                    last_text = f"AF final ok={ok} focused={focused}"

                new_full, new_score = capture_main_best_frame(picam2, CAPTURE_BURST_COUNT, CAPTURE_BURST_GAP_SEC)

                new_path = os.path.join(SAVE_DIR, f"new_full_{ts}.jpg")
                cv2.imwrite(new_path, new_full)
                print("[NEW] saved:", new_path, f"(sharpness={new_score:.1f})")

                crop_final, box_pts, bbox_xywh = crop_biggest_red_rotated_keep_full_rectangle(
                    new_full,
                    work_max_dim=RED_WORK_MAX_DIM,
                    pad_ratio=RED_PAD_RATIO,
                    trim=TRIM_GREEN_BORDER
                )

                if crop_final is None or crop_final.size == 0:
                    last_text = "No red region found -> crop failed."
                    print("[CROP] Failed: no red contour.")
                else:
                    crop_for_ocr, bbox_override, did_recrop = upscale2x_and_recrop_if_small(new_full, crop_final)
                    bbox_xywh_used = bbox_override if bbox_override is not None else bbox_xywh

                    if did_recrop:
                        print("[OCR] crop was small -> recropped from 2x full image:",
                              crop_for_ocr.shape[1], "x", crop_for_ocr.shape[0])

                    ocr_input_gray = make_ocr_input_gray(crop_for_ocr)
                    if ocr_input_gray is None:
                        last_text = "OCR input build failed."
                    else:
                        # Save the exact bytes we send (optional)
                        ocr_input_gray, rot_k = best_text_orientation_gray(ocr_input_gray)

                        payload_bytes, mime, used_q, used_gray = build_ocr_payload(ocr_input_gray)
                        if payload_bytes is None:
                            last_text = "OCR payload build failed."
                        else:
                            
                            ocr_input_path = os.path.join(SAVE_DIR, f"ocr_input_sent_{ts}.jpg")
                            with open(ocr_input_path, "wb") as f:
                                f.write(payload_bytes)
                            print("[OCR INPUT] saved:", ocr_input_path,
                                    f"bytes={len(payload_bytes)} q={used_q} dim={used_gray.shape[1]}x{used_gray.shape[0]}")

                            label_path = os.path.join(SAVE_DIR, f"new_full_{ts}.txt")
                            H, W = new_full.shape[:2]
                            save_yolo_bbox(label_path, W, H, bbox_xywh_used, cls_id=0)
                            print("[YOLO] bbox saved:", label_path, "bbox=", bbox_xywh_used)

                            # Gemini OCR (returns ONLY id)
                            found_id, ocr_status = extract_id_from_packet(used_gray)

                            if not found_id:
                                # ocr_status: "NOT_FOUND" or "ERROR: ..."
                                if ocr_status == "NOT_FOUND":
                                    msg = "OCR: ID not found in image."
                                else:
                                    msg = f"OCR: {ocr_status}"   # already contains ERROR: ...
                                #show_crop_and_text_window(
                                #    cv2.cvtColor(used_gray, cv2.COLOR_GRAY2BGR),
                                #    msg,
                                #    win="OCR_RESULT",
                                #    max_w=DEBUG_MAX_W
                                #)
                                last_text = msg
                            else:
                                found_id_ui = str(found_id).strip()          # keep original for UI
                                found_id_db = normalize_id_for_db(found_id_ui)  
                                sheet_res = read_row_by_id_and_update_if_receipt(
                                    spreadsheet_id=SPREADSHEET_ID,
                                    worksheet_name=WORKSHEET_NAME,
                                    id_value=found_id_db,
                                    content=DEFAULT_CONTENT,
                                    machine=DEFAULT_MACHINE,
                                    id_col=1
                                )

                                # --- If ID NOT FOUND -> popup ---
                                if isinstance(sheet_res, dict) and sheet_res.get("found") is False:
                                    details = (
                                        f"OCR ID: {found_id}\n\n"
                                        "This ID does not exist in the sheet.\n"
                                        "Are you sure the number is correct?"
                                    )
                                    user_yes = ask_yes_no("ID Not Found", details)

                                    if user_yes:
                                        msg = f"Confirmed ID: {found_id}\n(Do Method A here)"
                                        # do_method_A(found_id)
                                        added_res = add_row_if_user_confirms(
                                                spreadsheet_id=SPREADSHEET_ID,
                                                worksheet_name=WORKSHEET_NAME,
                                                id_value_ui=found_id_ui,     # e.g. "1223-12"
                                                id_value_db=normalize_id_for_db(found_id_db),     # e.g. "122312" (normalized)
                                                content=DEFAULT_CONTENT,
                                                machine=DEFAULT_MACHINE,
                                                service_account_json=SERVICE_ACCOUNT_JSON,
                                            )
                                        msg = added_res.get("reason", str(added_res))
                                    else:
                                        msg = f"Rejected ID: {found_id}\n(Do Method B here)"
                                        # do_method_B(found_id)

                                    #show_crop_and_text_window(
                                    #    cv2.cvtColor(used_gray, cv2.COLOR_GRAY2BGR),
                                    #    msg,
                                    #    win="OCR_RESULT",
                                    #    max_w=DEBUG_MAX_W
                                    #)
                                    last_text = msg

                                else:
                                    # Normal path (found or other status)
                                    sheet_msg = sheet_res.get("reason", str(sheet_res)) if isinstance(sheet_res, dict) else str(sheet_res)
                                    msg = f"ID: {found_id}\n{sheet_msg}"
                                    #show_crop_and_text_window(
                                    #    cv2.cvtColor(used_gray, cv2.COLOR_GRAY2BGR),
                                    #    msg,
                                    #    win="OCR_RESULT",
                                    #    max_w=DEBUG_MAX_W
                                    #)
                                    last_text = msg


                                

                countdown_active = False
                armed = False

        if (not countdown_active) and (not armed) and empty_count >= EMPTY_FRAMES_M:
            armed = True

        # ---- UI ----
        show = frame.copy()
        rect_color = (0, 255, 0) if armed else (0, 0, 255)
        cv2.rectangle(show, (rx, ry), (rx + rw, ry + rh), rect_color, 2)

        status = "ROI_CHANGED" if occupied else "ROI_STABLE"
        info = (
            f"{status} ratio={ratio:.3f} armed={armed} present={present_count}\n"
            f"ESC quit | r ref | f AF | l lock\n"
            f"{last_text}"
        )
        show = draw_text_box(show, info, x=10, y=10, w=show.shape[1] - 20, max_lines=7)

        if countdown_active:
            rem = max(0.0, CAPTURE_COUNTDOWN_SEC - (time.time() - countdown_start))
            cv2.putText(show, f"CAPTURE IN {rem:.1f}s",
                        (510, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.imshow("capture", show)
        #if SHOW_PREVIEW_DIFF:
        #    cv2.imshow("roi_diff_preview", diff_bin)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('f'):
            af_enabled = not af_enabled
            apply_controls(picam2, af_enabled=af_enabled, ae_awb_lock=ae_awb_locked, lock_values=lock_values)
            last_text = f"AF={af_enabled}"
        elif k == ord('c'):
            # Change ROI + reference (force reselect)
            clear_saved_roi_and_reference()
            cv2.destroyAllWindows()
            picam2.stop()
            return main()  # restart flow fresh
        elif k == ord('l'):
            ae_awb_locked = not ae_awb_locked
            if ae_awb_locked:
                if not lock_values:
                    lock_values = capture_lock_values(picam2)
                apply_controls(picam2, af_enabled=af_enabled, ae_awb_lock=True, lock_values=lock_values)
                last_text = "AE/AWB locked"
            else:
                apply_controls(picam2, af_enabled=af_enabled, ae_awb_lock=False, lock_values=None)
                last_text = "AE/AWB unlocked"
        elif k == ord('r'):
            last_text = "Recapturing FULL-RES reference..."
            if af_enabled and AF_FORCE_SINGLE_SHOT_BEFORE_CAPTURE:
                af_single_shot_and_wait(picam2, AF_WAIT_TIMEOUT_SEC, AF_POLL_INTERVAL_SEC, AF_SETTLE_SEC)
                try:
                    picam2.set_controls({"AfMode": 2})
                except Exception:
                    pass
                discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)

            ref_full, ref_score = capture_main_best_frame(picam2, CAPTURE_BURST_COUNT, CAPTURE_BURST_GAP_SEC)

            rgb_ref_lores = picam2.capture_array("lores")
            ref_lores = cv2.cvtColor(rgb_ref_lores, cv2.COLOR_RGB2BGR)
            ref_lab_preview = make_ref_lab_preview(ref_lores[ry:ry + rh, rx:rx + rw].copy())

            if LOCK_AE_AWB_AFTER_REFERENCE:
                lock_values = capture_lock_values(picam2)
                if lock_values:
                    ae_awb_locked = True
                    apply_controls(picam2, af_enabled=af_enabled, ae_awb_lock=True, lock_values=lock_values)

            armed = True
            occupied = False
            present_count = 0
            empty_count = 0
            countdown_active = False
            last_text = f"Reference updated (sharpness={ref_score:.1f})."

    cv2.destroyAllWindows()
    picam2.stop()
def rotate_gray(img, k):
    # k: 0,1,2,3 => 0°,90°,180°,270° clockwise
    if k % 4 == 0: return img
    if k % 4 == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k % 4 == 2: return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def best_text_orientation_gray(gray: np.ndarray):
    """
    Cheap heuristic: pick rotation that maximizes horizontal text-like edges.
    Returns (rotated_gray, k)
    """
    best_k = 0
    best_score = -1.0

    for k in (0, 1, 2, 3):
        g = rotate_gray(gray, k)
        g2 = cv2.GaussianBlur(g, (3, 3), 0)
        edges = cv2.Canny(g2, 60, 180)
        score = float(np.sum(edges))  # simple but works ok

        if score > best_score:
            best_score = score
            best_k = k

    return rotate_gray(gray, best_k), best_k

def run_workflow(bus: "EventBus", stop_event: threading.Event):
    """
    Runs your existing logic but:
    - no cv2.imshow / no cv2.waitKey
    - emits preview/crop/log/step/result to CustomTkinter
    - reacts to UI commands via bus events (optional)
    """
    global ROI

    def log(level, msg):
        bus.emit("log", level=level, msg=msg)

    # commands coming from UI (toggle AF, recapture ref, change ROI)
    pending_cmds = []

    # helper: drain command events fast
    def drain_commands():
        while True:
            try:
                ev = bus.q.get_nowait()
            except Exception:
                break
            if ev.type == "command":
                pending_cmds.append(ev.data.get("name"))
            else:
                # not a command -> put back for UI thread
                bus.q.put(ev)
                break

    try:
        bus.emit("step", index=0, status="Idle / Waiting")
        bus.emit("status", text="Initializing camera...")
        picam2 = init_camera()
        log("INFO", "Camera initialized")

        af_enabled = AF_CONTINUOUS_MODE
        ae_awb_locked = False
        lock_values = None

        # Try load saved ROI + reference
        loaded = load_roi_and_reference()
        if loaded:
            rx, ry, rw, rh, ref_lab_preview = loaded
            log("INFO", "Loaded saved ROI + reference. Skipping ROI picker.")
        else:
            # IMPORTANT:
            # For best UX: implement ROI selection in Tk later.
            # For now: force ROI to be pre-saved OR provide a simple fallback:
            raise SystemExit("No saved ROI. Please run ROI selection setup first (we’ll move ROI picker into Tk next).")

        # Trigger state
        armed = True
        occupied = False
        present_count = 0
        empty_count = 0
        countdown_active = False
        countdown_start = 0.0
        last_text = ""

        bus.emit("status", text="Running")
        bus.emit("step", index=0, status="Running")

        last_preview_push = 0.0

        while not stop_event.is_set():
            drain_commands()

            # Handle commands
            while pending_cmds:
                cmd = pending_cmds.pop(0)
                if cmd == "toggle_af":
                    af_enabled = not af_enabled
                    apply_controls(picam2, af_enabled=af_enabled, ae_awb_lock=ae_awb_locked, lock_values=lock_values)
                    log("INFO", f"AF set to {af_enabled}")
                elif cmd == "recapture_ref":
                    log("INFO", "Recapturing reference...")
                    # your existing 'r' logic but no imshow
                    if af_enabled and AF_FORCE_SINGLE_SHOT_BEFORE_CAPTURE:
                        af_single_shot_and_wait(picam2, AF_WAIT_TIMEOUT_SEC, AF_POLL_INTERVAL_SEC, AF_SETTLE_SEC)
                        try: picam2.set_controls({"AfMode": 2})
                        except Exception: pass
                        discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)

                    ref_full, ref_score = capture_main_best_frame(picam2, CAPTURE_BURST_COUNT, CAPTURE_BURST_GAP_SEC)
                    rgb_ref_lores = picam2.capture_array("lores")
                    ref_lores = cv2.cvtColor(rgb_ref_lores, cv2.COLOR_RGB2BGR)
                    ref_lab_preview = make_ref_lab_preview(ref_lores[ry:ry + rh, rx:rx + rw].copy())
                    save_roi_and_reference(rx, ry, rw, rh, ref_lab_preview)
                    log("INFO", f"Reference updated (sharpness={ref_score:.1f})")
                elif cmd == "change_roi":
                    log("WARN", "Change ROI requested (not implemented in Tk yet).")
                    clear_saved_roi_and_reference()
                    raise SystemExit("ROI cleared. Implement ROI selection UI next.")

            rgb = picam2.capture_array("lores")
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # push preview to UI at ~10 fps (avoid heavy UI load)
            now = time.time()
            if now - last_preview_push > 0.10:
                # draw ROI rect in the preview image (for UX)
                disp = frame.copy()
                rect_color = (0, 255, 0) if armed else (0, 0, 255)
                cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), rect_color, 2)
                bus.emit("preview", bgr=disp)
                last_preview_push = now

            roi_now = frame[ry:ry + rh, rx:rx + rw]
            ratio, diff_bin = roi_change_ratio_preview_shadow_suppressed(
                roi_now, ref_lab_preview, DIFF_THRESHOLD_PREVIEW_L, DIFF_THRESHOLD_PREVIEW_AB
            )

            if not occupied:
                occupied = ratio > ROI_RATIO_ON
            else:
                occupied = ratio > ROI_RATIO_OFF

            if occupied:
                present_count += 1
                empty_count = 0
            else:
                empty_count += 1
                present_count = 0

            if (not countdown_active) and armed and present_count >= PRESENT_FRAMES_N:
                countdown_active = True
                countdown_start = time.time()
                bus.emit("step", index=3, status="Countdown + AF (once)")
                log("INFO", "Countdown started")

                if af_enabled and REFOCUS_ON_COUNTDOWN_START:
                    ok, focused = af_single_shot_and_wait(picam2, AF_WAIT_TIMEOUT_SEC, AF_POLL_INTERVAL_SEC, AF_SETTLE_SEC)
                    try: picam2.set_controls({"AfMode": 2})
                    except Exception: pass
                    discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)
                    log("INFO", f"AF start ok={ok} focused={focused}")

            if countdown_active:
                remaining = CAPTURE_COUNTDOWN_SEC - (time.time() - countdown_start)
                bus.emit("status", text=f"Capture in {max(0.0, remaining):.1f}s")

                if remaining <= 0:
                    ts = int(time.time())
                    bus.emit("step", index=4, status="Capture Burst (Best Frame)")
                    log("INFO", "Capturing new full-res (best of burst)...")

                    if af_enabled and AF_FORCE_SINGLE_SHOT_BEFORE_CAPTURE:
                        ok, focused = af_single_shot_and_wait(picam2, AF_WAIT_TIMEOUT_SEC, AF_POLL_INTERVAL_SEC, AF_SETTLE_SEC)
                        try: picam2.set_controls({"AfMode": 2})
                        except Exception: pass
                        discard_main_frames(picam2, DISCARD_MAIN_FRAMES_AFTER_AF, DISCARD_MAIN_FRAME_GAP_SEC)
                        log("INFO", f"AF final ok={ok} focused={focused}")

                    new_full, new_score = capture_main_best_frame(picam2, CAPTURE_BURST_COUNT, CAPTURE_BURST_GAP_SEC)
                    new_path = os.path.join(SAVE_DIR, f"new_full_{ts}.jpg")
                    cv2.imwrite(new_path, new_full)
                    bus.emit("step", index=5, status="New Full-Res Image Saved")
                    log("INFO", f"Saved {new_path} (sharpness={new_score:.1f})")

                    bus.emit("step", index=6, status="Red Crop (Deskew)")
                    crop_final, box_pts, bbox_xywh = crop_biggest_red_rotated_keep_full_rectangle(
                        new_full, work_max_dim=RED_WORK_MAX_DIM, pad_ratio=RED_PAD_RATIO, trim=TRIM_GREEN_BORDER
                    )

                    if crop_final is None or crop_final.size == 0:
                        log("ERROR", "Crop failed: no red contour")
                        bus.emit("result", ocr="—", sheet="Crop failed")
                    else:
                        bus.emit("crop", bgr=crop_final)

                        crop_for_ocr, bbox_override, did_recrop = upscale2x_and_recrop_if_small(new_full, crop_final)
                        ocr_input_gray = make_ocr_input_gray(crop_for_ocr)

                        if ocr_input_gray is None:
                            log("ERROR", "OCR input build failed")
                            bus.emit("result", ocr="—", sheet="OCR input failed")
                        else:
                            ocr_input_gray, rot_k = best_text_orientation_gray(ocr_input_gray)
                            # show the rotated OCR image in UI (convert gray -> BGR for display)
                            bus.emit("crop", bgr=cv2.cvtColor(ocr_input_gray, cv2.COLOR_GRAY2BGR))
                            log("INFO", f"OCR rotation k={rot_k} (0,1,2,3 => 0/90/180/270 clockwise)")
                      
                            payload_bytes, mime, used_q, used_gray = build_ocr_payload(ocr_input_gray)
                            if payload_bytes is None:
                                log("ERROR", "OCR payload build failed")
                                bus.emit("result", ocr="—", sheet="OCR payload failed")
                            else:
                                label_path = os.path.join(SAVE_DIR, f"new_full_{ts}.txt")
                                H, W = new_full.shape[:2]
                                save_yolo_bbox(label_path, W, H, bbox_override if bbox_override else bbox_xywh, cls_id=0)
                                
                                # --- SAVE the exact OCR image we send to Gemini (debug / audit) ---
                                ocr_sent_path = os.path.join(SAVE_DIR, f"ocr_input_sent_{ts}.jpg")
                                with open(ocr_sent_path, "wb") as f:
                                    f.write(payload_bytes)
                                log("INFO", f"OCR input saved: {ocr_sent_path} bytes={len(payload_bytes)} q={used_q} dim={used_gray.shape[1]}x{used_gray.shape[0]}")
                                bus.emit("step", index=7, status="OCR (Gemini)")
                                found_id, ocr_status = extract_id_from_packet(used_gray)

                                if not found_id:
                                    msg = "ID not found" if ocr_status == "NOT_FOUND" else ocr_status
                                    log("WARN", f"OCR: {msg}")
                                    bus.emit("result", ocr="NOT_FOUND", sheet=msg)
                                else:
                                    bus.emit("result", ocr=str(found_id), sheet="Updating sheet...")
                                    bus.emit("step", index=8, status="Google Sheets Update")

                                    found_id_db = normalize_id_for_db(found_id)
                                    sheet_res = read_row_by_id_and_update_if_receipt(
                                        spreadsheet_id=SPREADSHEET_ID,
                                        worksheet_name=WORKSHEET_NAME,
                                        id_value=found_id_db,
                                        content=DEFAULT_CONTENT,
                                        machine=DEFAULT_MACHINE,
                                        id_col=1
                                    )

                                    if isinstance(sheet_res, dict) and sheet_res.get("found") is False:
                                        # In pro UX: show dialog in main window later. For now keep your popup:
                                        details = (
                                            f"OCR ID: {found_id}\n\n"
                                            "This ID does not exist in the sheet.\n"
                                            "Are you sure the number is correct?"
                                        )
                                        user_yes = ask_yes_no("ID Not Found", details)
                                        if user_yes:
                                            added_res = add_row_if_user_confirms(
                                                spreadsheet_id=SPREADSHEET_ID,
                                                worksheet_name=WORKSHEET_NAME,
                                                id_value_ui=str(found_id),
                                                id_value_db=normalize_id_for_db(found_id_db),
                                                content=DEFAULT_CONTENT,
                                                machine=DEFAULT_MACHINE,
                                                service_account_json=SERVICE_ACCOUNT_JSON,
                                            )
                                            sheet_msg = added_res.get("reason", str(added_res))
                                        else:
                                            sheet_msg = "Rejected by user"
                                    else:
                                        sheet_msg = sheet_res.get("reason", str(sheet_res)) if isinstance(sheet_res, dict) else str(sheet_res)

                                    bus.emit("step", index=9, status="Result UI Message")
                                    bus.emit("result", ocr=str(found_id), sheet=sheet_msg)
                                    log("INFO", f"Result: ID={found_id} | {sheet_msg}")

                    countdown_active = False
                    armed = False

            if (not countdown_active) and (not armed) and empty_count >= EMPTY_FRAMES_M:
                armed = True
                bus.emit("step", index=0, status="Re-armed / waiting")

        # end while

        picam2.stop()
        bus.emit("stopped")

    except Exception as e:
        try:
            bus.emit("log", level="ERROR", msg=str(e))
            bus.emit("result", ocr="—", sheet=f"ERROR: {e}")
            bus.emit("stopped")
        except Exception:
            pass
if __name__ == "__main__":
    from ui_ctk import PacketVisionApp  # your CustomTkinter window class
    from ui_bus import EventBus
    import threading

    bus = EventBus()
    stop_event = threading.Event()

    # start worker thread
    t = threading.Thread(target=run_workflow, args=(bus, stop_event), daemon=True)
    t.start()

    # start GUI (main thread)
    app = PacketVisionApp(bus, stop_event=stop_event)
    app.mainloop()
