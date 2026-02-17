"""
app_config.py

Centralized configuration loader for PacketVision.

Goals:
- Keep all "magic numbers" and paths in one place (config.json + defaults).
- Allow overriding defaults via config.json without breaking missing keys.
- Detect headless mode automatically (no DISPLAY).
- Compute common project paths (script_dir, captures folder, config path).
"""
import os, json
from dataclasses import dataclass
from typing import Any, Dict
def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

DEFAULTS = {
    "preview": {"w": 1280, "h": 720, "fps": 30},
    "capture": {"w": 2592, "h": 1944},
    "fill_green_bgr": [0, 255, 0],
    "roi_trigger": {
        "diff_threshold_l": 22,
        "diff_threshold_ab": 10,
        "roi_ratio_on": 0.60,
        "roi_ratio_off": 0.55,
        "present_frames_n": 4,
        "empty_frames_m": 8,
    },
    "countdown_sec": 5.0,
    "af": {
        "force_single_shot_before_capture": True,
        "wait_timeout_sec": 3.0,
        "poll_interval_sec": 0.05,
        "settle_sec": 0.20,
        "burst_count": 3,
        "burst_gap_sec": 0.10,
        "refocus_on_countdown_start": True,
        "refocus_every_sec_during_countdown": 0.0,
        "discard_main_frames_after_af": 2,
        "discard_main_frame_gap_sec": 0.03,
        "continuous_mode": True,
        "lock_ae_awb_after_reference": True,
    },
    "debug": {"show_preview_diff": True, "max_w": 1200},
    "red_crop": {"work_max_dim": 1200, "pad_ratio": 0.04, "trim_green_border": True},
    "ocr": {
        "min_crop_dim_for_ocr": 600,
        "full_res_upscale_if_small": 2,
        "max_dim": 1400,
        "max_pixels": 1_200_000,
        "jpeg_target_max_bytes": 900_000,
        "jpeg_start_quality": 85,
        "jpeg_min_quality": 45,
        "save_ocr_input": True,
        "min_seconds_between_calls": 35.0,
        "max_calls_per_day": 9,
        "rate_state_path": "/home/droplab/ocr_rate_state.json",
        "tz": "America/Toronto",
    },
    "sheets": {
        "service_account_json": "/home/droplab/Key/service_account.json",
        "spreadsheet_id": "1CN_zg-5KlpK04b5Y-xklb06NVXe3rUKkMUsbnxOXkK8",
        "worksheet_name": "Sheet1",
        "default_machine": "MACHINE_01",
        "default_content": "content",
    },
    "paths": {
        "roi_state_path": "/home/droplab/roi_state.json",
        "ref_lab_path": "/home/droplab/ref_lab_preview.npy",
    },
}

@dataclass(frozen=True)
class AppPaths:
    script_dir: str
    save_dir: str
    config_path: str

@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]
    paths: AppPaths
    headless: bool

    @staticmethod
    def load(script_file: str, config_filename: str = "config.json") -> "AppConfig":
        # headless detection
        headless = not bool(os.environ.get("DISPLAY"))

        script_dir = os.path.dirname(os.path.abspath(script_file))
        save_dir = os.path.join(script_dir, "captures")
        os.makedirs(save_dir, exist_ok=True)

        config_path = os.path.join(script_dir, config_filename)

        # load config override if exists
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            merged = deep_update(DEFAULTS, user_cfg)
        else:
            merged = DEFAULTS

        paths = AppPaths(script_dir=script_dir, save_dir=save_dir, config_path=config_path)
        return AppConfig(raw=merged, paths=paths, headless=headless)

    # small helper getters (optional)
    def get(self, *keys, default=None):
        d = self.raw
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d