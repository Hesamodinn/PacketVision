# ui_ctk.py
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2

from ui_bus import EventBus, UIEvent

STEPS = [
    ("Idle / Waiting", "System running, ROI monitoring."),
    ("ROI Confirm (Trigger Setup)", "ROI selection → capture full-res reference."),
    ("ROI Change Detected", "Presence frames reached → start countdown."),
    ("Countdown + AF (once)", "At countdown start: AF single-shot + settle (ONCE)."),
    ("Capture Burst (Best Frame)", "Capture burst → keep sharpest frame."),
    ("New Full-Res Image Saved", "Save new full-res + bbox file."),
    ("Red Crop (Deskew)", "Red mask → biggest contour → minAreaRect → warp/crop."),
    ("OCR (Gemini)", "Quota-safe OCR → returns clean ID or None."),
    ("Google Sheets Update", "Find row → apply rules → update fields."),
    ("Result UI Message", "Show ID + sheet result; reset arming after empty frames."),
]

def bgr_to_photoimage(bgr: np.ndarray, max_w: int, max_h: int):
    """Convert BGR numpy image to Tk PhotoImage (keeps aspect ratio)."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    scale = min(max_w / float(w), max_h / float(h), 1.0)
    nw, nh = int(w * scale), int(h * scale)
    if nw < 2 or nh < 2:
        return None

    if scale < 1.0:
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    im = Image.fromarray(rgb)
    return ImageTk.PhotoImage(im)

class PacketVisionApp(ctk.CTk):
    def __init__(self, bus: EventBus, start_worker_fn):
        super().__init__()
        self.bus = bus
        self.start_worker_fn = start_worker_fn

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("PacketVision — Workflow Viewer")
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()-90
        self.geometry(f"{sw}x{sh}+0+0")
        # Main layout: left workspace + right steps (steps full height)
        self.grid_rowconfigure(0, weight=1)
        #self.grid_columnconfigure(0, weight=7)   # workspace
        #self.grid_columnconfigure(1, weight=3)   # steps
        self.grid_columnconfigure(0, weight=3)   # steps (left)
        self.grid_columnconfigure(1, weight=7)   # workspace (right)

        # ======================
        # LEFT WORKSPACE
        # ======================
        #workspace = ctk.CTkFrame(self, corner_radius=16)
        #workspace.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        workspace = ctk.CTkFrame(self, corner_radius=16)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        workspace.grid_rowconfigure(1, weight=1)
        workspace.grid_columnconfigure(0, weight=3)  # preview column
        workspace.grid_columnconfigure(1, weight=2)  # crop column

        # Header across both columns
        header = ctk.CTkFrame(workspace, corner_radius=16)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            header, text="Status: Idle", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=10, pady=6)

        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="e", padx=10, pady=6)

        self.btn_start = ctk.CTkButton(btns, text="Start", command=self.on_start)
        self.btn_start.grid(row=0, column=0, padx=6)

        self.btn_stop = ctk.CTkButton(btns, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=6)

        self.btn_toggle_af = ctk.CTkButton(btns, text="Toggle AF", command=self.on_toggle_af, state="disabled")
        self.btn_toggle_af.grid(row=0, column=2, padx=6)

        self.btn_recapture_ref = ctk.CTkButton(btns, text="Recapture Ref", command=self.on_recapture_ref, state="disabled")
        self.btn_recapture_ref.grid(row=0, column=3, padx=6)

        self.btn_change_roi = ctk.CTkButton(btns, text="Change ROI", command=self.on_change_roi, state="disabled")
        self.btn_change_roi.grid(row=0, column=4, padx=6)

        # ---- Preview column: Preview on top, Logs below (NOT full width) ----
        preview_col = ctk.CTkFrame(workspace, corner_radius=16)
        preview_col.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 10))
        preview_col.grid_rowconfigure(1, weight=1)  # preview image grows
        preview_col.grid_rowconfigure(3, weight=0)  # logs fixed
        preview_col.grid_columnconfigure(0, weight=1)

        # Title row + light dot
        title_row = ctk.CTkFrame(preview_col, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_row, text="Live Preview", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.light_dot = ctk.CTkFrame(
            title_row, width=16, height=16, corner_radius=99, fg_color="#ff3344"
        )
        self.light_dot.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.light_dot.grid_propagate(False)

        self.preview_img_label = ctk.CTkLabel(preview_col, text="")
        self.preview_img_label.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 8))

        # Logs (below preview, not full width)
        ctk.CTkLabel(
            preview_col, text="Logs", font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(4, 6))

        self.log_box = ctk.CTkTextbox(preview_col, height=140)
        self.log_box.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.log_box.configure(state="disabled")

        # ---- Crop column: Crop + Result ----
        crop_col = ctk.CTkFrame(workspace, corner_radius=16)
        crop_col.grid(row=1, column=1, sticky="nsew", padx=(6, 10), pady=(0, 10))
        crop_col.grid_rowconfigure(1, weight=1)
        crop_col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            crop_col, text="Latest Crop (deskew)", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.crop_img_label = ctk.CTkLabel(crop_col, text="")
        self.crop_img_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Result background + bigger text
        self.result_bg = ctk.CTkFrame(crop_col, corner_radius=12, fg_color="#0b1220")
        self.result_bg.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.result_bg.grid_columnconfigure(0, weight=1)

        self.result_label = ctk.CTkLabel(
            self.result_bg,
            text="OCR: —\nSheet: —",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white",
            wraplength=520,
        )
        self.result_label.grid(row=0, column=0, sticky="ew", padx=14, pady=14)

        # ======================
        # RIGHT: WORKFLOW STEPS (FULL HEIGHT)
        # ======================
        #right = ctk.CTkFrame(self, corner_radius=16)
        #right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right = ctk.CTkFrame(self, corner_radius=16)
        right.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(4, weight=1)  # steps scroll uses all height

        ctk.CTkLabel(
            right, text="Workflow Steps", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self.step_badge = ctk.CTkLabel(right, text="Step 0 / 0", text_color="#9fb2c8")
        self.step_badge.grid(row=1, column=0, sticky="w", padx=12)

        self.progress = ctk.CTkProgressBar(right)
        self.progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 10))
        self.progress.set(0)

        self.active_step_title = ctk.CTkLabel(right, text="—", font=ctk.CTkFont(size=14, weight="bold"))
        self.active_step_title.grid(row=3, column=0, sticky="w", padx=12)

        self.steps_scroll = ctk.CTkScrollableFrame(right, corner_radius=16)
        self.steps_scroll.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 12))

        self.step_rows = []
        for i, (name, hint) in enumerate(STEPS, start=1):
            row = ctk.CTkFrame(self.steps_scroll, corner_radius=12)
            row.pack(fill="x", pady=6)

            n = ctk.CTkLabel(row, text=str(i), width=26)
            n.pack(side="left", padx=8, pady=10)

            txt = ctk.CTkFrame(row, fg_color="transparent")
            txt.pack(side="left", fill="x", expand=True, padx=0, pady=8)
            ctk.CTkLabel(txt, text=name, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(txt, text=hint, text_color="#9fb2c8", wraplength=320, justify="left").pack(anchor="w")

            self.step_rows.append(row)

        # ============ State ============
        self.worker_running = False
        self.last_preview_photo = None
        self.last_crop_photo = None
        self.current_step = 0
        self.stop_request = threading.Event()

        # poll events
        self.after(50, self.poll_events)

        # auto-start by default
        self.after(300, self.on_start)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- helpers ----------
    def log(self, line: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_light(self, is_green: bool):
        self.light_dot.configure(fg_color="#00ff66" if is_green else "#ff3344")

    def set_step(self, idx_0_based: int, message: str = ""):
        total = len(STEPS)
        self.current_step = max(0, min(idx_0_based, total - 1))
        self.step_badge.configure(text=f"Step {self.current_step + 1} / {total}")
        self.progress.set((self.current_step + 1) / total)
        self.active_step_title.configure(text=STEPS[self.current_step][0])

        # Light rule: Step 1 (index 0) -> green, otherwise red
        self._set_light(is_green=(self.current_step == 0))

        for i, row in enumerate(self.step_rows):
            if i < self.current_step:
                row.configure(fg_color="#1a2a1f")   # done
            elif i == self.current_step:
                row.configure(fg_color="#1a2433")   # active
            else:
                row.configure(fg_color="transparent")

        if message:
            self.status_label.configure(text=f"Status: {message}")

    # ---------- buttons ----------
    def on_start(self):
        if self.worker_running:
            return
        self.stop_request.clear()
        self.worker_running = True

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_toggle_af.configure(state="normal")
        self.btn_recapture_ref.configure(state="normal")
        self.btn_change_roi.configure(state="normal")

        self.log("=== START ===")
        self.start_worker_fn(self.bus, self.stop_request)

    def on_stop(self):
        if not self.worker_running:
            return
        self.log("Stop requested...")
        self.stop_request.set()

    def on_toggle_af(self):
        self.bus.emit("command", name="toggle_af")

    def on_recapture_ref(self):
        self.bus.emit("command", name="recapture_ref")

    def on_change_roi(self):
        self.bus.emit("command", name="change_roi")

    def on_close(self):
        self.stop_request.set()
        self.destroy()

    # ---------- event loop ----------
    def poll_events(self):
        while True:
            try:
                ev: UIEvent = self.bus.q.get_nowait()
            except Exception:
                break

            t = ev.type
            d = ev.data

            if t == "log":
                level = d.get("level", "INFO")
                msg = d.get("msg", "")
                self.log(f"[{level}] {msg}")

            elif t == "status":
                self.status_label.configure(text=f"Status: {d.get('text','')}")

            elif t == "step":
                self.set_step(int(d.get("index", 0)), d.get("status", ""))

            elif t == "preview":
                bgr = d.get("bgr", None)
                photo = bgr_to_photoimage(bgr, max_w=880, max_h=560)
                if photo is not None:
                    self.last_preview_photo = photo
                    self.preview_img_label.configure(image=photo)

            elif t == "crop":
                bgr = d.get("bgr", None)
                photo = bgr_to_photoimage(bgr, max_w=420, max_h=320)
                if photo is not None:
                    self.last_crop_photo = photo
                    self.crop_img_label.configure(image=photo)

            elif t == "result":
                ocr = d.get("ocr", "—")
                sheet = d.get("sheet", "—")
                self.result_label.configure(text=f"OCR: {ocr}\nSheet: {sheet}")

            elif t == "stopped":
                self.worker_running = False
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                self.btn_toggle_af.configure(state="disabled")
                self.btn_recapture_ref.configure(state="disabled")
                self.btn_change_roi.configure(state="disabled")
                self.log("=== STOPPED ===")
                self.status_label.configure(text="Status: Stopped")

        self.after(50, self.poll_events)
