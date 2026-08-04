# PacketVision — Camera-Triggered OCR + Google Sheets Automation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A) ![Gemini](https://img.shields.io/badge/OCR-Gemini-4285F4) ![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8) ![Status](https://img.shields.io/badge/Status-Production-brightgreen)

A computer-vision + automation pipeline that watches a Raspberry Pi camera feed, detects when a packet or document is placed in frame, and turns that into a structured Google Sheets update — with no manual data entry.

This started as a warehouse logging fix and was rebuilt into a clean, production-grade pipeline: camera trigger → deskew crop → Gemini OCR → spreadsheet write, with quota protection and retry logic built in from the start rather than bolted on.

## 📌 Project Summary

Manual packet logging is slow and error-prone — someone has to read a label, find the right row, and type the ID in by hand, over and over. PacketVision replaces that with a camera that watches for a packet, captures a sharp image the moment one appears, reads the ID with Gemini OCR, and writes the result straight into the sheet that tracks it.

It's built for real-world conditions on a warehouse or logistics station, where lighting varies, packets land at odd angles, and the pipeline can't afford to be babysat.

## ✨ Key Features

✅ Hands-free trigger via region-of-interest change detection
✅ Auto-focus-once + burst capture, with sharpest-frame selection
✅ Red-marker deskew crop for consistent OCR input regardless of packet placement
✅ Compressed, size-capped OCR payloads to control Gemini API cost
✅ Rate limiting (RPM) and daily quota capping (RPD), persisted across restarts
✅ Exponential backoff retry on 429 / quota errors
✅ Google Sheets integration with rule-based row updates
✅ OpenCV overlay or CustomTkinter UI for live operator feedback

## 🧠 Pipeline Architecture

| Stage | What happens |
|---|---|
| Trigger | Low-res preview stream monitors a user-defined ROI |
| Detect | ROI change starts a short, configurable countdown |
| Capture | Single autofocus pass + burst capture, sharpest frame selected via Laplacian scoring |
| Crop | Largest red contour located, `minAreaRect` computed, region warped from the full-res original |
| OCR | Image resized and JPEG-compressed under a byte limit, sent to Gemini with a structured JSON prompt |
| Write | Extracted ID matched against Column A, row updated per business rules |
| Feedback | Live overlay UI shows capture and result to the operator |

## 🔄 Workflow

1. **Watch** — a low-resolution preview stream monitors the ROI, so nothing expensive runs until something changes.
2. **Detect** — a packet enters frame, a short countdown gives the scene time to settle.
3. **Capture** — autofocus triggers once, a burst of frames is captured, and the sharpest one moves forward.
4. **Crop** — the largest red-marker contour is found and warped out of the full-resolution image; empty warp space is filled green.
5. **Read** — the cropped image is compressed and sent to Gemini, which returns a structured JSON result.
6. **Write** — the ID is matched in the sheet and the row is updated according to its current status.

## 📊 Reliability Notes

**Capture.** Lores preview keeps ROI detection cheap; full-res burst capture with sharpness scoring keeps the final frame in focus. AE/AWB can lock after a reference shot, and AF fires once rather than hunting mid-capture.

**OCR cost control.** Every payload is downscaled by max dimension and pixel count, then compressed to stay under 900KB by default — keeping both latency and API cost predictable.

**Quota protection.** A cooldown enforces requests-per-minute, a persisted state file (`ocr_rate_state.json`) enforces a daily cap, and exponential backoff absorbs 429s automatically — so a burst of activity degrades gracefully instead of taking the pipeline down.

**Sheet logic.** IDs are matched in Column A; rows already marked `done` are skipped, rows marked `receipt` are updated across the relevant columns, and unmatched IDs optionally prompt for confirmation before a new row is added.

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core language |
| Picamera2 | Raspberry Pi camera control |
| OpenCV | Image processing, red-mask detection, deskew/crop |
| Gemini API | OCR extraction |
| Google Sheets API | Data write-back |
| NumPy | Numerical operations |

## 🚀 Requirements

**Hardware:** a Raspberry Pi with a Picamera2-supported camera. Consistent lighting matters more than raw resolution for OCR accuracy.

**Software:** Python 3.9+, a working Picamera2 install, a Google service account credentials file, and a Gemini API key.

## 🔮 Future Improvements

- [ ] Web dashboard for live monitoring across multiple stations
- [ ] Configurable business-rule engine instead of hardcoded sheet logic
- [ ] Docker packaging for faster deployment to new Pi units
- [ ] Automated regression testing on captured sample images
- [ ] Support for multiple ROI zones on a single camera

## 💼 Portfolio Value

This project demonstrates practical experience with real-time computer vision on constrained hardware, LLM-based OCR integration with production-grade guardrails (rate limiting, retries, cost control), and end-to-end automation from camera trigger to business-system update.

## 👨‍💻 Author

**Hesam Fathollahi**
Senior Software Engineer — AI & Computer Vision

## 📄 License

This project is available under the MIT License.
