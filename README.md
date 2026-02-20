# PacketVision — Camera-Triggered OCR + Google Sheets Automation (Raspberry Pi)

PacketVision is a Raspberry Pi workflow that **automatically detects when a document/packet is placed under the camera**, captures a sharp full-resolution image, **deskews/crops the red-marked region**, runs **Gemini OCR** to extract an ID (Amazon Order ID or tracking number), and then **updates a Google Sheet** based on business rules.

It’s designed for real-world warehouse / logistics stations where speed and reliability matter:
- ROI change detection for “hands-free” triggering  
- Auto-focus handling + burst capture to avoid blur  
- Red-mask deskew crop for consistent OCR input  
- Rate limiting + retry logic to respect Gemini quotas  
- Google Sheets integration for simple “database-like” operations

---

## What it does (high level)

1. **ROI is used as a trigger only**  
   A low-resolution preview stream monitors a user-defined ROI.
2. **When ROI changes (packet detected)**  
   A countdown starts (configurable).
3. **Auto-focus once + capture burst**  
   The system triggers AF once and captures multiple frames, selecting the sharpest.
4. **Full-res capture + red-region deskew crop**  
   It detects the largest red contour, computes `minAreaRect`, and warps the region from the original full-res image.
   Empty pixels from the warp are filled with **green (0,255,0)**.
5. **OCR with Gemini**  
   It prepares a controlled OCR payload (resize + JPEG compression under a byte limit) and asks Gemini to return a JSON result.
6. **Google Sheets update**  
   It finds the row where **Column A == ID** and applies business rules (update only if status is `receipt`, etc.).
7. **UI feedback**  
   Either OpenCV overlay UI or CustomTkinter UI (depending on your version).

---

## Features

### Capture reliability
- **Lores preview for fast ROI detection**
- **Full-res burst capture** + best-frame selection using Laplacian sharpness
- Optional **AE/AWB lock** after reference capture
- Optional **AF single-shot** before capture

### Smart cropping (red marker workflow)
- Fast red mask computed on a resized copy
- Finds biggest red contour → `minAreaRect` → perspective warp from original full-res
- Optional trimming of the green border after warping

### OCR payload control (cost + performance)
- Downscale by max dimension and max pixels
- JPEG compression to stay under a target size (default: < 900KB)
- Saves the exact OCR image bytes sent to Gemini (optional)

### Gemini quota protection
- Cooldown between calls (RPM protection)
- Daily max call cap (RPD protection)
- Persisted state file (`ocr_rate_state.json`)
- Retry with exponential backoff on 429 / quota errors

### Google Sheets automation
- Looks up ID in **Column A**
- Business rules:
  - If Column D = `done` → “packet is done already”
  - If Column D = `receipt` → update columns D / H / I / O
  - If ID doesn’t exist → optional user confirmation to add a new row

---

## Requirements

### Hardware
- Raspberry Pi + Pi Camera supported by **Picamera2**
- Stable lighting recommended for best OCR

### Software
- Python 3.9+ recommended
- Picamera2 installed and working
- Google service account credentials JSON
- Gemini API key

---
