# PacketVision — Camera-Triggered OCR + Google Sheets Automation

PacketVision watches a Raspberry Pi camera feed for the moment a packet or document lands in frame, captures a sharp full-resolution shot, deskews and crops the marked region, and runs it through Gemini OCR to pull an ID (an Amazon order ID or tracking number). That ID then drives an update to a Google Sheet — no manual entry required.

It was built for warehouse and logistics stations, where a slow or unreliable capture pipeline turns into a bottleneck on the floor. A few things made that possible:

- Change detection on a region of interest, so the camera triggers itself
- Auto-focus plus burst capture, so blur doesn't ruin a read
- A red-marker deskew step that keeps OCR input consistent regardless of how the packet is placed
- Rate limiting and retry logic that respect Gemini's quotas instead of hammering the API
- A Google Sheets integration that behaves like a lightweight database for status updates

## How it works

The pipeline runs in six stages, from an idle camera to an updated spreadsheet row:

**1. Watch, don't scan.** A low-resolution preview stream monitors a user-defined region of interest. Nothing expensive happens until something changes there.

**2. Detect and wait.** Once the ROI changes — a packet has been placed — a short, configurable countdown gives the scene time to settle.

**3. Focus once, shoot several.** The camera triggers autofocus a single time, then captures a burst of frames. The sharpest one, picked automatically, moves forward.

**4. Crop to the marker.** The system finds the largest red contour in the frame, computes its bounding rectangle, and warps that region out of the original full-resolution image. Any empty space left by the warp is filled with green so it doesn't get mistaken for content.

**5. Read it with Gemini.** The cropped image is resized and compressed to stay under a byte limit, then sent to Gemini with a structured prompt that asks for a JSON result — not free text to parse.

**6. Write it back.** The extracted ID is matched against Column A in the target sheet, and the row is updated according to a small set of business rules (for instance, only touching a row if its status is still "receipt").

A UI layer sits on top of all this — either an OpenCV overlay or a CustomTkinter window, depending on the build — so an operator can see what the camera sees and confirm a read if needed.

## What makes it reliable in practice

**Capture.** A low-res preview handles ROI detection cheaply, while full-resolution bursts with Laplacian-based sharpness scoring make sure the frame that actually gets processed is in focus. Exposure and white balance can be locked after a reference capture, and autofocus runs once rather than continuously, so it doesn't hunt mid-shot.

**Cropping.** The red-marker workflow computes a fast mask on a downscaled copy, locates the largest contour, and only then warps the corresponding region out of the full-resolution original — so cropping is fast without sacrificing image quality. A trimming step cleans up the green border left behind by the warp.

**OCR cost control.** Every image sent to Gemini is downscaled by both maximum dimension and pixel count, then JPEG-compressed to a target size (under 900KB by default). The exact bytes sent can optionally be saved locally, which makes debugging a bad read straightforward.

**Quota protection.** A cooldown enforces a request-per-minute limit, a persisted state file (`ocr_rate_state.json`) enforces a daily cap, and exponential backoff kicks in automatically on 429 or quota errors — so a burst of activity degrades gracefully instead of taking the pipeline down.

**Sheet updates.** IDs are matched in Column A, with rules like: skip rows already marked done, update the relevant columns when a row is in "receipt" status, and prompt for confirmation before adding a row for an ID that isn't found at all.

## Requirements

**Hardware:** a Raspberry Pi with a camera supported by Picamera2. Consistent lighting matters more than raw resolution for OCR accuracy.

**Software:** Python 3.9+, a working Picamera2 install, a Google service account credentials file, and a Gemini API key.
