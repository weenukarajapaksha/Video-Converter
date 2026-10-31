"""Wav2Lip inference, run as a standalone script inside the isolated lip-sync env.

    <lipsync-python> wav2lip_runner.py --repo third_party/Wav2Lip --video in.mp4 \
        --audio speech.wav --out frames.mp4 [--segments segs.json]

Improvements over the reference ``inference.py``:
  * frames are streamed (constant memory for long / HD videos) and encoded straight
    to H.264 through an ffmpeg pipe;
  * only frames inside dubbed speech segments are touched - everything else stays
    bit-identical to the source, which keeps silent sections artifact-free;
  * face detection runs on a downscaled copy; boxes are temporally smoothed;
  * the generated mouth is feather-blended into the lower face only, so eyes and
    upper face keep the original sharpness instead of Wav2Lip's 96x96 resolution;
  * fp16 autocast on CUDA (fits easily in 4 GB VRAM).

This file must not import ``dubber``: it runs under a different interpreter.
Progress is reported on stdout as ``PROGRESS <fraction> <message>`` lines and the
final summary as ``RESULT <json>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from scipy import signal

IMG = 96  # Wav2Lip face resolution
MEL_STEP = 16  # mel frames per video frame window
MEL_FPS = 80.0  # 16 kHz / hop 200


def progress(frac: float, msg: str) -> None:
    print(f"PROGRESS {frac:.4f} {msg}", flush=True)


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """Same features as Wav2Lip's audio.py / hparams.py, on modern librosa."""
    wav = signal.lfilter([1, -0.97], [1], wav)
    spec = np.abs(librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800))
    basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    db = 20 * np.log10(np.maximum(1e-5, basis @ spec)) - 20
    return np.clip(8 * ((db + 100) / 100) - 4, -4, 4).astype(np.float32)


def blend_mask() -> np.ndarray:
    """Soft mask covering the generated lower half of the 96x96 face."""
    m = np.zeros((IMG, IMG), np.float32)
    m[int(IMG * 0.52):, :] = 1.0
    m[:, :6] = m[:, -6:] = 0.0
    m[-3:, :] = 0.0
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=4)
    return m[..., None]


def in_segments(t: float, segments: list[tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in segments)


def open_video(path: str) -> tuple[cv2.VideoCapture, int, int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, n, cap.get(cv2.CAP_PROP_FPS)


def detect_faces(args, device, active, n_frames) -> dict[int, tuple[int, int, int, int]]:
    from face_detection.detection.sfd.sfd_detector import SFDDetector

    detector = SFDDetector(device=device, path_to_detector=str(
        Path(args.repo) / "face_detection" / "detection" / "sfd" / "s3fd.pth"))
    cap, w, h, _, _ = open_video(args.video)
    scale = min(1.0, args.det_size / max(w, h))
    small = (int(w * scale), int(h * scale))
    boxes: dict[int, tuple[int, int, int, int]] = {}
    batch, idxs = [], []

    def flush():
        if not batch:
            return
        dets = detector.detect_from_batch(np.asarray(batch))
        for i, d in zip(idxs, dets):
            if len(d):
                # largest face = most likely the speaker in frame
                x1, y1, x2, y2, _ = max(d, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = (np.array([x1, y1, x2, y2]) / scale).astype(int)
                pad = int((y2 - y1) * 0.08)  # include the chin, like Wav2Lip's pads
                boxes[i] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2 + pad))
        batch.clear()
        idxs.clear()

    i = 0
    wanted: list[int] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if active(i):
            wanted.append(i)
            # Faces move little between adjacent frames: detect on every `det_stride`-th
            # frame and fill the rest from neighbours (the detector is the bottleneck).
            if i % args.det_stride == 0:
                batch.append(cv2.cvtColor(cv2.resize(frame, small), cv2.COLOR_BGR2RGB))
                idxs.append(i)
                if len(batch) == args.det_batch:
                    flush()
                    progress(0.4 * i / max(1, n_frames), "detecting faces")
        i += 1
    flush()
    cap.release()
    del detector
    if device == "cuda":
        torch.cuda.empty_cache()
    return smooth_boxes(fill_gaps(boxes, wanted, args.det_stride))


def fill_gaps(boxes: dict[int, tuple], wanted: list[int], stride: int) -> dict[int, tuple]:
    """Give undetected frames the box of the nearest detection within `stride` frames."""
    out = dict(boxes)
    for i in wanted:
        if i in out:
            continue
        near = [j for j in range(i - stride, i + stride + 1) if j in boxes]
        if near:
            out[i] = boxes[min(near, key=lambda j: abs(j - i))]
    return out


def smooth_boxes(boxes: dict[int, tuple], radius: int = 2) -> dict[int, tuple]:
    out = {}
    for i in boxes:
        window = [boxes[j] for j in range(i - radius, i + radius + 1) if j in boxes]
        out[i] = tuple(int(v) for v in np.mean(window, axis=0))
    return out


def load_model(args, device):
    from models import Wav2Lip

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception:  # older checkpoints pickle optimizer state; source is pinned in setup
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model = Wav2Lip()
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True, help="speech-only audio (no music)")
    p.add_argument("--out", required=True, help="video-only H.264 output")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--segments", default="", help="json [[start, end], ...] in seconds")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--det-batch", type=int, default=4)  # ~1.9 GB peak VRAM
    p.add_argument("--det-stride", type=int, default=2)
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--crf", default="18")
    p.add_argument("--preset", default="medium")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    cap, w, h, n_frames, cv_fps = open_video(args.video)
    cap.release()
    fps = args.fps or cv_fps
    segments = [tuple(s) for s in json.loads(Path(args.segments).read_text())] \
        if args.segments else []
    pad = 0.1  # let the mouth open/close slightly around each line

    def active(i: int) -> bool:
        return not segments or in_segments(i / fps, [(s - pad, e + pad) for s, e in segments])

    progress(0.0, "loading audio")
    wav, _ = librosa.load(args.audio, sr=16000)
    mel = melspectrogram(wav)

    boxes = detect_faces(args, device, active, n_frames)
    progress(0.4, f"faces found in {len(boxes)} frames")
    model = load_model(args, device)
    mask = blend_mask()

    enc = subprocess.Popen(
        [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-an", "-c:v", "libx264", "-crf", args.crf, "-preset", args.preset,
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE)

    pending: list[tuple[np.ndarray, int | None]] = []  # (frame, idx if it needs inference)
    n_jobs = 0
    processed = 0

    def flush():
        nonlocal n_jobs, processed
        jobs = [(k, idx) for k, (_, idx) in enumerate(pending) if idx is not None]
        if jobs:
            faces, mels = [], []
            for k, idx in jobs:
                x1, y1, x2, y2 = boxes[idx]
                faces.append(cv2.resize(pending[k][0][y1:y2, x1:x2], (IMG, IMG)))
                start = int(idx * MEL_FPS / fps)
                chunk = mel[:, start:start + MEL_STEP]
                if chunk.shape[1] < MEL_STEP:
                    chunk = mel[:, -MEL_STEP:]
                mels.append(chunk)
            img = np.asarray(faces)
            masked = img.copy()
            masked[:, IMG // 2:] = 0
            img_t = torch.from_numpy(
                np.concatenate([masked, img], axis=3).transpose(0, 3, 1, 2) / 255.0
            ).float().to(device)
            mel_t = torch.from_numpy(np.asarray(mels)[:, None]).float().to(device)
            with torch.no_grad(), torch.autocast(device, dtype=torch.float16,
                                                 enabled=device == "cuda"):
                pred = model(mel_t, img_t)
            pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for (k, idx), face in zip(jobs, pred):
                frame = pending[k][0]
                x1, y1, x2, y2 = boxes[idx]
                size = (x2 - x1, y2 - y1)
                gen = cv2.resize(face, size)
                m = cv2.resize(mask, size)[..., None]
                roi = frame[y1:y2, x1:x2].astype(np.float32)
                frame[y1:y2, x1:x2] = np.clip(roi * (1 - m) + gen * m, 0, 255).astype(np.uint8)
            processed += len(jobs)
        for frame, _ in pending:
            enc.stdin.write(frame.tobytes())
        pending.clear()
        n_jobs = 0

    cap, *_ = open_video(args.video)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = i if (i in boxes and active(i)) else None
        if idx is None and not pending:
            enc.stdin.write(frame.tobytes())
        else:
            pending.append((frame, idx))
            n_jobs += idx is not None
            if n_jobs >= args.batch or len(pending) >= 4 * args.batch:
                flush()
                progress(0.4 + 0.6 * i / max(1, n_frames), "generating lips")
        i += 1
    flush()
    cap.release()
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg encoder failed")

    vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    progress(1.0, "done")
    print("RESULT " + json.dumps({
        "frames": i, "lipsynced_frames": processed, "face_frames": len(boxes),
        "seconds": round(time.perf_counter() - t0, 2), "device": device,
        "peak_vram_mb": round(vram)}), flush=True)


if __name__ == "__main__":
    main()
