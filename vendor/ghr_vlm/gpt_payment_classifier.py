#!/usr/bin/env python3
"""
python gpt_payment_classifier.py --clips_dir outputs_C3_3 --model gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont


PAYMENT_TYPES = [
    "QR Code with phone",
    "Cash",
    "Tap with card",
    "Swipe with card",
    "Evade",
]


PROMPT = f"""
You are classifying a short ROI-cropped bus farebox video clip.
The image is a contact sheet of frames sampled in time order from one passenger clip.

Choose exactly one payment_type from this list:
{json.dumps(PAYMENT_TYPES, indent=2)}

Visual decision rules:
- QR Code with phone: passenger holds a phone toward the farebox. The QR scanner is on the right side of the farebox.
- Cash: passenger inserts bills/coins at the farebox slot. The cash and coin slot is on the right side of the farebox, near the QR scanner. Coin insertion is usually a quick tap of the coin into the slot, while bill insertion is a longer motion of sliding the bill into the slot.
- Tap with card: passenger holds a card near the tap reader briefly, usually a flat card touching or hovering at the reader. The tap reader is on the left side of the farebox.
- Swipe with card: The passenger holds a card vertically and slides it along the straight card slot on the farebox. Card slot is a vertical strait slot between the tap reader and the QR scanner.
- Evade: passenger enters without a visible payment interaction in the ROI, or the clip shows no usable evidence of payment.

Use the contact sheet temporally. Look for the hand/object trajectory, not just one isolated frame.
If evidence is weak or occluded, choose the most likely label and lower the confidence.
Return only a JSON object with this shape: 
{{
  "payment_type": "one of the five labels",
  "confidence": "high|medium|low",
  "reasoning": "brief visual evidence",
  "evidence_frames": ["frame labels that support the decision"]
}}
""".strip()


FINAL_PROMPT = f"""
You are classifying a short ROI-cropped bus farebox video clip.
The image is a contact sheet of frames sampled in time order from the most
important payment-interaction window of one passenger clip.

Choose exactly one payment_type from this list:
{json.dumps(PAYMENT_TYPES, indent=2)}

Visual decision rules:
- QR Code with phone: passenger holds a phone toward the farebox. The QR scanner is on the right side of the farebox.
- Cash: passenger inserts bills/coins at the farebox slot. The cash and coin slot is on the right side of the farebox, near the QR scanner. Coin insertion is usually a quick tap of the coin into the slot, while bill insertion is a longer motion of sliding the bill into the slot.
- Tap with card: passenger holds a card near the tap reader briefly, usually a flat card touching or hovering at the reader. The tap reader is on the left side of the farebox.
- Swipe with card: The passenger holds a card vertically and slides it along the straight card slot on the farebox. Card slot is a vertical strait slot between the tap reader and the QR scanner.
- Evade: passenger enters without a visible payment interaction in the ROI, or the clip shows no usable evidence of payment.

Use the contact sheet temporally. Look for the hand/object trajectory, not just one isolated frame.
If evidence is weak or occluded, choose the most likely label and lower the confidence.
Return only a JSON object with this shape:
{{
  "payment_type": "one of the five labels",
  "confidence": "high|medium|low",
  "reasoning": "brief visual evidence"
}}
""".strip()


@dataclass
class ClipResult:
    clip: str
    payment_type: str
    confidence: str
    reasoning: str
    evidence_frames: str
    raw_response: str
    contact_sheet: str
    stage1_payment_type: str
    stage1_confidence: str
    stage1_reasoning: str
    stage1_evidence_frames: str
    stage1_raw_response: str
    stage1_contact_sheet: str


def parse_crop(value: str) -> tuple[int, int, int, int] | None:
    if value.lower() in {"none", "full", ""}:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,w,h or 'none'")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("crop width and height must be positive")
    return x, y, w, h


def parse_roi(raw_roi: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in raw_roi.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--payment-roi must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--payment-roi values must be numbers") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--payment-roi must satisfy x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def parse_margin_triplet(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("margin triplet must be x,top,bottom")
    try:
        x_ratio, top_ratio, bottom_ratio = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("margin values must be numbers") from exc
    if x_ratio < 0:
        raise ValueError("x margin must be >= 0")
    if top_ratio < 0:
        raise ValueError("top margin must be >= 0")
    return x_ratio, top_ratio, bottom_ratio


def parse_stage_margins(raw: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    groups = [group.strip() for group in raw.split(";")]
    if len(groups) != 2:
        raise argparse.ArgumentTypeError(
            "--payment-roi-margins must be 'stage1_x,stage1_top,stage1_bottom;stage2_x,stage2_top,stage2_bottom'"
        )
    try:
        return parse_margin_triplet(groups[0]), parse_margin_triplet(groups[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def find_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() == ".mp4")


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if sample_count <= 1:
        return [frame_count // 2]
    return [
        min(frame_count - 1, max(0, round(i * (frame_count - 1) / (sample_count - 1))))
        for i in range(sample_count)
    ]


def sample_frame_indices_between(start_frame: int, end_frame: int, sample_count: int) -> list[int]:
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame, int(end_frame))
    if sample_count <= 1:
        return [(start_frame + end_frame) // 2]
    if start_frame == end_frame:
        return [start_frame] * sample_count
    return [
        min(end_frame, max(start_frame, round(start_frame + i * (end_frame - start_frame) / (sample_count - 1))))
        for i in range(sample_count)
    ]


def resolve_roi_pixels(
    roi: tuple[float, float, float, float],
    width: int,
    height: int,
    roi_format: str,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    use_normalized = roi_format == "normalized" or (
        roi_format == "auto" and max(abs(value) for value in roi) <= 1.0
    )
    if use_normalized:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    px1 = max(0, min(width - 1, int(round(x1))))
    py1 = max(0, min(height - 1, int(round(y1))))
    px2 = max(px1 + 1, min(width, int(round(x2))))
    py2 = max(py1 + 1, min(height, int(round(y2))))
    return px1, py1, px2, py2


def expand_roi_pixels(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_x_ratio: float,
    margin_top_ratio: float,
    margin_bottom_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = int(round(box_width * max(0.0, margin_x_ratio)))
    margin_top = int(round(box_height * max(0.0, margin_top_ratio)))
    margin_bottom = int(round(box_height * margin_bottom_ratio))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_top),
        min(width, x2 + margin_x),
        max(y1 + 1, min(height, y2 + margin_bottom)),
    )


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return x1, y1, x2, y2


def read_frame_as_image(video_path: Path, frame_index: int) -> tuple[Image.Image, dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip for ROI detection: {video_path}")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not read ROI-detection frame from: {video_path}")
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb), {
            "frame_index": frame_index,
            "original_width": width,
            "original_height": height,
        }
    finally:
        cap.release()


def save_roi_debug_image(
    frame: Image.Image,
    bbox: tuple[float, float, float, float],
    path: Path,
) -> None:
    debug = frame.copy().convert("RGB")
    draw = ImageDraw.Draw(debug)
    draw.rectangle(bbox, outline=(255, 0, 0), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(path, format="JPEG", quality=92)
    debug.close()


def save_roi_crop_debug_image(
    frame: Image.Image,
    bbox: tuple[float, float, float, float],
    path: Path,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    crop = frame.crop((x1, y1, x2, y2)).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, format="JPEG", quality=92)
    crop.close()


def resize_frame_if_needed(frame: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return frame
    width, height = frame.size
    largest = max(width, height)
    if largest <= max_side:
        return frame
    scale = max_side / largest
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return frame.resize(new_size, Image.Resampling.LANCZOS)


def detect_payment_roi_sam3(
    *,
    clip_path: Path,
    frame_index: int,
    image_size: int,
    model_name: str,
    prompt: str,
    prompt_variants: str,
    thresholds: str,
    timeout: float,
    debug_dir: Path,
) -> dict[str, Any]:
    original_frame, frame_metadata = read_frame_as_image(clip_path, frame_index)
    model_frame = resize_frame_if_needed(original_frame, image_size)
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_json_path = debug_dir / "payment_roi_sam3_raw.json"
        sam3_source_path = debug_dir / "payment_roi_sam3_source.jpg"
        sam3_overlay_path = debug_dir / "payment_roi_sam3_overlay.jpg"
        debug_image_path = debug_dir / "payment_roi_detected.jpg"
        debug_crop_path = debug_dir / "payment_roi_detected_crop.jpg"
        model_frame.save(sam3_source_path, format="JPEG", quality=92)

        command = [
            sys.executable,
            str(Path(__file__).with_name("sam3_door_detector.py")),
            "--model",
            model_name,
            "--image",
            str(sam3_source_path),
            "--prompt",
            prompt,
            "--prompt-variants",
            prompt_variants,
            "--thresholds",
            thresholds,
            "--output",
            str(raw_json_path),
            "--overlay",
            str(sam3_overlay_path),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "SAM3 could not find the farebox/payment ROI: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
        if not bool(payload.get("found")):
            raise RuntimeError(f"SAM3 did not return a farebox/payment ROI for prompt {prompt!r}.")

        model_width, model_height = model_frame.size
        x1, y1, x2, y2 = clamp_bbox(
            float(payload.get("x1", 0.0)),
            float(payload.get("y1", 0.0)),
            float(payload.get("x2", 0.0)),
            float(payload.get("y2", 0.0)),
            model_width,
            model_height,
        )
        original_width = int(frame_metadata["original_width"])
        original_height = int(frame_metadata["original_height"])
        scale_x = original_width / model_width
        scale_y = original_height / model_height
        original_bbox = clamp_bbox(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
            original_width,
            original_height,
        )
        normalized_roi = (
            original_bbox[0] / original_width,
            original_bbox[1] / original_height,
            original_bbox[2] / original_width,
            original_bbox[3] / original_height,
        )
        save_roi_debug_image(original_frame, original_bbox, debug_image_path)
        save_roi_crop_debug_image(original_frame, original_bbox, debug_crop_path)
        return {
            "source": "sam3",
            "clip_path": str(clip_path),
            "frame_index": frame_index,
            "roi_format": "normalized",
            "roi": list(normalized_roi),
            "bbox_original_pixels": list(original_bbox),
            "bbox_model_pixels": [x1, y1, x2, y2],
            "original_width": original_width,
            "original_height": original_height,
            "model_width": model_width,
            "model_height": model_height,
            "confidence": payload.get("confidence", 0.0),
            "prompt": payload.get("prompt", prompt),
            "requested_prompt": payload.get("requested_prompt", prompt),
            "prompt_variants": payload.get("prompt_variants", []),
            "threshold": payload.get("threshold"),
            "thresholds": payload.get("thresholds", thresholds),
            "model": payload.get("model", model_name),
            "raw_json_path": str(raw_json_path),
            "sam3_source_image_path": str(sam3_source_path),
            "sam3_overlay_path": str(sam3_overlay_path),
            "debug_image_path": str(debug_image_path),
            "debug_crop_path": str(debug_crop_path),
        }
    finally:
        if model_frame is not original_frame:
            model_frame.close()
        original_frame.close()


def resolve_payment_roi(args: argparse.Namespace, clips: list[Path], output_root: Path, output_dir: Path) -> dict[str, Any]:
    roi_output = Path(args.roi_output).expanduser().resolve() if args.roi_output else output_root / "payment_roi_detected.json"
    if args.payment_roi:
        roi_metadata = {
            "source": "manual",
            "roi": list(args.payment_roi),
            "roi_format": args.roi_format,
            "confidence": None,
            "reasoning": "Manual --payment-roi provided.",
        }
    elif roi_output.exists() and not args.force_roi_detect:
        print(f"Loading cached payment ROI from {roi_output}", flush=True)
        roi_metadata = json.loads(roi_output.read_text(encoding="utf-8"))
        if "roi" not in roi_metadata or "roi_format" not in roi_metadata:
            raise ValueError(f"Cached ROI file is missing roi/roi_format: {roi_output}")
        roi_metadata["source"] = roi_metadata.get("source", "cached")
        roi_metadata["loaded_from_cache"] = True
    else:
        print(f"Detecting shared payment ROI with SAM3 from {clips[0].name}", flush=True)
        roi_metadata = detect_payment_roi_sam3(
            clip_path=clips[0],
            frame_index=0,
            image_size=args.roi_detect_image_size,
            model_name=args.sam3_payment_model,
            prompt=args.sam3_payment_prompt,
            prompt_variants=args.sam3_payment_prompt_variants,
            thresholds=args.sam3_payment_thresholds,
            timeout=args.sam3_payment_timeout,
            debug_dir=output_dir / "payment_roi_debug",
        )
        roi_metadata["loaded_from_cache"] = False

    roi_output.parent.mkdir(parents=True, exist_ok=True)
    roi_output.write_text(json.dumps(roi_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return roi_metadata


def extract_roi_frames_at_indices(
    video_path: Path,
    frame_indices: list[int],
    roi: tuple[float, float, float, float],
    roi_format: str,
    margin_x_ratio: float,
    margin_top_ratio: float,
    margin_bottom_ratio: float,
    max_image_size: int,
) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[Image.Image] = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        roi_pixels = resolve_roi_pixels(roi, width, height, roi_format)
        x1, y1, x2, y2 = expand_roi_pixels(
            roi_pixels,
            width,
            height,
            margin_x_ratio,
            margin_top_ratio,
            margin_bottom_ratio,
        )
        frame = frame[y1:y2, x1:x2]
        if frame.size == 0:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        frames.append(resize_frame_if_needed(image, max_image_size))
        if frames[-1] is not image:
            image.close()

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frames


def extract_roi_frames(
    video_path: Path,
    sample_count: int,
    roi: tuple[float, float, float, float],
    roi_format: str,
    margin_x_ratio: float,
    margin_top_ratio: float,
    margin_bottom_ratio: float,
    max_image_size: int,
) -> tuple[list[Image.Image], list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    frame_indices = sample_frame_indices(frame_count, sample_count)
    frames = extract_roi_frames_at_indices(
        video_path=video_path,
        frame_indices=frame_indices,
        roi=roi,
        roi_format=roi_format,
        margin_x_ratio=margin_x_ratio,
        margin_top_ratio=margin_top_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
        max_image_size=max_image_size,
    )
    return frames, frame_indices


def resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def build_contact_sheet(
    frames: list[Image.Image],
    columns: int,
    scale: float,
    title: str,
    max_image_side: int,
) -> tuple[Image.Image, dict[str, list[int]]]:
    if columns <= 0:
        raise ValueError("columns must be positive")

    label_height = 22
    padding = 12
    title_height = 28
    font = ImageFont.load_default()

    resized_frames = []
    for frame in frames:
        width, height = frame.size
        resized = frame.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        resized_frames.append(resized)

    cell_width = max(frame.width for frame in resized_frames)
    cell_height = max(frame.height for frame in resized_frames) + label_height
    rows = (len(resized_frames) + columns - 1) // columns
    sheet_width = columns * cell_width + (columns + 1) * padding
    sheet_height = title_height + rows * cell_height + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 8), title[:180], fill=(0, 0, 0), font=font)
    frame_boxes: dict[str, list[int]] = {}

    for index, frame in enumerate(resized_frames):
        row, col = divmod(index, columns)
        x = padding + col * (cell_width + padding)
        y = title_height + padding + row * (cell_height + padding)
        sheet.paste(frame, (x, y))
        draw.rectangle([x, y, x + frame.width - 1, y + frame.height - 1], outline=(30, 30, 30), width=1)
        label = f"frame_{index + 1:02d}"
        frame_boxes[label] = [x, y, x + frame.width - 1, y + frame.height - 1]
        draw.text((x, y + frame.height + 4), label, fill=(0, 0, 0), font=font)

    resized = resize_to_max_side(sheet, max_image_side)
    if resized is not sheet:
        scale_x = resized.width / sheet.width
        scale_y = resized.height / sheet.height
        frame_boxes = {
            label: [
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            ]
            for label, (x1, y1, x2, y2) in frame_boxes.items()
        }
        sheet.close()
    return resized, frame_boxes


def save_jpeg(image: Image.Image, path: Path, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality, optimize=True)


def normalize_evidence_label(raw: Any) -> str | None:
    text = str(raw).strip().lower()
    match = re.search(r"frame[_\s-]*(\d+)", text)
    if not match:
        match = re.search(r"\b(\d+)\b", text)
    if not match:
        return None
    return f"frame_{int(match.group(1)):02d}"


def evidence_labels_from_text(evidence_frames: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(r"frame[_\s-]*(\d+)", evidence_frames, flags=re.IGNORECASE):
        label = f"frame_{int(match.group(1)):02d}"
        if label not in labels:
            labels.append(label)
    if labels:
        return labels
    for item in re.split(r"[;,]\s*|\n+", evidence_frames):
        label = normalize_evidence_label(item)
        if label and label not in labels:
            labels.append(label)
    return labels


def annotate_evidence_frames(
    sheet_path: Path,
    frame_boxes: dict[str, list[int]],
    evidence_frames: str,
    quality: int,
) -> None:
    labels = evidence_labels_from_text(evidence_frames)
    if not labels:
        return
    image = Image.open(sheet_path).convert("RGB")
    try:
        draw = ImageDraw.Draw(image)
        width = max(4, round(max(image.size) / 300))
        for label in labels:
            bbox = frame_boxes.get(label)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            for offset in range(width):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=(255, 0, 0),
                )
        save_jpeg(image, sheet_path, quality=quality)
    finally:
        image.close()


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).strip()


def response_dump(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    if hasattr(response, "json"):
        return response.json(indent=2)
    return str(response)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def classify_with_gpt(
    image_path: Path,
    prompt: str,
    model: str,
    reasoning_effort: str | None,
    temperature: float | None,
) -> tuple[dict[str, Any], str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc

    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            },
        ],
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
    if temperature is not None:
        request["temperature"] = temperature

    response = client.responses.create(**request)
    text = response_text(response)
    return parse_json_object(text), response_dump(response)


def normalize_prediction(data: dict[str, Any]) -> tuple[str, str, str, str]:
    payment_type = str(data.get("payment_type", "")).strip()
    if payment_type not in PAYMENT_TYPES:
        raise ValueError(f"Model returned invalid payment_type: {payment_type!r}")

    confidence = str(data.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    reasoning = str(data.get("reasoning", "")).strip()
    evidence = data.get("evidence_frames", [])
    if isinstance(evidence, list):
        evidence_frames = ";".join(str(item) for item in evidence)
    else:
        evidence_frames = str(evidence)
    return payment_type, confidence, reasoning, evidence_frames


def normalize_final_prediction(data: dict[str, Any]) -> tuple[str, str, str]:
    payment_type = str(data.get("payment_type", "")).strip()
    if payment_type not in PAYMENT_TYPES:
        raise ValueError(f"Model returned invalid payment_type: {payment_type!r}")

    confidence = str(data.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    reasoning = str(data.get("reasoning", "")).strip()
    return payment_type, confidence, reasoning


def evidence_source_window(evidence_frames: str, source_indices: list[int]) -> tuple[int, int]:
    labels = evidence_labels_from_text(evidence_frames)
    selected_indices: list[int] = []
    for label in labels:
        match = re.search(r"(\d+)$", label)
        if not match:
            continue
        contact_index = int(match.group(1)) - 1
        if 0 <= contact_index < len(source_indices):
            selected_indices.append(source_indices[contact_index])
    if not selected_indices:
        return source_indices[0], source_indices[-1]
    return min(selected_indices), max(selected_indices)


def write_outputs(results: list[ClipResult], output_dir: Path) -> None:
    csv_path = output_dir / "gpt_payment_predictions.csv"
    jsonl_path = output_dir / "gpt_payment_predictions.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip",
                "payment_type",
                "confidence",
                "reasoning",
                "evidence_frames",
                "contact_sheet",
                "raw_response",
                "stage1_payment_type",
                "stage1_confidence",
                "stage1_reasoning",
                "stage1_evidence_frames",
                "stage1_contact_sheet",
                "stage1_raw_response",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")


def classify_videos(args: argparse.Namespace) -> int:
    output_root = Path(args.clips_dir).expanduser().resolve()
    root = output_root / "passenger_clips_2"
    output_dir = Path(args.output).expanduser().resolve() if args.output else output_root / "_gpt_payment_review"
    stage1_sheet_dir = output_dir / "stage1_contact_sheets"
    stage2_sheet_dir = output_dir / "stage2_contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_sheet_dir.mkdir(parents=True, exist_ok=True)
    stage2_sheet_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Passenger clip directory does not exist: {root}", file=sys.stderr)
        return 1

    videos = find_videos(root)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"No .mp4 files found in {root}", file=sys.stderr)
        return 1

    roi_metadata = resolve_payment_roi(args, videos, output_root, output_dir)
    roi = tuple(float(value) for value in roi_metadata["roi"])
    roi_format = str(roi_metadata["roi_format"])
    stage1_margins, stage2_margins = args.payment_roi_margins
    print(f"Using payment ROI ({roi_format}): {roi}", flush=True)
    print(
        f"Using stage margins: stage1={stage1_margins}, stage2={stage2_margins}",
        flush=True,
    )

    results: list[ClipResult] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        stage1_sheet_path = stage1_sheet_dir / f"{video.stem}_stage1.jpg"
        stage2_sheet_path = stage2_sheet_dir / f"{video.stem}_stage2.jpg"
        stage1_frame_boxes: dict[str, list[int]] = {}
        stage1_source_indices: list[int] = []

        frames, stage1_source_indices = extract_roi_frames(
            video,
            sample_count=args.frames,
            roi=roi,
            roi_format=roi_format,
            margin_x_ratio=stage1_margins[0],
            margin_top_ratio=stage1_margins[1],
            margin_bottom_ratio=stage1_margins[2],
            max_image_size=args.openai_frame_image_size,
        )
        sheet, stage1_frame_boxes = build_contact_sheet(
            frames,
            columns=args.columns,
            scale=args.scale,
            title=video.name,
            max_image_side=args.max_image_side,
        )
        save_jpeg(sheet, stage1_sheet_path, quality=args.jpeg_quality)
        sheet.close()
        for frame in frames:
            frame.close()

        if args.dry_run:
            result = ClipResult(
                clip=video.name,
                payment_type="DRY_RUN",
                confidence="",
                reasoning="Two-stage contact sheets generated; API calls skipped.",
                evidence_frames="",
                raw_response="",
                contact_sheet=str(stage2_sheet_path),
                stage1_payment_type="DRY_RUN",
                stage1_confidence="",
                stage1_reasoning="Stage 1 contact sheet generated; API call skipped.",
                stage1_evidence_frames="",
                stage1_raw_response="",
                stage1_contact_sheet=str(stage1_sheet_path),
            )
        else:
            try:
                stage1_prediction, stage1_raw_response = classify_with_gpt(
                    stage1_sheet_path,
                    prompt=PROMPT,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature,
                )
                stage1_payment_type, stage1_confidence, stage1_reasoning, evidence_frames = normalize_prediction(stage1_prediction)
                annotate_evidence_frames(
                    sheet_path=stage1_sheet_path,
                    frame_boxes=stage1_frame_boxes,
                    evidence_frames=evidence_frames,
                    quality=args.jpeg_quality,
                )

                start_frame, end_frame = evidence_source_window(evidence_frames, stage1_source_indices)
                stage2_source_indices = sample_frame_indices_between(
                    start_frame,
                    end_frame,
                    args.frames,
                )
                stage2_frames = extract_roi_frames_at_indices(
                    video_path=video,
                    frame_indices=stage2_source_indices,
                    roi=roi,
                    roi_format=roi_format,
                    margin_x_ratio=stage2_margins[0],
                    margin_top_ratio=stage2_margins[1],
                    margin_bottom_ratio=stage2_margins[2],
                    max_image_size=args.openai_frame_image_size,
                )
                try:
                    stage2_sheet, _ = build_contact_sheet(
                        stage2_frames,
                        columns=args.columns,
                        scale=args.scale,
                        title=f"{video.name} evidence_window_{start_frame}_{end_frame}",
                        max_image_side=args.max_image_side,
                    )
                    save_jpeg(stage2_sheet, stage2_sheet_path, quality=args.jpeg_quality)
                    stage2_sheet.close()
                finally:
                    for frame in stage2_frames:
                        frame.close()

                final_prediction, final_raw_response = classify_with_gpt(
                    stage2_sheet_path,
                    prompt=FINAL_PROMPT,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature,
                )
                payment_type, confidence, reasoning = normalize_final_prediction(final_prediction)
                result = ClipResult(
                    clip=video.name,
                    payment_type=payment_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    evidence_frames=evidence_frames,
                    raw_response=final_raw_response,
                    contact_sheet=str(stage2_sheet_path),
                    stage1_payment_type=stage1_payment_type,
                    stage1_confidence=stage1_confidence,
                    stage1_reasoning=stage1_reasoning,
                    stage1_evidence_frames=evidence_frames,
                    stage1_raw_response=stage1_raw_response,
                    stage1_contact_sheet=str(stage1_sheet_path),
                )
                print(
                    f"  -> {payment_type} ({confidence}); "
                    f"stage1={stage1_payment_type} evidence={evidence_frames}",
                    flush=True,
                )
            except Exception as exc:
                result = ClipResult(
                    clip=video.name,
                    payment_type="ERROR",
                    confidence="",
                    reasoning=str(exc),
                    evidence_frames="",
                    raw_response="",
                    contact_sheet=str(stage2_sheet_path),
                    stage1_payment_type="ERROR",
                    stage1_confidence="",
                    stage1_reasoning=str(exc),
                    stage1_evidence_frames="",
                    stage1_raw_response="",
                    stage1_contact_sheet=str(stage1_sheet_path),
                )
                print(f"  !! {exc}", file=sys.stderr, flush=True)

            if args.sleep > 0 and index < len(videos):
                time.sleep(args.sleep)

        results.append(result)
        write_outputs(results, output_dir)

    print(f"\nWrote: {output_dir / 'gpt_payment_predictions.csv'}")
    print(f"Wrote: {output_dir / 'gpt_payment_predictions.jsonl'}")
    print(f"Stage 1 contact sheets: {stage1_sheet_dir}")
    print(f"Stage 2 contact sheets: {stage2_sheet_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a shared farebox/payment ROI with SAM3 from passenger clips, "
            "then classify payment method from ROI contact sheets with GPT vision."
        )
    )
    parser.add_argument(
        "--clips_dir",
        required=True,
        help="Output root directory, e.g. outputs_C3_3. Clips are read from <clips_dir>/passenger_clips_2.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to <clips_dir>/_gpt_payment_review.",
    )
    parser.add_argument(
        "--roi-output",
        default=None,
        help="Where to save/load payment ROI metadata. Defaults to <clips_dir>/payment_roi_detected.json.",
    )
    parser.add_argument(
        "--force-roi-detect",
        action="store_true",
        help="Rerun SAM3 ROI detection even if the ROI JSON already exists.",
    )
    parser.add_argument(
        "--payment-roi",
        type=parse_roi,
        default=None,
        help="Optional manual farebox/payment ROI as x1,y1,x2,y2, pixel or normalized coordinates.",
    )
    parser.add_argument(
        "--roi-format",
        choices=["auto", "pixel", "normalized"],
        default="auto",
        help="How to interpret --payment-roi. auto treats all values <= 1 as normalized.",
    )
    parser.add_argument(
        "--payment-roi-margins",
        type=parse_stage_margins,
        default=parse_stage_margins("0.35,0.35,-0.5;0.0,0.10,-0.5"),
        help=(
            "Stage-specific ROI margins as "
            "'stage1_x,stage1_top,stage1_bottom;stage2_x,stage2_top,stage2_bottom'. "
            "x/top must be >= 0; bottom can be negative to crop upward. "
            "Default: '0.35,0.35,-0.5;0.0,0.10,-0.5'."
        ),
    )
    parser.add_argument(
        "--roi-detect-image-size",
        type=int,
        default=1280,
        help="Max long side for the first frame sent to SAM3. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--sam3-payment-model",
        default="facebook/sam3",
        help="SAM3 model id/path used for automatic farebox/payment ROI detection.",
    )
    parser.add_argument(
        "--sam3-payment-prompt",
        default="farebox payment box",
        help="Primary SAM3 text prompt for locating the farebox/payment ROI.",
    )
    parser.add_argument(
        "--sam3-payment-prompt-variants",
        default="bus farebox|payment box|fare payment machine|card reader|payment terminal",
        help="Additional SAM3 ROI prompts separated by '|'.",
    )
    parser.add_argument(
        "--sam3-payment-thresholds",
        default="0.50,0.40,0.30,0.20",
        help="Comma-separated SAM3 score thresholds to try for payment ROI.",
    )
    parser.add_argument(
        "--sam3-payment-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for SAM3 payment ROI detection.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"), help="OpenAI vision-capable model")
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT", "high"),
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort to send as reasoning.effort; use none for models that do not support reasoning",
    )
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample per clip")
    parser.add_argument(
        "--openai-frame-image-size",
        type=int,
        default=512,
        help="Max long side for each ROI frame before building the contact sheet. Use 0 to keep original size.",
    )
    parser.add_argument("--columns", type=int, default=4, help="Contact sheet columns")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale applied to cropped frames before making sheet")
    parser.add_argument("--max-image-side", type=int, default=2048, help="Downscale contact sheet to this max side")
    parser.add_argument("--jpeg-quality", type=int, default=88, help="Contact sheet JPEG quality")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional model sampling temperature. Omitted by default because reasoning models often require the default.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to wait between API calls")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N videos")
    parser.add_argument("--dry-run", action="store_true", help="Generate sheets and output files without API calls")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.openai_frame_image_size < 0:
        parser.error("--openai-frame-image-size must be greater than or equal to 0")
    if args.columns <= 0:
        parser.error("--columns must be positive")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if args.max_image_side < 0:
        parser.error("--max-image-side must be greater than or equal to 0")
    if args.roi_detect_image_size < 0:
        parser.error("--roi-detect-image-size must be greater than or equal to 0")
    if args.reasoning_effort == "none":
        args.reasoning_effort = None
    return classify_videos(args)


if __name__ == "__main__":
    raise SystemExit(main())
