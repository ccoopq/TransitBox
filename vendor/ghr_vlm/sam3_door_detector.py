#!/usr/bin/env python3
"""Detect prompted objects with Transformers SAM3 and write bbox JSON.

This adapter is used by demo_flow.py. It loads facebook/sam3 through
Transformers and supports both the legacy single-best detection and batched
all-instance detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="Input RGB image path.")
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Batch of RGB image paths. The model is loaded once for the batch.",
    )
    parser.add_argument("--prompt", default="bus front door", help="SAM3 text prompt.")
    parser.add_argument(
        "--prompt-variants",
        default="",
        help="Additional prompts separated by '|'. They are tried after --prompt.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--model",
        default="facebook/sam3",
        help="Transformers SAM3 model id or local path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Instance score threshold for SAM3 post-processing.",
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated thresholds to try. Defaults to --threshold only.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold for SAM3 post-processing.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional overlay image path for debugging.",
    )
    parser.add_argument(
        "--all-instances",
        action="store_true",
        help="Return every detected instance instead of only the best instance.",
    )
    parser.add_argument(
        "--track-video",
        action="store_true",
        help="Use SAM3 Video to detect and track text-matched instances across --images.",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.005,
        help=(
            "Remove tracked boxes smaller than this fraction of the frame area. "
            "Default 0.005 means 0.5%%."
        ),
    )
    parser.add_argument(
        "--max-width-height-ratio",
        type=float,
        default=1.0,
        help=(
            "Remove tracked boxes whose width/height ratio is above this value. "
            "Default 1.0 keeps boxes no wider than tall."
        ),
    )
    return parser.parse_args()


def parse_prompt_list(primary_prompt: str, prompt_variants: str) -> List[str]:
    prompts = [primary_prompt]
    if prompt_variants.strip():
        prompts.extend(prompt_variants.split("|"))

    normalized: List[str] = []
    seen = set()
    for prompt in prompts:
        text = prompt.strip()
        if not text or text.lower() in seen:
            continue
        normalized.append(text)
        seen.add(text.lower())
    return normalized


def parse_thresholds(primary_threshold: float, threshold_text: str) -> List[float]:
    values = [primary_threshold]
    if threshold_text.strip():
        for item in threshold_text.split(","):
            item = item.strip()
            if not item:
                continue
            values.append(float(item))

    normalized: List[float] = []
    for value in values:
        clipped = max(0.0, min(1.0, float(value)))
        if clipped not in normalized:
            normalized.append(clipped)
    return sorted(normalized, reverse=True)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_box(box: Any, width: int, height: int) -> Tuple[float, float, float, float]:
    flat = tensor_to_numpy(box).astype(np.float32).reshape(-1)
    if flat.size < 4:
        raise RuntimeError(f"SAM3 returned an invalid box: {flat.tolist()}")
    x1, y1, x2, y2 = [float(v) for v in flat[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"SAM3 returned a degenerate box: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def bbox_from_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Selected SAM3 mask is empty.")
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    box: Tuple[float, float, float, float],
    path: Path,
) -> None:
    raw = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = raw.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        0.45 * overlay[mask_bool] + 0.55 * np.array([255, 0, 0])
    ).astype(np.uint8)
    blended = Image.fromarray(overlay)
    draw = ImageDraw.Draw(blended)
    draw.rectangle(box, outline="lime", width=max(2, int(round(max(image.size) / 250))))
    path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(path)


def load_transformers_sam3(model_id: str, device: str) -> Tuple[Any, Any]:
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "Transformers with SAM3 support is required. Install or upgrade with: "
            "pip install -U transformers accelerate"
        ) from exc

    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(model_id)
    return model, processor


def load_transformers_sam3_video(
    model_id: str,
    device: torch.device,
) -> Tuple[Any, Any, torch.dtype]:
    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
    except ImportError as exc:
        raise ImportError(
            "Transformers with SAM3 Video support is required. Install or upgrade "
            "with: pip install -U transformers accelerate"
        ) from exc

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Sam3VideoModel.from_pretrained(model_id).to(device=device, dtype=dtype)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    return model, processor, dtype


def choose_best_sam3_detection(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    prompts: List[str],
    thresholds: List[float],
    mask_threshold: float,
    device: torch.device,
    prefer_more_instances: bool = False,
) -> Optional[Dict[str, Any]]:
    attempted: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = inputs.get("original_sizes").tolist()
        for threshold in thresholds:
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )[0]
            masks = results.get("masks")
            boxes = results.get("boxes")
            scores = results.get("scores")
            count = 0 if scores is None else len(scores)
            attempted.append(
                {
                    "prompt": prompt,
                    "threshold": threshold,
                    "detection_count": int(count),
                }
            )
            if masks is None or scores is None or count == 0:
                continue

            scores_np = tensor_to_numpy(scores).reshape(-1)
            best_idx = int(np.argmax(scores_np))
            confidence = float(scores_np[best_idx])
            candidate = {
                "prompt": prompt,
                "threshold": threshold,
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
                "best_idx": best_idx,
                "confidence": confidence,
                "detection_count": int(count),
                "attempted": attempted,
            }
            if prefer_more_instances:
                should_replace = (
                    best is None
                    or threshold > float(best["threshold"])
                    or (
                        threshold == float(best["threshold"])
                        and confidence > float(best["confidence"])
                    )
                )
            else:
                should_replace = (
                    best is None or confidence > float(best["confidence"])
                )
            if should_replace:
                best = candidate

    if best is not None:
        best["attempted"] = attempted
    return best


def serialize_detections(
    detection: Dict[str, Any],
    width: int,
    height: int,
    all_instances: bool,
) -> List[Dict[str, Any]]:
    masks = detection["masks"]
    boxes = detection["boxes"]
    scores = tensor_to_numpy(detection["scores"]).reshape(-1)
    indices = list(range(len(scores))) if all_instances else [int(detection["best_idx"])]
    indices.sort(key=lambda index: float(scores[index]), reverse=True)

    serialized: List[Dict[str, Any]] = []
    for index in indices:
        mask = tensor_to_numpy(masks[index]).astype(bool)
        try:
            if boxes is not None and len(boxes) > index:
                bbox = normalize_box(boxes[index], width, height)
            else:
                bbox = bbox_from_mask(mask)
        except RuntimeError:
            if boxes is None or len(boxes) <= index:
                continue
            try:
                bbox = bbox_from_mask(mask)
            except RuntimeError:
                continue
        serialized.append(
            {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "bbox": list(bbox),
                "confidence": float(scores[index]),
                "mask_area": int(mask.sum()),
            }
        )
    return serialized


def detect_image(
    *,
    image_path: Path,
    model: Any,
    processor: Any,
    prompts: List[str],
    thresholds: List[float],
    mask_threshold: float,
    device: torch.device,
    all_instances: bool,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    detection = choose_best_sam3_detection(
        model=model,
        processor=processor,
        image=image,
        prompts=prompts,
        thresholds=thresholds,
        mask_threshold=mask_threshold,
        device=device,
        prefer_more_instances=all_instances,
    )
    if detection is None:
        return {
            "image": str(image_path),
            "found": False,
            "image_width": width,
            "image_height": height,
            "detections": [],
        }

    detections = serialize_detections(
        detection,
        width=width,
        height=height,
        all_instances=all_instances,
    )
    return {
        "image": str(image_path),
        "found": bool(detections),
        "prompt": detection["prompt"],
        "image_width": width,
        "image_height": height,
        "threshold": detection["threshold"],
        "attempted": detection["attempted"],
        "detections": detections,
    }


def track_video_images(
    *,
    image_paths: List[Path],
    model_id: str,
    prompt: str,
    thresholds: List[float],
    device: torch.device,
    min_box_area_ratio: float,
    max_width_height_ratio: float,
) -> Dict[str, Any]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    model, processor, dtype = load_transformers_sam3_video(model_id, device)
    session = processor.init_video_session(
        video=images,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        inference_state_device=device,
        dtype=dtype,
    )
    session = processor.add_text_prompt(
        inference_session=session,
        text=prompt,
    )

    raw_frames: Dict[int, Dict[str, Any]] = {}
    with torch.inference_mode():
        iterator = model.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=0,
            max_frame_num_to_track=max(0, len(images) - 1),
            show_progress_bar=False,
        )
        for model_outputs in iterator:
            processed = processor.postprocess_outputs(session, model_outputs)
            frame_idx = int(model_outputs.frame_idx)
            raw_frames[frame_idx] = processed

    selected_threshold = thresholds[-1]
    for threshold in thresholds:
        if any(
            bool((tensor_to_numpy(frame["scores"]).reshape(-1) >= threshold).any())
            for frame in raw_frames.values()
        ):
            selected_threshold = threshold
            break

    image_results: List[Dict[str, Any]] = []
    for frame_idx, (image_path, image) in enumerate(zip(image_paths, images)):
        processed = raw_frames.get(frame_idx)
        detections: List[Dict[str, Any]] = []
        skipped_invalid_boxes = 0
        if processed is not None:
            object_ids = tensor_to_numpy(processed["object_ids"]).reshape(-1)
            scores = tensor_to_numpy(processed["scores"]).reshape(-1)
            boxes = tensor_to_numpy(processed["boxes"])
            masks = tensor_to_numpy(processed["masks"])
            for index, score in enumerate(scores):
                if float(score) < selected_threshold:
                    continue
                mask = np.asarray(masks[index]).astype(bool)
                try:
                    bbox = normalize_box(boxes[index], image.size[0], image.size[1])
                except RuntimeError:
                    try:
                        bbox = bbox_from_mask(mask)
                    except RuntimeError:
                        skipped_invalid_boxes += 1
                        continue
                detections.append(
                    {
                        "object_id": int(object_ids[index]),
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "bbox": list(bbox),
                        "confidence": float(score),
                        "mask_area": int(mask.sum()),
                    }
                )
        detections.sort(key=lambda item: int(item["object_id"]))
        image_results.append(
            {
                "image": str(image_path),
                "frame_index": frame_idx,
                "found": bool(detections),
                "prompt": prompt,
                "image_width": image.size[0],
                "image_height": image.size[1],
                "threshold": selected_threshold,
                "skipped_invalid_boxes": skipped_invalid_boxes,
                "detections": detections,
            }
        )

    for image_result in image_results:
        filtered_detections = []
        frame_area = float(image_result["image_width"] * image_result["image_height"])
        for detection in image_result["detections"]:
            box_width = float(detection["x2"]) - float(detection["x1"])
            box_height = float(detection["y2"]) - float(detection["y1"])
            aspect_ratio = box_width / box_height if box_height > 0 else 0.0
            box_area_ratio = (
                box_width * box_height / frame_area
                if frame_area > 0 and box_width > 0 and box_height > 0
                else 0.0
            )
            detection["width_height_ratio"] = aspect_ratio
            detection["box_area_ratio"] = box_area_ratio
            if (
                aspect_ratio <= max_width_height_ratio
                and box_area_ratio >= min_box_area_ratio
            ):
                filtered_detections.append(detection)
        image_result["detections"] = filtered_detections

    object_frames: Dict[int, List[int]] = {}
    for frame_idx, image_result in enumerate(image_results):
        for detection in image_result["detections"]:
            object_id = int(detection["object_id"])
            object_frames.setdefault(object_id, []).append(frame_idx)

    continuous_object_ids = {
        object_id
        for object_id, frame_indices in object_frames.items()
        if any(
            current == previous + 1
            for previous, current in zip(frame_indices, frame_indices[1:])
        )
    }
    for image_result in image_results:
        image_result["detections"] = [
            detection
            for detection in image_result["detections"]
            if int(detection["object_id"]) in continuous_object_ids
        ]
        image_result["found"] = bool(image_result["detections"])

    return {
        "requested_prompt": prompt,
        "source": "transformers_sam3_video",
        "model": model_id,
        "thresholds": thresholds,
        "threshold": selected_threshold,
        "tracking": True,
        "postprocessing": {
            "maximum_width_height_ratio": max_width_height_ratio,
            "minimum_box_area_ratio": min_box_area_ratio,
            "minimum_consecutive_frames": 2,
            "retained_object_ids": sorted(continuous_object_ids),
        },
        "images": image_results,
    }


def main() -> int:
    args = parse_args()
    if not args.image and not args.images:
        raise ValueError("Provide --image or --images.")
    if args.image and args.images:
        raise ValueError("Use only one of --image or --images.")
    if args.track_video and not args.images:
        raise ValueError("--track-video requires --images.")
    if not 0.0 <= args.min_box_area_ratio <= 1.0:
        raise ValueError("--min-box-area-ratio must be between 0 and 1.")
    if args.max_width_height_ratio <= 0:
        raise ValueError("--max-width-height-ratio must be greater than 0.")

    output_path = Path(args.output).expanduser().resolve()
    device = torch.device(args.device)
    prompts = parse_prompt_list(args.prompt, args.prompt_variants)
    thresholds = parse_thresholds(args.threshold, args.thresholds)

    if args.track_video:
        image_paths = [Path(path).expanduser().resolve() for path in args.images]
        output_payload = track_video_images(
            image_paths=image_paths,
            model_id=args.model,
            prompt=prompts[0],
            thresholds=thresholds,
            device=device,
            min_box_area_ratio=args.min_box_area_ratio,
            max_width_height_ratio=args.max_width_height_ratio,
        )
    elif args.images:
        model, processor = load_transformers_sam3(args.model, str(device))
        image_paths = [Path(path).expanduser().resolve() for path in args.images]
        image_results = [
            detect_image(
                image_path=image_path,
                model=model,
                processor=processor,
                prompts=prompts,
                thresholds=thresholds,
                mask_threshold=args.mask_threshold,
                device=device,
                all_instances=args.all_instances,
            )
            for image_path in image_paths
        ]
        output_payload = {
            "requested_prompt": args.prompt,
            "prompt_variants": prompts,
            "source": "transformers_sam3",
            "model": args.model,
            "thresholds": thresholds,
            "mask_threshold": args.mask_threshold,
            "all_instances": args.all_instances,
            "images": image_results,
        }
    else:
        model, processor = load_transformers_sam3(args.model, str(device))
        image_path = Path(args.image).expanduser().resolve()
        result = detect_image(
            image_path=image_path,
            model=model,
            processor=processor,
            prompts=prompts,
            thresholds=thresholds,
            mask_threshold=args.mask_threshold,
            device=device,
            all_instances=args.all_instances,
        )
        if not result["found"]:
            raise RuntimeError(
                "No SAM3 prediction for prompts "
                f"{prompts!r} at thresholds {thresholds!r}."
            )
        best = result["detections"][0]
        output_payload = {
            **result,
            **best,
            "requested_prompt": args.prompt,
            "prompt_variants": prompts,
            "source": "transformers_sam3",
            "model": args.model,
            "thresholds": thresholds,
            "mask_threshold": args.mask_threshold,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    if args.overlay and not args.images:
        image = Image.open(image_path).convert("RGB")
        detection = choose_best_sam3_detection(
            model=model,
            processor=processor,
            image=image,
            prompts=prompts,
            thresholds=thresholds,
            mask_threshold=args.mask_threshold,
            device=device,
            prefer_more_instances=args.all_instances,
        )
        if detection is not None:
            best_idx = int(detection["best_idx"])
            mask = tensor_to_numpy(detection["masks"][best_idx]).astype(bool)
            bbox = tuple(output_payload["bbox"])
            save_overlay(
                image,
                mask,
                bbox,
                Path(args.overlay).expanduser().resolve(),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
