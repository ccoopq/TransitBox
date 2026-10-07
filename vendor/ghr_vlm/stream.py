'''
python stream.py --dataset_root dataset/C3_3.mp4  \
    --door_track yolo --track sam3 --st_time 390 --ed_time 540
'''

from __future__ import annotations 

import argparse
import base64
import gc
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Helps PyTorch reuse fragmented CUDA memory during long multi-video runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
except ImportError:
    torch = None

try:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    T = None
    InterpolationMode = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    AutoConfig = None
    AutoModel = None
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


QWEN2_5_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN3_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
INTERNVL3_5_MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
MODEL_NAME = QWEN3_MODEL_NAME
OPENAI_API_KEY = ""  # Configure credentials using environment variables.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_INPUT_SIZE = 448
INTERNVL_SINGLE_IMAGE_MAX_TILES = 12
INTERNVL_MULTI_IMAGE_MAX_TILES = 1


def default_sam3d_door_command() -> str:
    script_path = Path(__file__).with_name("sam3_door_detector.py")
    return (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} "
        "--model facebook/sam3 "
        "--image {image} --prompt {prompt} --prompt-variants {prompt_variants} "
        "--thresholds {thresholds} --output {output}"
    )

DOOR_CHANGE_INSTRUCTION = """You are monitoring a continuous bus surveillance video clip.

You will see one contact-sheet image containing frames sampled in chronological order from one short clip. The clip duration is provided in the prompt text.

The front door region is highlighted with red bounding boxes. The door may be closed, open, or in motion. Your task is to detect whether the bus front door changes status during this clip.

Possible transitions:
- closed_to_open: the door opens to let passengers in
- open_to_closed: the door closes after passengers board
- none: no door status change

If there is a status change, find the sampled frame number where the status change is first clearly visible. For closed_to_open, select the frame where the door just opens. For open_to_closed, select the frame where the door just closes. Use the 1-based frame numbers printed in the contact-sheet tiles. If the change happens between two sampled frames, choose the later frame where the new door state is first visible.

If there is no status change, set change_frame_index to 0.

Sometimes passenger movement or occlusion can make the door appear to change status. That's FAKE!

Return only valid JSON:
{
  "status_changed": true or false,
  "transition": "closed_to_open" or "open_to_closed" or "none",
  "change_frame_index": frame number where the change is first clearly visible, or 0 if none,
  "confidence": confidence score between 0.0 and 1.0, where 1.0 is very confident,
  "reasoning": "brief temporal visual reason"
}"""

DOOR_CHANGE_VERIFICATION_INSTRUCTION = """It is reported that the bus front door changes status during this clip.

Possible transitions:
- closed_to_open: the door opens to let passengers in
- open_to_closed: the door closes after passengers board
- none: no door status change

You should verify whether the reported door status change is real or fake. If the change is real, confirm the transition type and the 1-based frame index where the status change is first clearly visible. If the change is fake, set status_changed to false and use zeros for the other fields.

Especially check passenger movement or occlusion which can make the door appear to change status. That's FAKE!

Return only valid JSON:
{
  "status_changed": true or false,
  "transition": "closed_to_open" or "open_to_closed" or "none",
  "change_frame_index": frame number where the change is first clearly visible, or 0 if none,
  "confidence": confidence score between 0.0 and 1.0, where 1.0 is very confident,
  "reasoning": "brief temporal visual reason"
}"""

DOOR_BBOX_INSTRUCTION = """You are locating the front passenger entrance door of a bus in one surveillance frame.

Task:
Find the bus front entrance door area. This is the door used by boarding passengers near the front of the bus. Return one tight bounding box around the visible moving door region, including the door panels and doorway area needed to judge whether the door is open or closed.

Use pixel coordinates in the provided image, with origin at the top-left. If the door is partly occluded, return the visible door area. If the front entrance door is not visible, set found to false and use zeros for the coordinates.

Return only valid JSON:
{
  "found": true,
  "x1": 0.0,
  "y1": 0.0,
  "x2": 0.0,
  "y2": 0.0,
  "confidence": 0.0,
  "reasoning": "brief visual reason"
}"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run streaming bus payment analysis: detect stops, split passengers, "
            "and save each passenger payment clip locally."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help=(
            "Path to the continuous bus video for streaming flow inference, "
            "for example dataset/video.mp4."
        ),
    )
    parser.add_argument(
        "--st_time",
        type=float,
        default=0.0,
        help="Start time in seconds for streaming video processing.",
    )
    parser.add_argument(
        "--ed_time",
        type=float,
        default=None,
        help=(
            "Optional end time in seconds for streaming video processing. "
            "If omitted, process until the end of the video."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where results.json and results.csv will be written.",
    )
    parser.add_argument(
        "--door_change_window",
        type=float,
        default=10.0,
        help="Seconds per sliding clip for door status-change detection.",
    )
    parser.add_argument(
        "--door_track",
        choices=["vlm", "yolo"],
        default="vlm",
        help=(
            "Door status tracking backend. Use vlm for the original VLM sliding "
            "window status-change detector, or yolo for the trained YOLO11 "
            "open/closed classifier on every frame."
        ),
    )
    parser.add_argument(
        "--door_yolo_model",
        type=str,
        default="runs/classify/runs/door_status/yolo11s_doorset/weights/best.pt",
        help="Trained YOLO11 door open/closed classifier used when --door_track yolo.",
    )
    parser.add_argument(
        "--door_yolo_batch_size",
        type=int,
        default=64,
        help="Number of cropped door frames classified per YOLO call.",
    )
    parser.add_argument(
        "--door_yolo_min_consecutive_frames",
        type=int,
        default=3,
        help=(
            "Confirm a YOLO door status change only after the new status appears "
            "for this many consecutive frames. Use 1 for raw frame-by-frame changes."
        ),
    )
    parser.add_argument(
        "--door_yolo_crop_margin",
        type=float,
        default=0.0,
        help=(
            "Fractional padding around the door ROI before YOLO classification. "
            "Default 0.0 matches the generated training dataset crop."
        ),
    )
    parser.add_argument(
        "--door_yolo_min_confidence",
        type=float,
        default=0.80,
        help=(
            "Minimum YOLO door classification confidence accepted for a frame. "
            "Lower-confidence frames reuse the previous door status."
        ),
    )
    parser.add_argument(
        "--door_change_overlap",
        type=float,
        default=2.0,
        help="Seconds of overlap between adjacent door status-change clips.",
    )
    parser.add_argument(
        "--door_change_frames",
        type=int,
        default=12,
        help="Number of query frames sampled from each door status-change clip.",
    )
    parser.add_argument(
        "--door_change_merge_tolerance",
        type=float,
        default=3,
        help="Seconds within which duplicate door status-change pins are merged.",
    )
    parser.add_argument(
        "--door_change_verification_radius",
        type=float,
        default=5.0,
        help=(
            "Seconds before and after a detected door-change timestamp to re-check "
            "with an independent VLM pass. Default 5.0 means t-5s to t+5s."
        ),
    )
    parser.add_argument(
        "--door_change_verification_frames",
        type=int,
        default=12,
        help=(
            "Number of query frames sampled from each door status-change "
            "verification clip. This is separate from --door_change_frames."
        ),
    )
    parser.add_argument(
        "--door_image_size",
        type=int,
        default=700,
        help="Max image size for temporary door-check frames. Use 0 to keep the original size.",
    )
    parser.add_argument(
        "--door_roi_margin",
        type=float,
        default=0.20,
        help=(
            "Fractional padding added around the detected front-door ROI before "
            "cropping door-check frames. Use 0 for a tight crop."
        ),
    )
    parser.add_argument(
        "--door_roi_detector",
        type=str,
        choices=["sam3d", "vlm", "none"],
        default="sam3d",
        help=(
            "Detector used for the first-frame front-door ROI. sam3d calls "
            "--sam3d_door_command, vlm uses the clip VLM, none uses full frames."
        ),
    )
    parser.add_argument(
        "--door_roi_bbox_path",
        type=str,
        default="outputs/door_roi_bbox.json",
        help=(
            "Cached door ROI bbox JSON to reuse before running the ROI detector. "
            "Default reads outputs/door_roi_bbox.json."
        ),
    )
    parser.add_argument(
        "--disable_door_roi",
        action="store_true",
        help="Deprecated alias for --door_roi_detector none.",
    )
    parser.add_argument(
        "--sam3d_door_command",
        type=str,
        default=os.environ.get("SAM3D_DOOR_COMMAND", default_sam3d_door_command()),
        help=(
            "Command used to run SAM3D front-door detection. It should write a JSON "
            "file with x1/y1/x2/y2 or bbox fields. Template variables are "
            "{image}, {output}, {prompt}, {prompt_variants}, {thresholds}, "
            "and optional {overlay}. Defaults to sam3_door_detector.py with "
            "Transformers facebook/sam3, or uses SAM3D_DOOR_COMMAND when set."
        ),
    )
    parser.add_argument(
        "--sam3d_door_prompt",
        type=str,
        default="front passenger entrance door",
        help="Text prompt passed to SAM3D for door ROI detection.",
    )
    parser.add_argument(
        "--sam3d_door_prompt_variants",
        type=str,
        default=(
            "bus front door|front bus door|bus entrance door|bus passenger door|"
            "vehicle door|door"
        ),
        help="Additional SAM3 door prompts separated by '|'.",
    )
    parser.add_argument(
        "--sam3d_door_thresholds",
        type=str,
        default="0.5,0.35,0.2,0.1",
        help="Comma-separated SAM3 score thresholds to try before failing.",
    )
    parser.add_argument(
        "--sam3d_door_image_size",
        type=int,
        default=0,
        help=(
            "Max image size for the SAM3 door localization frame. "
            "Default 0 keeps full resolution; this is separate from --door_image_size."
        ),
    )
    parser.add_argument(
        "--sam3d_door_timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for the SAM3D door detection command.",
    )
    parser.add_argument(
        "--sam3d_allow_full_frame_fallback",
        action="store_true",
        help="If SAM3D door detection fails, continue with full frames instead of stopping.",
    )
    parser.add_argument(
        "--passenger_split_image_size",
        type=int,
        default=700,
        help=(
            "Max image size for passenger-splitting frames. "
            "Use 0 to keep the original size."
        ),
    )
    parser.add_argument(
        "--passenger_split_frames",
        type=int,
        default=16,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--passenger_split_frame_stride",
        type=int,
        default=2,
        help=(
            "Use every Nth original video frame for passenger tracking and "
            "the debug contact sheet."
        ),
    )
    parser.add_argument(
        "--track",
        choices=["sam3", "yolo"],
        default="sam3",
        help=(
            "Passenger tracking backend. Use sam3 for the existing SAM3 video "
            "tracking path, or yolo for YOLO11 detection with BoT-SORT tracking."
        ),
    )
    parser.add_argument(
        "--passenger_split_merge_frame_distance",
        type=int,
        default=3,
        help=(
            "Merge adjacent repaired-ID passenger runs whose start frame pins "
            "are within this many sampled frames. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--sam3d_passenger_prompt",
        type=str,
        default="passenger",
        help="SAM3 text prompt used to track passengers in sampled frames.",
    )
    parser.add_argument(
        "--sam3d_passenger_thresholds",
        type=str,
        default="0.40",
        help="Comma-separated SAM3 passenger score thresholds.",
    )
    parser.add_argument(
        "--sam3d_passenger_min_box_area_ratio",
        "--passenger_min_box_area_ratio",
        dest="sam3d_passenger_min_box_area_ratio",
        type=float,
        default=0.05,
        help="Minimum tracked passenger box area as a fraction of frame area.",
    )
    parser.add_argument(
        "--sam3d_passenger_max_width_height_ratio",
        "--passenger_max_width_height_ratio",
        dest="sam3d_passenger_max_width_height_ratio",
        type=float,
        default=1.0,
        help=(
            "Remove tracked passenger boxes whose width/height ratio is above "
            "this value."
        ),
    )
    parser.add_argument(
        "--sam3d_passenger_timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds for SAM3 passenger tracking.",
    )
    parser.add_argument(
        "--sam3d_passenger_batch_size",
        type=int,
        default=16,
        help="Maximum number of sampled passenger frames passed to SAM3 per call.",
    )
    parser.add_argument(
        "--yolo_passenger_model",
        type=str,
        default="yolo11s.pt",
        help="YOLO model used when --track yolo.",
    )
    parser.add_argument(
        "--yolo_passenger_conf",
        type=float,
        default=0.7,
        help="YOLO confidence threshold used when --track yolo.",
    )
    parser.add_argument(
        "--yolo_passenger_iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold used when --track yolo.",
    )
    parser.add_argument(
        "--yolo_passenger_tracker",
        type=str,
        default="botsort.yaml",
        help="Ultralytics tracker config used when --track yolo.",
    )
    parser.add_argument(
        "--no_lighting_preprocess",
        action="store_true",
        help=(
            "Disable passenger-stage lighting preprocessing. By default, "
            "passenger tracking frames and saved passenger clips brighten dark "
            "bus-interior regions and suppress very bright outside regions. "
            "Door checks use raw frames."
        ),
    )
    parser.add_argument(
        "--min_clip_duration",
        type=float,
        default=0.5,
        help="Minimum extracted payment clip duration in seconds.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=(
            "Default model used for door status changes and passenger segmentation "
            "when --model_clip is not set. Supports Qwen aliases and InternVL "
            f"aliases such as internvl or {INTERNVL3_5_MODEL_NAME}."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Deprecated alias for --model.",
    )
    parser.add_argument(
        "--model_clip",
        type=str,
        default=None,
        help=(
            "Model for door status changes and passenger segmentation, "
            f"for example gpt-5.4 or {INTERNVL3_5_MODEL_NAME}."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "local", "openai"],
        default="auto",
        help=(
            "Inference backend. auto resolves the clip-understanding model: "
            "GPT/o-series models use OpenAI, otherwise a local VLM."
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, uses OPENAI_API_KEY env var or OPENAI_API_KEY in this script.",
    )
    parser.add_argument(
        "--openai_detail",
        type=str,
        choices=["low", "auto", "high"],
        default="low",
        help="OpenAI image detail level. low is cheaper and safer for many sampled frames.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per video.",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantized loading and use bfloat16/fp16 directly.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    aliases = {
        "qwen": MODEL_NAME,
        "qwen3": QWEN3_MODEL_NAME,
        "qwen3-vl": QWEN3_MODEL_NAME,
        "qwen3-vl-8b": QWEN3_MODEL_NAME,
        "qwen2.5": QWEN2_5_MODEL_NAME,
        "qwen2.5-vl": QWEN2_5_MODEL_NAME,
        "qwen2.5-vl-7b": QWEN2_5_MODEL_NAME,
        "qwen2_5": QWEN2_5_MODEL_NAME,
        "qwen2_5_vl": QWEN2_5_MODEL_NAME,
        "qwen2_5_vl_7b": QWEN2_5_MODEL_NAME,
        "internvl": INTERNVL3_5_MODEL_NAME,
        "internvl3.5": INTERNVL3_5_MODEL_NAME,
        "internvl3_5": INTERNVL3_5_MODEL_NAME,
        "internvl3.5-8b": INTERNVL3_5_MODEL_NAME,
        "internvl3_5-8b": INTERNVL3_5_MODEL_NAME,
        "gpt5": "gpt-5",
        "gpt-5": "gpt-5",
    }
    return aliases.get(normalized.lower(), normalized)


def resolve_local_model_family(model_name: str) -> str:
    lower_model = model_name.lower()
    if "internvl" in lower_model:
        return "internvl"
    return "qwen"


def resolve_model_and_backend(model_name: str, backend: str) -> Tuple[str, str]:
    model = normalize_model_name(model_name)
    if backend == "auto":
        lower_model = model.lower()
        if lower_model.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt-")):
            backend = "openai"
        else:
            backend = "local"
    return model, backend


def require_common_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if pd is None:
        missing.append("pandas")
    if Image is None:
        missing.append("pillow")

    if missing:
        unique_missing = sorted(set(missing))
        raise ImportError(
            "Missing required package(s): "
            + ", ".join(unique_missing)
            + "\nInstall dependencies with:\n"
            + "pip install torch torchvision transformers accelerate "
            + "qwen-vl-utils opencv-python pillow pandas bitsandbytes"
        )


def require_local_dependencies(model_name: str) -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if resolve_local_model_family(model_name) == "internvl":
        if AutoModel is None or AutoTokenizer is None:
            missing.append("transformers")
        if T is None or InterpolationMode is None:
            missing.append("torchvision")
    elif AutoProcessor is None or BitsAndBytesConfig is None:
        missing.append("transformers")

    if missing:
        unique_missing = sorted(set(missing))
        model_family = resolve_local_model_family(model_name)
        raise ImportError(
            f"Missing required package(s) for local {model_family} inference: "
            + ", ".join(unique_missing)
            + "\nInstall dependencies with:\n"
            + "pip install torch torchvision transformers accelerate "
            + "qwen-vl-utils bitsandbytes"
        )


def require_openai_dependencies() -> None:
    if OpenAI is None:
        raise ImportError(
            "Missing required package for OpenAI API inference: openai\n"
            "Install it with:\n"
            "pip install openai"
        )


def get_openai_api_key(cli_api_key: Optional[str]) -> str:
    api_key = cli_api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OpenAI API key is required for --backend openai. "
            "Set OPENAI_API_KEY, pass --api_key, or fill OPENAI_API_KEY in demo.py."
        )
    return api_key


def get_torch_dtype() -> "torch.dtype":
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def cleanup_memory() -> None:
    """Release per-video Python and CUDA allocations while keeping the model loaded."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def close_frames(frames: List[Any]) -> None:
    for frame in frames:
        close = getattr(frame, "close", None)
        if close is not None:
            close()


def resolve_local_model_class(model_name: str) -> Any:
    if AutoConfig is None:
        raise ImportError("Could not import AutoConfig. Please upgrade transformers.")

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as exc:
        if "qwen3_vl" in str(exc).lower():
            raise ImportError(
                "Your installed transformers does not recognize Qwen3-VL yet. "
                "Install a newer transformers build, for example:\n"
                "pip install git+https://github.com/huggingface/transformers"
            ) from exc
        raise

    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_vl":
        candidates = [
            Qwen3VLForConditionalGeneration,
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoModelForVision2Seq,
        ]
        for candidate in candidates:
            if candidate is not None:
                return candidate
        raise ImportError(
            "Qwen3-VL requires Qwen3VLForConditionalGeneration or a compatible "
            "AutoModel class. Please upgrade transformers, for example:\n"
            "pip install git+https://github.com/huggingface/transformers"
        )

    if model_type == "qwen2_5_vl":
        candidates = [Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq]
    else:
        candidates = [
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoModelForVision2Seq,
            Qwen2_5_VLForConditionalGeneration,
        ]

    for candidate in candidates:
        if candidate is not None:
            return candidate

    raise ImportError(
        "Could not import a compatible local vision-language model class. "
        "Please upgrade transformers."
    )


def load_internvl_model_and_tokenizer(
    model_name: str, prefer_quantized: bool = True
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    dtype = get_torch_dtype()
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["use_flash_attn"] = True

    def from_pretrained_with_flash_fallback(**kwargs: Any) -> Any:
        try:
            return AutoModel.from_pretrained(model_name, **kwargs)
        except Exception:
            if not kwargs.get("use_flash_attn"):
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs["use_flash_attn"] = False
            print("InternVL FlashAttention loading failed; retrying without FlashAttention.")
            return AutoModel.from_pretrained(model_name, **retry_kwargs)

    if torch.cuda.is_available() and prefer_quantized:
        try:
            print("Loading InternVL in 8-bit quantized mode...")
            model = from_pretrained_with_flash_fallback(
                load_in_8bit=True,
                **load_kwargs,
            )
            model.eval()
            return model, tokenizer
        except Exception as exc:
            print(f"8-bit InternVL loading unavailable, falling back to normal loading: {exc}")

    print(f"Loading InternVL with dtype={dtype}...")
    model = from_pretrained_with_flash_fallback(**load_kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def load_model_and_processor(
    model_name: str, prefer_4bit: bool = True
) -> Tuple[Any, Any]:
    if resolve_local_model_family(model_name) == "internvl":
        return load_internvl_model_and_tokenizer(
            model_name,
            prefer_quantized=prefer_4bit,
        )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model_class = resolve_local_model_class(model_name)

    if torch.cuda.is_available() and prefer_4bit:
        try:
            print("Loading model in 4-bit quantized mode...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
            )
            model = model_class.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            return model, processor
        except Exception as exc:
            print(f"4-bit loading unavailable, falling back to normal loading: {exc}")

    dtype = get_torch_dtype()
    device_map = "auto" if torch.cuda.is_available() else None
    print(f"Loading model with dtype={dtype}...")
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, processor


def resize_frame_if_needed(frame: Image.Image, max_image_size: int) -> Image.Image:
    if max_image_size <= 0:
        return frame

    width, height = frame.size
    long_side = max(width, height)
    if long_side <= max_image_size:
        return frame

    scale = max_image_size / long_side
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return frame.resize(new_size, Image.Resampling.LANCZOS)


def preprocess_bus_lighting(frame: Image.Image) -> Image.Image:
    if cv2 is None or np is None:
        return frame

    rgb = np.array(frame.convert("RGB"))
    if rgb.size == 0:
        return frame

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    ).astype("float32") / 255.0

    luminance = (
        0.2126 * enhanced[:, :, 0]
        + 0.7152 * enhanced[:, :, 1]
        + 0.0722 * enhanced[:, :, 2]
    )
    shadow_mask = np.clip((0.55 - luminance) / 0.55, 0.0, 1.0)
    highlight_mask = np.clip((luminance - 0.68) / 0.32, 0.0, 1.0)

    enhanced *= 1.0 + 0.65 * shadow_mask[:, :, None]
    enhanced *= 1.0 - 0.35 * highlight_mask[:, :, None]
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return Image.fromarray((enhanced * 255.0).astype("uint8"))


def crop_frame_to_normalized_bbox(
    frame: Image.Image,
    bbox: Dict[str, float],
    margin: float = 0.0,
) -> Image.Image:
    width, height = frame.size
    x1 = max(0.0, min(1.0, coerce_float(bbox.get("x1_norm"), 0.0)))
    y1 = max(0.0, min(1.0, coerce_float(bbox.get("y1_norm"), 0.0)))
    x2 = max(0.0, min(1.0, coerce_float(bbox.get("x2_norm"), 1.0)))
    y2 = max(0.0, min(1.0, coerce_float(bbox.get("y2_norm"), 1.0)))
    if x2 <= x1 or y2 <= y1:
        return frame

    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(0.0, margin) * box_width
    pad_y = max(0.0, margin) * box_height
    left = int(round(max(0.0, x1 - pad_x) * width))
    top = int(round(max(0.0, y1 - pad_y) * height))
    right = int(round(min(1.0, x2 + pad_x) * width))
    bottom = int(round(min(1.0, y2 + pad_y) * height))
    if right <= left or bottom <= top:
        return frame
    return frame.crop((left, top, right, bottom))


def normalized_bbox_to_pixels(
    frame: Image.Image,
    bbox: Dict[str, float],
) -> Optional[Tuple[int, int, int, int]]:
    width, height = frame.size
    x1 = max(0.0, min(1.0, coerce_float(bbox.get("x1_norm"), 0.0)))
    y1 = max(0.0, min(1.0, coerce_float(bbox.get("y1_norm"), 0.0)))
    x2 = max(0.0, min(1.0, coerce_float(bbox.get("x2_norm"), 1.0)))
    y2 = max(0.0, min(1.0, coerce_float(bbox.get("y2_norm"), 1.0)))
    if x2 <= x1 or y2 <= y1:
        return None
    left = int(round(x1 * width))
    top = int(round(y1 * height))
    right = int(round(x2 * width))
    bottom = int(round(y2 * height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def load_overlay_font(frame: Image.Image) -> Any:
    font_size = max(28, int(round(min(frame.size) * 0.06)))
    if ImageFont is None:
        return None
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except OSError:
                continue
    try:
        return ImageFont.load_default()
    except OSError:
        return None


def draw_labeled_bbox(
    frame: Image.Image,
    bbox: Dict[str, float],
    label: str = "front door",
) -> Image.Image:
    if ImageDraw is None:
        return frame

    pixel_box = normalized_bbox_to_pixels(frame, bbox)
    if pixel_box is None:
        return frame

    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    left, top, right, bottom = pixel_box
    width = max(6, int(round(max(annotated.size) / 140)))
    color = (255, 0, 0)
    draw.rectangle((left, top, right, bottom), outline=color, width=width)

    label_text = label.strip() or "front door"
    font = load_overlay_font(annotated)
    try:
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(label_text, font=font)
    pad = max(8, width)
    label_left = left
    label_top = max(0, top - text_height - 2 * pad)
    label_right = min(annotated.size[0], label_left + text_width + 2 * pad)
    label_bottom = min(annotated.size[1], label_top + text_height + 2 * pad)
    draw.rectangle((label_left, label_top, label_right, label_bottom), fill=color)
    draw.text(
        (label_left + pad, label_top + pad),
        label_text,
        fill=(255, 255, 255),
        font=font,
    )
    return annotated


def draw_tracked_passenger_bboxes(
    frame: Image.Image,
    bboxes: List[Dict[str, float]],
) -> Image.Image:
    if ImageDraw is None:
        return frame

    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    colors = (
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 190, 190),
        (220, 50, 210),
        (170, 110, 40),
    )
    font_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    )
    line_width = max(5, int(round(max(frame.size) / 140)))

    for index, bbox in enumerate(bboxes, start=1):
        pixel_box = normalized_bbox_to_pixels(annotated, bbox)
        if pixel_box is None:
            continue
        left, top, right, bottom = pixel_box
        object_id = coerce_int(bbox.get("object_id"), index)
        color = colors[max(0, object_id) % len(colors)]

        color_fill = Image.new(
            "RGB",
            (max(1, right - left), max(1, bottom - top)),
            color,
        )
        tinted_region = Image.blend(
            annotated.crop((left, top, right, bottom)).convert("RGB"),
            color_fill,
            0.28,
        )
        annotated.paste(tinted_region, (left, top))
        color_fill.close()
        tinted_region.close()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)

        text = f"passenger {object_id}"
        box_width = max(1, right - left)
        box_height = max(1, bottom - top)
        font = None
        text_width = text_height = 0
        if ImageFont is not None:
            start_size = max(
                12,
                min(
                    52,
                    max(12, box_height // 2),
                    max(12, box_width // max(1, len(text) // 2)),
                ),
            )
            for font_size in range(start_size, 7, -1):
                for font_path in font_paths:
                    try:
                        candidate = ImageFont.truetype(font_path, font_size)
                    except OSError:
                        continue
                    text_box = draw.textbbox((0, 0), text, font=candidate)
                    candidate_width = text_box[2] - text_box[0]
                    candidate_height = text_box[3] - text_box[1]
                    if candidate_width <= box_width - 6 and candidate_height <= box_height - 6:
                        font = candidate
                        text_width = candidate_width
                        text_height = candidate_height
                        break
                if font is not None:
                    break
        if font is None:
            font = ImageFont.load_default() if ImageFont is not None else None
            text_box = draw.textbbox((0, 0), text, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]

        pad = 4
        label_width = min(box_width, text_width + 2 * pad)
        label_height = min(box_height, text_height + 2 * pad)
        label_left = left + (box_width - label_width) // 2
        label_top = top + (box_height - label_height) // 2
        draw.rectangle(
            (
                label_left,
                label_top,
                label_left + label_width,
                label_top + label_height,
            ),
            fill=color,
        )
        draw.text(
            (
                label_left + max(0, (label_width - text_width) // 2),
                label_top + max(0, (label_height - text_height) // 2),
            ),
            text,
            fill=(255, 255, 255),
            font=font,
        )
    return annotated


def draw_passenger_bbox_outline(
    frame: Image.Image,
    bbox: Dict[str, float],
    object_id: int,
) -> Image.Image:
    if ImageDraw is None:
        return frame.copy()

    annotated = frame.copy()
    pixel_box = normalized_bbox_to_pixels(annotated, bbox)
    if pixel_box is None:
        return annotated

    colors = (
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 190, 190),
        (220, 50, 210),
        (170, 110, 40),
    )
    color = colors[max(0, object_id) % len(colors)]
    line_width = max(5, int(round(max(frame.size) / 140)))
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(pixel_box, outline=color, width=line_width)
    return annotated


def mask_outside_bbox_with_checkerboard(
    frame: Image.Image,
    bbox: Dict[str, float],
) -> Image.Image:
    pixel_box = normalized_bbox_to_pixels(frame, bbox)
    if pixel_box is None or ImageDraw is None:
        return frame

    width, height = frame.size
    tile_size = max(20, int(round(min(width, height) * 0.045)))
    masked = Image.new("RGB", frame.size, (224, 224, 224))
    draw = ImageDraw.Draw(masked)
    colors = ((224, 224, 224), (255, 255, 255))
    for top in range(0, height, tile_size):
        row = top // tile_size
        for left in range(0, width, tile_size):
            col = left // tile_size
            draw.rectangle(
                (
                    left,
                    top,
                    min(width, left + tile_size),
                    min(height, top + tile_size),
                ),
                fill=colors[(row + col) % 2],
            )

    left, top, right, bottom = pixel_box
    masked.paste(frame.crop((left, top, right, bottom)), (left, top))
    return masked


def load_contact_sheet_font(tile_size: Tuple[int, int]) -> Any:
    font_size = max(24, int(round(min(tile_size) * 0.06)))
    if ImageFont is None:
        return None
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except OSError:
                continue
    try:
        return ImageFont.load_default()
    except OSError:
        return None


def contact_sheet_grid(frame_count: int) -> Tuple[int, int]:
    if frame_count <= 1:
        return 1, 1
    rows = max(1, int(math.floor(math.sqrt(frame_count))))
    cols = int(math.ceil(frame_count / rows))
    return rows, cols


def contact_sheet_layout(frames: List[Image.Image]) -> Dict[str, int]:
    rows, cols = contact_sheet_grid(len(frames))
    tile_width = max(frame.size[0] for frame in frames)
    tile_height = max(frame.size[1] for frame in frames)
    pad = max(8, int(round(max(tile_width, tile_height) * 0.012)))
    label_band = max(34, int(round(tile_height * 0.08)))
    return {
        "rows": rows,
        "cols": cols,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "pad": pad,
        "label_band": label_band,
    }


def build_frame_contact_sheet(frames: List[Image.Image]) -> Image.Image:
    if not frames:
        raise ValueError("Cannot build a contact sheet with no frames.")

    layout = contact_sheet_layout(frames)
    rows = layout["rows"]
    cols = layout["cols"]
    tile_width = layout["tile_width"]
    tile_height = layout["tile_height"]
    pad = layout["pad"]
    label_band = layout["label_band"]
    sheet_width = cols * tile_width + (cols + 1) * pad
    sheet_height = rows * (tile_height + label_band) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet) if ImageDraw is not None else None
    font = load_contact_sheet_font((tile_width, tile_height))

    for index, frame in enumerate(frames, start=1):
        row = (index - 1) // cols
        col = (index - 1) % cols
        tile_x = pad + col * (tile_width + pad)
        tile_y = pad + row * (tile_height + label_band + pad)
        label_y = tile_y
        image_y = tile_y + label_band
        label_text = f"Frame {index}"
        if draw is not None:
            draw.rectangle(
                (tile_x, label_y, tile_x + tile_width, label_y + label_band),
                fill=(30, 30, 30),
            )
            try:
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                _, text_height = draw.textsize(label_text, font=font)
            draw.text(
                (tile_x + pad, label_y + max(0, (label_band - text_height) // 2)),
                label_text,
                fill=(255, 255, 255),
                font=font,
            )
        sheet.paste(frame.convert("RGB"), (tile_x, image_y))
    return sheet


def build_fixed_frame_contact_sheet(
    frames: List[Image.Image],
    *,
    rows: int,
    cols: int,
    labels: Optional[List[str]] = None,
) -> Image.Image:
    if not frames:
        raise ValueError("Cannot build a contact sheet with no frames.")
    if rows <= 0 or cols <= 0:
        raise ValueError("Contact sheet rows and cols must be positive.")

    tile_width = max(frame.size[0] for frame in frames)
    tile_height = max(frame.size[1] for frame in frames)
    pad = max(8, int(round(max(tile_width, tile_height) * 0.012)))
    label_band = max(34, int(round(tile_height * 0.08)))
    sheet_width = cols * tile_width + (cols + 1) * pad
    sheet_height = rows * (tile_height + label_band) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet) if ImageDraw is not None else None
    font = load_contact_sheet_font((tile_width, tile_height))

    tile_count = rows * cols
    for index in range(tile_count):
        frame = frames[index] if index < len(frames) else frames[-1]
        row = index // cols
        col = index % cols
        tile_x = pad + col * (tile_width + pad)
        tile_y = pad + row * (tile_height + label_band + pad)
        label_y = tile_y
        image_y = tile_y + label_band
        label_text = (
            labels[index]
            if labels is not None and index < len(labels)
            else f"Frame {index + 1}"
        )
        if draw is not None:
            draw.rectangle(
                (tile_x, label_y, tile_x + tile_width, label_y + label_band),
                fill=(30, 30, 30),
            )
            try:
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                _, text_height = draw.textsize(label_text, font=font)
            draw.text(
                (tile_x + pad, label_y + max(0, (label_band - text_height) // 2)),
                label_text,
                fill=(255, 255, 255),
                font=font,
            )
        sheet.paste(frame.convert("RGB"), (tile_x, image_y))
    return sheet


def highlight_contact_sheet_frame(
    contact_sheet: Image.Image,
    frames: List[Image.Image],
    selected_frame_index: int,
) -> Image.Image:
    if (
        ImageDraw is None
        or selected_frame_index <= 0
        or selected_frame_index > len(frames)
    ):
        return contact_sheet

    layout = contact_sheet_layout(frames)
    cols = layout["cols"]
    tile_width = layout["tile_width"]
    tile_height = layout["tile_height"]
    pad = layout["pad"]
    label_band = layout["label_band"]

    row = (selected_frame_index - 1) // cols
    col = (selected_frame_index - 1) % cols
    left = pad + col * (tile_width + pad)
    top = pad + row * (tile_height + label_band + pad)
    right = left + tile_width
    bottom = top + label_band + tile_height

    highlighted = contact_sheet.copy()
    draw = ImageDraw.Draw(highlighted)
    color = (255, 215, 0)
    border_width = max(8, int(round(max(tile_width, tile_height) * 0.018)))
    for inset in range(border_width):
        draw.rectangle(
            (left - inset, top - inset, right + inset, bottom + inset),
            outline=color,
        )
    return highlighted


def highlight_contact_sheet_frames(
    contact_sheet: Image.Image,
    frames: List[Image.Image],
    selected_frame_indices: List[int],
) -> Image.Image:
    valid_indices = sorted(
        {
            index
            for index in selected_frame_indices
            if 1 <= index <= len(frames)
        }
    )
    if ImageDraw is None or not valid_indices:
        return contact_sheet

    layout = contact_sheet_layout(frames)
    cols = layout["cols"]
    tile_width = layout["tile_width"]
    tile_height = layout["tile_height"]
    pad = layout["pad"]
    label_band = layout["label_band"]
    border_width = max(8, int(round(max(tile_width, tile_height) * 0.018)))

    highlighted = contact_sheet.copy()
    draw = ImageDraw.Draw(highlighted)
    color = (255, 215, 0)
    for selected_frame_index in valid_indices:
        row = (selected_frame_index - 1) // cols
        col = (selected_frame_index - 1) % cols
        left = pad + col * (tile_width + pad)
        top = pad + row * (tile_height + label_band + pad)
        right = left + tile_width
        bottom = top + label_band + tile_height
        for inset in range(border_width):
            draw.rectangle(
                (left - inset, top - inset, right + inset, bottom + inset),
                outline=color,
            )
    return highlighted


def save_door_contact_sheet(
    args: argparse.Namespace,
    contact_sheet: Image.Image,
    frames: List[Image.Image],
    sheet_kind: str,
    start_time: float,
    end_time: float,
    selected_frame_index: int,
) -> Path:
    output_dir = Path(args.output_dir) / "door_status_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = int(getattr(args, "_door_contact_sheet_counter", 0)) + 1
    setattr(args, "_door_contact_sheet_counter", counter)
    selected_suffix = (
        f"frame{selected_frame_index:02d}"
        if selected_frame_index > 0
        else "frame00"
    )
    filename = (
        f"{counter:04d}_{sheet_kind}_"
        f"{start_time:09.2f}s_{end_time:09.2f}s_{selected_suffix}.jpg"
    )
    path = output_dir / filename
    sheet_to_save = highlight_contact_sheet_frame(
        contact_sheet=contact_sheet,
        frames=frames,
        selected_frame_index=selected_frame_index,
    )
    try:
        sheet_to_save.save(path, format="JPEG", quality=92)
    finally:
        if sheet_to_save is not contact_sheet:
            sheet_to_save.close()
    return path


def read_frame_at_time(
    cap: Any,
    timestamp: float,
    max_image_size: int,
    roi_bbox: Optional[Dict[str, float]] = None,
    roi_margin: float = 0.0,
    roi_mode: str = "crop",
    roi_label: str = "front door",
    lighting_preprocess: bool = False,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = max(0, int(round(timestamp * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None, f"Could not decode frame at {timestamp:.2f}s."

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame_rgb)
    if lighting_preprocess:
        frame = preprocess_bus_lighting(frame)
    if roi_bbox is not None:
        if roi_mode == "masked_overlay":
            frame = mask_outside_bbox_with_checkerboard(frame, roi_bbox)
            frame = draw_labeled_bbox(frame, roi_bbox, roi_label)
        elif roi_mode == "overlay":
            frame = draw_labeled_bbox(frame, roi_bbox, roi_label)
        else:
            frame = crop_frame_to_normalized_bbox(frame, roi_bbox, roi_margin)
    return resize_frame_if_needed(frame, max_image_size), None


def sample_time_window_frames(
    cap: Any,
    start_time: float,
    duration: float,
    num_frames: int,
    max_image_size: int,
    roi_bbox: Optional[Dict[str, float]] = None,
    roi_margin: float = 0.0,
    roi_mode: str = "crop",
    roi_label: str = "front door",
    lighting_preprocess: bool = False,
) -> Tuple[List[Image.Image], List[float], Optional[str]]:
    if duration <= 0 or num_frames <= 0:
        return [], [], "Invalid time window sampling request."

    if num_frames == 1:
        offsets = [duration / 2.0]
    else:
        offsets = [
            i * duration / (num_frames - 1)
            for i in range(num_frames)
        ]

    frames: List[Image.Image] = []
    frame_offsets: List[float] = []
    for offset in offsets:
        frame, frame_error = read_frame_at_time(
            cap,
            start_time + offset,
            max_image_size,
            roi_bbox=roi_bbox,
            roi_margin=roi_margin,
            roi_mode=roi_mode,
            roi_label=roi_label,
            lighting_preprocess=lighting_preprocess,
        )
        if frame_error is None and frame is not None:
            frames.append(frame)
            frame_offsets.append(offset)

    if not frames:
        return [], [], f"No frames decoded in window starting at {start_time:.2f}s."
    return frames, frame_offsets, None


def sample_all_time_window_frames(
    cap: Any,
    start_time: float,
    end_time: float,
    max_image_size: int,
    lighting_preprocess: bool = False,
) -> Tuple[List[Image.Image], Optional[str]]:
    if end_time <= start_time:
        return [], "Invalid time window sampling request."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = max(0, int(round(start_time * fps)))
    end_idx = max(start_idx + 1, int(round(end_time * fps)))
    if total_frames > 0:
        end_idx = min(end_idx, total_frames)
    if end_idx <= start_idx:
        return [], f"No frames in window {start_time:.2f}s-{end_time:.2f}s."

    frames: List[Image.Image] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frame_idx = start_idx
    while frame_idx < end_idx:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame_rgb)
        if lighting_preprocess:
            frame = preprocess_bus_lighting(frame)
        frames.append(resize_frame_if_needed(frame, max_image_size))
        frame_idx += 1

    if not frames:
        return [], f"No frames decoded in window {start_time:.2f}s-{end_time:.2f}s."
    return frames, None


def sample_time_window_frames_by_stride(
    cap: Any,
    start_time: float,
    end_time: float,
    frame_stride: int,
    max_image_size: int,
    lighting_preprocess: bool = False,
) -> Tuple[List[Image.Image], List[float], Optional[str]]:
    if end_time <= start_time or frame_stride <= 0:
        return [], [], "Invalid time window stride sampling request."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = max(0, int(round(start_time * fps)))
    end_idx = max(start_idx + 1, int(round(end_time * fps)))
    if total_frames > 0:
        end_idx = min(end_idx, total_frames)
    if end_idx <= start_idx:
        return [], [], f"No frames in window {start_time:.2f}s-{end_time:.2f}s."

    frames: List[Image.Image] = []
    frame_offsets: List[float] = []
    for frame_idx in range(start_idx, end_idx, frame_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame_rgb)
        if lighting_preprocess:
            frame = preprocess_bus_lighting(frame)
        frames.append(resize_frame_if_needed(frame, max_image_size))
        frame_offsets.append(max(0.0, frame_idx / fps - start_time))

    if not frames:
        return [], [], f"No frames decoded in window {start_time:.2f}s-{end_time:.2f}s."
    return frames, frame_offsets, None


def save_video_clip(
    cap: Any,
    start_time: float,
    end_time: float,
    output_path: Path,
    lighting_preprocess: bool = False,
) -> Optional[str]:
    if end_time <= start_time:
        return "Invalid clip time range."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = max(0, int(round(start_time * fps)))
    end_idx = max(start_idx + 1, int(round(end_time * fps)))
    if total_frames > 0:
        end_idx = min(end_idx, total_frames)
    if end_idx <= start_idx:
        return f"No frames in clip {start_time:.2f}s-{end_time:.2f}s."

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        return f"Could not decode first frame for clip at {start_time:.2f}s."

    def prepare_frame(frame_bgr: Any) -> Any:
        if not lighting_preprocess:
            return frame_bgr
        if Image is None or np is None:
            raise RuntimeError(
                "Lighting preprocessing requires Pillow and NumPy."
            )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        processed = preprocess_bus_lighting(Image.fromarray(frame_rgb))
        return cv2.cvtColor(np.array(processed.convert("RGB")), cv2.COLOR_RGB2BGR)

    try:
        first_frame = prepare_frame(first_frame)
    except Exception as exc:
        return f"Could not preprocess first frame for clip at {start_time:.2f}s: {exc}"

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return f"Could not create clip writer: {output_path}"

    frames_written = 0
    complete = True
    try:
        writer.write(first_frame)
        frames_written += 1
        frame_idx = start_idx + 1
        while frame_idx < end_idx:
            ok, frame = cap.read()
            if not ok or frame is None:
                complete = False
                break
            frame = prepare_frame(frame)
            writer.write(frame)
            frames_written += 1
            frame_idx += 1
    except Exception as exc:
        complete = False
        return_error = f"Could not write clip {start_time:.2f}s-{end_time:.2f}s: {exc}"
    finally:
        writer.release()
    if not complete:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return locals().get(
            "return_error",
            f"Could not decode all frames for clip {start_time:.2f}s-{end_time:.2f}s.",
        )
    if frames_written <= 0:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return f"No frames written for clip {start_time:.2f}s-{end_time:.2f}s."
    try:
        if not temp_path.exists() or temp_path.stat().st_size <= 0:
            temp_path.unlink(missing_ok=True)
            return f"Clip writer produced an empty file: {output_path}"
        validation_cap = cv2.VideoCapture(str(temp_path))
        try:
            valid_clip = (
                validation_cap.isOpened()
                and int(validation_cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            )
        finally:
            validation_cap.release()
        if not valid_clip:
            temp_path.unlink(missing_ok=True)
            return f"Clip writer produced an invalid video file: {output_path}"
        temp_path.replace(output_path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return f"Could not finalize clip {output_path}: {exc}"

    return None


def validate_video_file(path: Path) -> Optional[str]:
    if not path.exists() or path.stat().st_size <= 0:
        return "Video file is empty or missing."
    validation_cap = cv2.VideoCapture(str(path))
    try:
        if not validation_cap.isOpened():
            return "Video file cannot be opened."
        if int(validation_cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
            return "Video file has no frames."
    finally:
        validation_cap.release()
    return None


def build_messages(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": content}]


def content_contains_video(content: List[Dict[str, Any]]) -> bool:
    return any(item.get("type") == "video" for item in content)


def prepare_inputs(processor: Any, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    has_video = any(
        item.get("type") == "video"
        for message in messages
        for item in message.get("content", [])
    )

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        if has_video:
            raise ImportError(
                "Video VLM input requires qwen-vl-utils with process_vision_info."
            )
        image_inputs = [
            item["image"]
            for message in messages
            for item in message.get("content", [])
            if item.get("type") == "image"
        ]
        video_inputs = None

    processor_kwargs = {
        "text": [text],
        "images": image_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    return processor(**processor_kwargs)


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_pad_token_id(processor_or_tokenizer: Any, model: Any) -> Optional[int]:
    for owner in (processor_or_tokenizer, getattr(processor_or_tokenizer, "tokenizer", None)):
        if owner is None:
            continue
        pad_token_id = getattr(owner, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(owner, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is not None:
            if isinstance(eos_token_id, list):
                return int(eos_token_id[0])
            return int(eos_token_id)
    return None


def generate_response_from_content(
    model: Any,
    processor: Any,
    content: List[Dict[str, Any]],
    max_new_tokens: int,
) -> str:
    messages = None
    inputs = None
    generated_ids = None
    generated_ids_trimmed = None

    try:
        messages = build_messages(content)
        inputs = prepare_inputs(processor, messages)

        inputs = inputs.to(model_device(model))

        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            pad_token_id = resolve_pad_token_id(processor, model)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            generated_ids = model.generate(
                **inputs,
                **generate_kwargs,
            )

        input_token_count = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_token_count:]
        decoded = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
    finally:
        del generated_ids_trimmed
        del generated_ids
        del inputs
        del messages
        cleanup_memory()


def build_internvl_transform(input_size: int) -> Any:
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = INTERNVL_INPUT_SIZE,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    sorted_ratios = sorted(target_ratios, key=lambda item: item[0] * item[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        sorted_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    grid_width = target_width // image_size
    for i in range(blocks):
        box = (
            (i % grid_width) * image_size,
            (i // grid_width) * image_size,
            ((i % grid_width) + 1) * image_size,
            ((i // grid_width) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def internvl_image_to_pixel_values(
    image: Image.Image,
    input_size: int,
    max_tiles: int,
) -> "torch.Tensor":
    transform = build_internvl_transform(input_size)
    tiles = dynamic_preprocess_internvl(
        image.convert("RGB"),
        image_size=input_size,
        max_num=max_tiles,
        use_thumbnail=True,
    )
    return torch.stack([transform(tile) for tile in tiles])


def content_to_internvl_question_and_images(
    content: List[Dict[str, Any]]
) -> Tuple[str, List[Image.Image]]:
    parts: List[str] = []
    images: List[Image.Image] = []
    for item in content:
        if item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text:
                parts.append(text)
        elif item.get("type") == "image":
            images.append(item["image"])
            parts.append(f"Frame{len(images)}: <image>")
    return "\n".join(parts), images


def generate_internvl_response_from_content(
    model: Any,
    tokenizer: Any,
    content: List[Dict[str, Any]],
    max_new_tokens: int,
) -> str:
    pixel_values = None
    try:
        question, images = content_to_internvl_question_and_images(content)
        if images:
            max_tiles = (
                INTERNVL_SINGLE_IMAGE_MAX_TILES
                if len(images) == 1
                else INTERNVL_MULTI_IMAGE_MAX_TILES
            )
            pixel_values_list = [
                internvl_image_to_pixel_values(
                    image,
                    input_size=INTERNVL_INPUT_SIZE,
                    max_tiles=max_tiles,
                )
                for image in images
            ]
            num_patches_list = [values.shape[0] for values in pixel_values_list]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            pixel_values = pixel_values.to(dtype=get_torch_dtype(), device=model_device(model))
        else:
            num_patches_list = None

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = resolve_pad_token_id(tokenizer, model)
        if pad_token_id is not None:
            generation_config["pad_token_id"] = pad_token_id
        chat_kwargs = {
            "tokenizer": tokenizer,
            "pixel_values": pixel_values,
            "question": question,
            "generation_config": generation_config,
            "history": None,
            "return_history": False,
        }
        if num_patches_list is not None and len(num_patches_list) > 1:
            chat_kwargs["num_patches_list"] = num_patches_list
        response = model.chat(**chat_kwargs)
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()
    finally:
        del pixel_values
        cleanup_memory()


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def content_to_openai_content(
    content: List[Dict[str, Any]],
    detail: str,
) -> List[Dict[str, Any]]:
    openai_content: List[Dict[str, Any]] = []
    for item in content:
        if item.get("type") == "text":
            openai_content.append({"type": "input_text", "text": item.get("text", "")})
        elif item.get("type") == "image":
            openai_content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(item["image"]),
                    "detail": detail,
                }
            )
    return openai_content


def generate_openai_json_response_from_content(
    client: Any,
    model: str,
    content: List[Dict[str, Any]],
    max_new_tokens: int,
    detail: str,
    schema_name: str,
    schema: Dict[str, Any],
) -> str:
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content_to_openai_content(content, detail)}],
        max_output_tokens=max_new_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    return response.output_text.strip()


def door_change_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status_changed": {"type": "boolean"},
            "transition": {"type": "string"},
            "change_frame_index": {"type": "integer"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "status_changed",
            "transition",
            "change_frame_index",
            "confidence",
            "reasoning",
        ],
    }


def door_bbox_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "found": {"type": "boolean"},
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "found",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "reasoning",
        ],
    }


def build_door_change_content(
    contact_sheet: Image.Image,
    frame_count: int,
    clip_start_time: float,
    clip_duration: float,
    door_annotation_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sheet_rows, sheet_cols = contact_sheet_grid(frame_count)
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Query clip starts at absolute video time {clip_start_time:.2f}s "
                f"and lasts {clip_duration:.2f}s. The following single image is "
                f"a contact sheet containing {frame_count} sampled frames in "
                f"chronological order, laid out as {sheet_rows} rows x "
                f"{sheet_cols} columns. Read the frame numbers shown in each tile."
            ),
        }
    ]
    if door_annotation_label:
        content.append(
            {
                "type": "text",
                "text": (
                    "Each tile keeps the original pixels inside the red bounding "
                    f"box labeled \"{door_annotation_label}\". Everything outside "
                    "the box is intentionally masked with a gray and white "
                    "checkerboard. Judge only the door inside the labeled box."
                ),
            }
        )
    content.append({"type": "image", "image": contact_sheet})
    content.append({"type": "text", "text": DOOR_CHANGE_INSTRUCTION})
    return content


def build_door_bbox_content(frame: Image.Image, frame_time: float) -> List[Dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": (
                "This is the first frame used for this streaming analysis, "
                f"at absolute video time {frame_time:.2f}s. Locate the bus "
                "front passenger entrance door in this image."
            ),
        },
        {"type": "image", "image": frame},
        {"type": "text", "text": DOOR_BBOX_INSTRUCTION},
    ]


def build_door_change_verification_content(
    contact_sheet: Image.Image,
    frame_count: int,
    clip_start_time: float,
    clip_duration: float,
    candidate_transition: str,
    candidate_change_time: float,
    door_annotation_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    candidate_offset = candidate_change_time - clip_start_time
    sheet_rows, sheet_cols = contact_sheet_grid(frame_count)
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Verification clip starts at absolute video time {clip_start_time:.2f}s "
                f"and lasts {clip_duration:.2f}s. The following single image is "
                f"a contact sheet containing {frame_count} sampled frames in "
                f"chronological order, laid out as {sheet_rows} rows x "
                f"{sheet_cols} columns. Read the frame numbers shown in each tile."
            ),
        },
        {
            "type": "text",
            "text": (
                "First-round proposed answer to verify: "
                f"transition={candidate_transition}, "
                f"absolute_change_time={candidate_change_time:.2f}s, "
                f"offset_in_this_verification_clip={candidate_offset:.2f}s."
            ),
        },
    ]
    if door_annotation_label:
        content.append(
            {
                "type": "text",
                "text": (
                    "Each tile keeps the original pixels inside the red bounding "
                    f"box labeled \"{door_annotation_label}\". Everything outside "
                    "the box is intentionally masked with a gray and white "
                    "checkerboard. Verify only the door inside the labeled box."
                ),
            }
        )
    content.append({"type": "image", "image": contact_sheet})
    content.append({"type": "text", "text": DOOR_CHANGE_VERIFICATION_INSTRUCTION})
    return content


def ask_json_probe(
    *,
    backend: str,
    model_family: str,
    model_name: str,
    model: Any,
    processor: Any,
    openai_client: Any,
    content: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
    max_new_tokens: int,
    detail: str,
) -> Dict[str, Any]:
    if content_contains_video(content) and (
        backend != "local" or model_family == "internvl"
    ):
        raise RuntimeError(
            "Video clip VLM input is only supported by the local Qwen/Qwen3 "
            "backend in this script."
        )

    if backend == "openai":
        raw_response = generate_openai_json_response_from_content(
            client=openai_client,
            model=model_name,
            content=content,
            max_new_tokens=max_new_tokens,
            detail=detail,
            schema_name=schema_name,
            schema=schema,
        )
    elif model_family == "internvl":
        raw_response = generate_internvl_response_from_content(
            model=model,
            tokenizer=processor,
            content=content,
            max_new_tokens=max_new_tokens,
        )
    else:
        raw_response = generate_response_from_content(
            model=model,
            processor=processor,
            content=content,
            max_new_tokens=max_new_tokens,
        )

    parsed = extract_json_object(raw_response)
    if parsed is None:
        return {
            "raw_response": raw_response,
            "confidence": 0.0,
            "reasoning": "Could not parse valid JSON.",
        }
    parsed["raw_response"] = raw_response
    return parsed


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start() :])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def read_json_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_bbox_json_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"detections": payload}
    return {}


def coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def bool_from_probe(parsed: Dict[str, Any], key: str) -> bool:
    value = parsed.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "open"}
    return False


def write_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(csv_path, index=False)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def tensor_to_numpy(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return np.asarray(value) if np is not None else value


def repair_main_passenger_ids(
    raw_main_ids: List[int],
    max_short_id_run: int = 3,
) -> Tuple[List[int], Dict[str, Any]]:
    first_pass_ids = list(raw_main_ids)
    first_pass_repairs: List[Dict[str, Any]] = []
    frame_count = len(first_pass_ids)

    for index in range(frame_count):
        previous_id = first_pass_ids[index - 1] if index > 0 else -1
        current_id = first_pass_ids[index]
        next_id = first_pass_ids[index + 1] if index + 1 < frame_count else -1
        if current_id != previous_id and current_id != next_id:
            first_pass_ids[index] = previous_id
            first_pass_repairs.append(
                {
                    "frame_index": index,
                    "from_passenger_id": current_id,
                    "to_passenger_id": previous_id,
                    "reason": "single_frame_unconfirmed_id",
                }
            )

    repaired_main_ids = list(first_pass_ids)
    second_pass_repairs: List[Dict[str, Any]] = []
    index = 0
    while index < frame_count:
        passenger_id = repaired_main_ids[index]
        run_start = index
        while index + 1 < frame_count and repaired_main_ids[index + 1] == passenger_id:
            index += 1
        run_end = index
        run_length = run_end - run_start + 1

        if run_length <= max_short_id_run and passenger_id >= 0 and run_start > 0:
            replacement_id = repaired_main_ids[run_start - 1]
            if replacement_id != passenger_id:
                for repair_index in range(run_start, run_end + 1):
                    repaired_main_ids[repair_index] = replacement_id
                second_pass_repairs.append(
                    {
                        "start_frame_index": run_start,
                        "end_frame_index": run_end,
                        "frame_count": run_length,
                        "from_passenger_id": passenger_id,
                        "to_passenger_id": replacement_id,
                        "reason": "short_id_run_replaced_by_previous_id",
                    }
                )
        elif (
            run_length <= max_short_id_run
            and passenger_id < 0
            and run_start > 0
            and run_end + 1 < frame_count
        ):
            previous_id = repaired_main_ids[run_start - 1]
            next_id = repaired_main_ids[run_end + 1]
            if previous_id >= 0 and previous_id == next_id:
                for repair_index in range(run_start, run_end + 1):
                    repaired_main_ids[repair_index] = previous_id
                second_pass_repairs.append(
                    {
                        "start_frame_index": run_start,
                        "end_frame_index": run_end,
                        "frame_count": run_length,
                        "from_passenger_id": passenger_id,
                        "to_passenger_id": previous_id,
                        "reason": "short_missing_run_filled_between_same_ids",
                    }
                )

        index += 1

    return repaired_main_ids, {
        "method": "two_pass_previous_frame_overwrite",
        "first_pass": {
            "rule": (
                "If a frame's ID differs from both the previous frame and the "
                "next frame, replace it with the previous frame ID."
            ),
            "repairs": first_pass_repairs,
        },
        "second_pass": {
            "max_short_id_run": max_short_id_run,
            "rule": (
                "After first-pass repair, replace any non-negative passenger ID "
                "run with length <= max_short_id_run by the ID immediately before "
                "that run. Also fill any -1 run with length <= max_short_id_run "
                "only when the IDs immediately before and after the run are the "
                "same valid passenger ID."
            ),
            "repairs": second_pass_repairs,
        },
    }


def derive_largest_box_passenger_changes(
    frame_detections: List[Dict[str, Any]],
    frame_offsets: List[float],
    stop_duration: float,
) -> Tuple[
    List[int],
    List[int],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    frame_count = len(frame_offsets)
    raw_main_ids = [-1] * frame_count
    main_box_areas = [0.0] * frame_count

    for frame_index, frame_detection in enumerate(frame_detections[:frame_count]):
        instances = frame_detection.get("instances", [])
        if not isinstance(instances, list):
            continue
        valid_instances = [
            item
            for item in instances
            if isinstance(item, dict)
            and coerce_int(item.get("object_id"), -1) >= 0
            and coerce_float(item.get("box_area"), 0.0) > 0.0
        ]
        if not valid_instances:
            continue
        largest = max(
            valid_instances,
            key=lambda item: coerce_float(item.get("box_area"), 0.0),
        )
        raw_main_ids[frame_index] = coerce_int(largest.get("object_id"), -1)
        main_box_areas[frame_index] = coerce_float(largest.get("box_area"), 0.0)

    repaired_main_ids, repair_summary = repair_main_passenger_ids(raw_main_ids)

    transitions: List[Dict[str, Any]] = []
    split_points: List[Dict[str, Any]] = []
    previous_id = -1
    for index, passenger_id in enumerate(repaired_main_ids):
        if passenger_id < 0:
            continue
        if previous_id >= 0 and passenger_id != previous_id:
            split_frame_index = index + 1
            offset = frame_index_to_offset(split_frame_index, frame_offsets)
            if 0.0 < offset < stop_duration:
                transition = {
                    "split_frame_index": split_frame_index,
                    "from_passenger_id": previous_id,
                    "to_passenger_id": passenger_id,
                    "new_main_box_area": main_box_areas[index],
                }
                transitions.append(transition)
                split_points.append(
                    {
                        "time_offset": offset,
                        "split_frame_index": split_frame_index,
                        "confidence": 1.0,
                        "reasoning": (
                            f"Largest SAM3 box changes from passenger "
                            f"{previous_id} to passenger {passenger_id}."
                        ),
                    }
                )
        previous_id = passenger_id

    return raw_main_ids, repaired_main_ids, transitions, split_points, repair_summary


def build_payment_clips_from_main_id_runs(
    *,
    stop_start: float,
    stop_end: float,
    main_passenger_ids: List[int],
    frame_offsets: List[float],
    min_duration: float,
    frame_merge_distance: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stop_duration = max(0.0, stop_end - stop_start)
    frame_count = min(len(main_passenger_ids), len(frame_offsets))
    clips: List[Dict[str, Any]] = []
    raw_run_segments: List[Dict[str, Any]] = []

    index = 0
    run_index = 1
    while index < frame_count:
        passenger_id = coerce_int(main_passenger_ids[index], -1)
        if passenger_id < 0:
            index += 1
            continue

        start_index = index
        while (
            index + 1 < frame_count
            and coerce_int(main_passenger_ids[index + 1], -1) == passenger_id
        ):
            index += 1
        end_index = index

        start_offset = max(0.0, min(stop_duration, frame_offsets[start_index]))
        if end_index + 1 < frame_count:
            end_offset = frame_offsets[end_index + 1]
        else:
            end_offset = stop_duration
        end_offset = max(start_offset, min(stop_duration, end_offset))
        duration = end_offset - start_offset

        segment = {
            "run_index": run_index,
            "main_passenger_id": passenger_id,
            "main_passenger_ids": [passenger_id],
            "merged_run_indices": [run_index],
            "start_frame_index": start_index + 1,
            "end_frame_index": end_index + 1,
            "start_time_offset": start_offset,
            "end_time_offset": end_offset,
            "duration": duration,
            "merged_by_close_frame_pins": False,
        }
        raw_run_segments.append(segment)

        run_index += 1
        index += 1

    run_segments: List[Dict[str, Any]] = []
    for segment in raw_run_segments:
        if run_segments and frame_merge_distance >= 0:
            previous = run_segments[-1]
            start_distance = (
                coerce_int(segment.get("start_frame_index"), 0)
                - coerce_int(previous.get("start_frame_index"), 0)
            )
            if 0 <= start_distance <= frame_merge_distance:
                previous["end_frame_index"] = max(
                    coerce_int(previous.get("end_frame_index"), 0),
                    coerce_int(segment.get("end_frame_index"), 0),
                )
                previous["end_time_offset"] = max(
                    coerce_float(previous.get("end_time_offset"), 0.0),
                    coerce_float(segment.get("end_time_offset"), 0.0),
                )
                previous["duration"] = (
                    coerce_float(previous.get("end_time_offset"), 0.0)
                    - coerce_float(previous.get("start_time_offset"), 0.0)
                )
                previous["merged_by_close_frame_pins"] = True
                previous["merge_frame_distance"] = frame_merge_distance
                previous["merged_run_indices"] = (
                    list(previous.get("merged_run_indices", []))
                    + list(segment.get("merged_run_indices", []))
                )
                previous["main_passenger_ids"] = sorted(
                    set(previous.get("main_passenger_ids", []))
                    | set(segment.get("main_passenger_ids", []))
                )
                previous.setdefault("merged_segments", []).append(segment)
                continue
        run_segments.append(dict(segment))

    passenger_index = 1
    for segment in run_segments:
        duration = coerce_float(segment.get("duration"), 0.0)
        kept = duration >= min_duration
        segment["passenger_index"] = passenger_index if kept else None
        segment["kept"] = kept
        if kept:
            clips.append(
                {
                    "start_time": stop_start
                    + coerce_float(segment.get("start_time_offset"), 0.0),
                    "end_time": stop_start
                    + coerce_float(segment.get("end_time_offset"), 0.0),
                    "passenger_index": passenger_index,
                    "main_passenger_id": coerce_int(
                        segment.get("main_passenger_id"),
                        -1,
                    ),
                    "main_passenger_ids": list(
                        segment.get("main_passenger_ids", [])
                    ),
                    "start_frame_index": coerce_int(
                        segment.get("start_frame_index"),
                        0,
                    ),
                    "end_frame_index": coerce_int(
                        segment.get("end_frame_index"),
                        0,
                    ),
                    "merged_by_close_frame_pins": bool(
                        segment.get("merged_by_close_frame_pins")
                    ),
                    "merged_run_indices": list(
                        segment.get("merged_run_indices", [])
                    ),
                }
            )
            passenger_index += 1

    split_points: List[Dict[str, Any]] = []
    for segment in run_segments[1:]:
        split_points.append(
            {
                "time_offset": segment["start_time_offset"],
                "split_frame_index": segment["start_frame_index"],
                "confidence": 1.0,
                "reasoning": (
                    "New repaired main passenger ID run starts with passenger "
                    f"{segment['main_passenger_id']}."
                ),
            }
        )

    return clips, run_segments, split_points


def evenly_spaced_indices(start_index: int, end_index: int, count: int) -> List[int]:
    if count <= 0 or end_index < start_index:
        return []
    if count == 1 or start_index == end_index:
        return [start_index] * count
    span = end_index - start_index
    return [
        int(round(start_index + (span * offset / max(1, count - 1))))
        for offset in range(count)
    ]


def find_passenger_instance_for_clip(
    frame_detection: Dict[str, Any],
    clip: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    instances = frame_detection.get("instances", [])
    if not isinstance(instances, list):
        return None

    main_id = coerce_int(clip.get("main_passenger_id"), -1)
    target_ids: List[int] = []
    if main_id >= 0:
        target_ids.append(main_id)
    for passenger_id in clip.get("main_passenger_ids", []):
        passenger_id = coerce_int(passenger_id, -1)
        if passenger_id >= 0 and passenger_id not in target_ids:
            target_ids.append(passenger_id)

    for target_id in target_ids:
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            if coerce_int(instance.get("object_id"), -1) == target_id:
                return instance
    return None


def save_passenger_clip_annotation_sheet(
    *,
    frames: List[Image.Image],
    frame_detections: List[Dict[str, Any]],
    clip: Dict[str, Any],
    sheet_path: Path,
) -> Optional[str]:
    if not frames:
        return "No sampled frames are available for this passenger clip."

    start_index = coerce_int(clip.get("start_frame_index"), 0) - 1
    end_index = coerce_int(clip.get("end_frame_index"), 0) - 1
    start_index = max(0, min(len(frames) - 1, start_index))
    end_index = max(0, min(len(frames) - 1, end_index))
    if end_index < start_index:
        start_index, end_index = end_index, start_index

    selected_indices = evenly_spaced_indices(start_index, end_index, 8)
    if not selected_indices:
        return "No sampled frame indices are available for this passenger clip."

    annotated_frames: List[Image.Image] = []
    labels: List[str] = []
    for frame_index in selected_indices:
        frame = frames[frame_index]
        frame_detection = (
            frame_detections[frame_index]
            if frame_index < len(frame_detections)
            and isinstance(frame_detections[frame_index], dict)
            else {}
        )
        instance = find_passenger_instance_for_clip(frame_detection, clip)
        if instance is not None and isinstance(instance.get("bbox"), dict):
            bbox = instance["bbox"]
            object_id = coerce_int(
                instance.get("object_id"),
                coerce_int(clip.get("main_passenger_id"), -1),
            )
            masked = mask_outside_bbox_with_checkerboard(frame, bbox)
            annotated = draw_passenger_bbox_outline(
                masked,
                {**bbox, "object_id": object_id},
                object_id,
            )
            if masked is not frame:
                masked.close()
        else:
            annotated = frame.copy()
        annotated_frames.append(annotated)
        labels.append(f"Frame {frame_index + 1}")

    clip["passenger_sheet_format"] = "2x4_checkerboard_outline_box_contact_sheet"
    clip["passenger_sheet_frame_indices"] = [
        index + 1 for index in selected_indices
    ]

    try:
        sheet = build_fixed_frame_contact_sheet(
            annotated_frames,
            rows=2,
            cols=4,
            labels=labels,
        )
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path, format="JPEG", quality=92)
        sheet.close()
    finally:
        close_frames(annotated_frames)

    return None


def normalize_transition(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "closed_to_open": "closed_to_open",
        "close_to_open": "closed_to_open",
        "closed2open": "closed_to_open",
        "open": "closed_to_open",
        "open_to_closed": "open_to_closed",
        "open_to_close": "open_to_closed",
        "open2closed": "open_to_closed",
        "closed": "open_to_closed",
        "close": "open_to_closed",
        "none": "none",
        "no": "none",
        "no_change": "none",
        "unchanged": "none",
    }
    return aliases.get(text, "none")


def normalize_change_frame_index(
    parsed: Dict[str, Any],
    frame_count: int,
    status_changed: bool,
) -> int:
    if frame_count <= 0 or not status_changed:
        return 0
    value = parsed.get("change_frame_index")
    index = coerce_int(value, 0)
    if index <= 0:
        return 0
    return max(1, min(frame_count, index))


def frame_index_to_offset(change_frame_index: int, frame_offsets: List[float]) -> float:
    if change_frame_index <= 0 or not frame_offsets:
        return 0.0
    index = max(1, min(len(frame_offsets), change_frame_index))
    return frame_offsets[index - 1]


def parse_bbox_values(parsed: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if all(key in parsed for key in ("x1", "y1", "x2", "y2")):
        return (
            coerce_float(parsed.get("x1")),
            coerce_float(parsed.get("y1")),
            coerce_float(parsed.get("x2")),
            coerce_float(parsed.get("y2")),
        )

    for key in ("bbox_2d", "bbox", "box"):
        value = parsed.get(key)
        if isinstance(value, list) and len(value) >= 4:
            return tuple(coerce_float(item) for item in value[:4])  # type: ignore[return-value]

    return None


def select_bbox_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    candidates = parsed.get("detections")
    if not isinstance(candidates, list):
        candidates = parsed.get("objects")
    if not isinstance(candidates, list):
        candidates = parsed.get("instances")
    if not isinstance(candidates, list):
        return parsed

    valid_candidates = [item for item in candidates if isinstance(item, dict)]
    if not valid_candidates:
        return parsed

    def candidate_score(item: Dict[str, Any]) -> Tuple[float, float]:
        confidence = coerce_confidence(
            item.get("confidence", item.get("score", item.get("iou", 0.0)))
        )
        values = parse_bbox_values(item)
        if values is None:
            return confidence, 0.0
        x1, y1, x2, y2 = values
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return confidence, area

    best = max(valid_candidates, key=candidate_score)
    merged = dict(parsed)
    merged.update(best)
    if "found" not in merged:
        merged["found"] = True
    return merged


def normalize_door_bbox(
    parsed: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, float]]:
    parsed = select_bbox_payload(parsed)

    values = parse_bbox_values(parsed)
    if values is None:
        return None
    if "found" in parsed and not bool_from_probe(parsed, "found"):
        return None

    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.0:
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height
    elif (x2 > image_width or y2 > image_height) and max_coord <= 1000.0:
        x1 = x1 / 1000.0 * image_width
        x2 = x2 / 1000.0 * image_width
        y1 = y1 / 1000.0 * image_height
        y2 = y2 / 1000.0 * image_height

    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x1_norm": x1 / image_width,
        "y1_norm": y1 / image_height,
        "x2_norm": x2 / image_width,
        "y2_norm": y2 / image_height,
    }


def save_door_bbox_debug(
    *,
    output_dir: Path,
    frame: Image.Image,
    frame_time: float,
    parsed: Dict[str, Any],
    bbox: Optional[Dict[str, float]],
    roi_margin: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_payload = {
        "frame_time": frame_time,
        "image_width": frame.size[0],
        "image_height": frame.size[1],
        "found": bbox is not None,
        "bbox": bbox,
        "roi_margin": roi_margin,
        "confidence": coerce_confidence(parsed.get("confidence", parsed.get("score"))),
        "reasoning": str(parsed.get("reasoning", "")),
        "raw_response": str(parsed.get("raw_response", "")),
    }
    json_path = output_dir / "door_roi_bbox.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(debug_payload, f, indent=2, ensure_ascii=False)

    if bbox is not None:
        cleanup_stale_door_debug_images(output_dir)
        overlay = draw_labeled_bbox(frame, bbox, "front door")
        overlay.save(output_dir / "door_roi_overlay.jpg", format="JPEG", quality=92)


def cleanup_stale_door_debug_images(output_dir: Path) -> None:
    for filename in (
        "door_roi_source.jpg",
        "door_roi_sam3d_overlay.jpg",
        "door_roi_bbox.jpg",
        "door_roi_crop.jpg",
    ):
        path = output_dir / filename
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            print(f"Could not remove stale debug image {path}: {exc}")


def cleanup_stale_passenger_clips(output_dir: Path) -> None:
    stale_outputs = (
        (output_dir / "passenger_clips", "*.mp4", "passenger clip"),
        (output_dir / "passenger_clip_sheets", "*.jpg", "passenger clip sheet"),
    )
    for output_path, pattern, label in stale_outputs:
        if not output_path.exists():
            continue
        for path in output_path.glob(pattern):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Could not remove stale {label} {path}: {exc}")


def load_cached_door_roi(
    *,
    output_dir: Path,
    flow_events: List[Dict[str, Any]],
    cache_path: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    cache_path = cache_path or (output_dir / "door_roi_bbox.json")
    if not cache_path.exists():
        return None

    try:
        payload = coerce_bbox_json_payload(read_json_payload(cache_path))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read cached door ROI {cache_path}: {exc}")
        return None

    bbox = payload.get("bbox")
    if isinstance(bbox, dict):
        required = ("x1_norm", "y1_norm", "x2_norm", "y2_norm")
        if all(key in bbox for key in required):
            x1 = coerce_float(bbox.get("x1_norm"), 0.0)
            y1 = coerce_float(bbox.get("y1_norm"), 0.0)
            x2 = coerce_float(bbox.get("x2_norm"), 0.0)
            y2 = coerce_float(bbox.get("y2_norm"), 0.0)
            if 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0:
                cached_bbox = {
                    "x1": coerce_float(bbox.get("x1")),
                    "y1": coerce_float(bbox.get("y1")),
                    "x2": coerce_float(bbox.get("x2")),
                    "y2": coerce_float(bbox.get("y2")),
                    "x1_norm": x1,
                    "y1_norm": y1,
                    "x2_norm": x2,
                    "y2_norm": y2,
                }
                flow_events.append(
                    {
                        "type": "door_roi_detection",
                        "detector": "cache",
                        "found": True,
                        "bbox": cached_bbox,
                        "cache_path": str(cache_path),
                        "reasoning": "Reused cached door ROI bbox.",
                    }
                )
                print(f"Reusing cached door ROI from {cache_path}")
                return cached_bbox

    image_width = coerce_int(payload.get("image_width"), 0)
    image_height = coerce_int(payload.get("image_height"), 0)
    if image_width > 0 and image_height > 0:
        normalized = normalize_door_bbox(payload, image_width, image_height)
        if normalized is not None:
            flow_events.append(
                {
                    "type": "door_roi_detection",
                    "detector": "cache",
                    "found": True,
                    "bbox": normalized,
                    "cache_path": str(cache_path),
                    "reasoning": "Reused cached door ROI bbox.",
                }
            )
            print(f"Reusing cached door ROI from {cache_path}")
            return normalized

    print(f"Cached door ROI is invalid, rerunning detector: {cache_path}")
    return None


def build_sam3d_door_command(
    command_template: str,
    image_path: Path,
    output_path: Path,
    overlay_path: Path,
    prompt: str,
    prompt_variants: str,
    thresholds: str,
) -> List[str]:
    if not command_template.strip():
        raise ValueError(
            "SAM3D door ROI detection requires --sam3d_door_command or "
            "SAM3D_DOOR_COMMAND. Example: "
            "\"conda run -n sam3d python detect_door.py "
            "--image {image} --prompt {prompt} --output {output}\""
        )

    replacements = {
        "image": shlex.quote(str(image_path)),
        "output": shlex.quote(str(output_path)),
        "overlay": shlex.quote(str(overlay_path)),
        "prompt": shlex.quote(prompt),
        "prompt_variants": shlex.quote(prompt_variants),
        "thresholds": shlex.quote(thresholds),
    }
    if any("{" + key + "}" in command_template for key in replacements):
        command_text = command_template.format(**replacements)
        return shlex.split(command_text)

    command = shlex.split(command_template)
    command.extend(
        [
            "--image",
            str(image_path),
            "--prompt",
            prompt,
            "--prompt-variants",
            prompt_variants,
            "--thresholds",
            thresholds,
            "--output",
            str(output_path),
        ]
    )
    return command


def detect_front_door_roi_sam3d(
    *,
    cap: Any,
    frame_time: float,
    output_dir: Path,
    args: argparse.Namespace,
    flow_events: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    frame: Optional[Image.Image] = None
    raw_json_path = output_dir / "door_roi_sam3d_raw.json"
    try:
        frame, frame_error = read_frame_at_time(
            cap=cap,
            timestamp=frame_time,
            max_image_size=args.sam3d_door_image_size,
        )
        if frame_error is not None or frame is None:
            event = {
                "type": "door_roi_detection",
                "detector": "sam3d",
                "frame_time": frame_time,
                "found": False,
                "reasoning": frame_error or "Could not decode first frame.",
            }
            flow_events.append(event)
            if args.sam3d_allow_full_frame_fallback:
                print(f"SAM3D door ROI detection skipped: {event['reasoning']}")
                return None
            raise RuntimeError(str(event["reasoning"]))

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="door_roi_") as tmp_dir:
            image_path = Path(tmp_dir) / "door_roi_source.jpg"
            overlay_path = Path(tmp_dir) / "door_roi_sam3d_overlay.jpg"
            frame.save(image_path, format="JPEG", quality=92)
            command = build_sam3d_door_command(
                args.sam3d_door_command,
                image_path=image_path,
                output_path=raw_json_path,
                overlay_path=overlay_path,
                prompt=args.sam3d_door_prompt,
                prompt_variants=args.sam3d_door_prompt_variants,
                thresholds=args.sam3d_door_thresholds,
            )
            print(
                "Running SAM3D door ROI detection "
                f"prompt={args.sam3d_door_prompt!r} image={image_path}"
            )
            completed = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                text=True,
                capture_output=True,
                timeout=args.sam3d_door_timeout,
                check=False,
            )
        if completed.returncode != 0:
            reason = (
                f"SAM3D door command failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
            event = {
                "type": "door_roi_detection",
                "detector": "sam3d",
                "frame_time": frame_time,
                "found": False,
                "command": command,
                "reasoning": reason,
            }
            flow_events.append(event)
            if args.sam3d_allow_full_frame_fallback:
                print(f"{reason} Using full frames.")
                return None
            raise RuntimeError(reason)

        if raw_json_path.exists():
            payload = read_json_payload(raw_json_path)
        else:
            payload = extract_json_object(completed.stdout) or {}
            with raw_json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        parsed = coerce_bbox_json_payload(payload)
        bbox = normalize_door_bbox(
            parsed,
            image_width=frame.size[0],
            image_height=frame.size[1],
        )
        save_door_bbox_debug(
            output_dir=output_dir,
            frame=frame,
            frame_time=frame_time,
            parsed={
                **select_bbox_payload(parsed),
                "raw_response": json.dumps(payload, ensure_ascii=False),
            },
            bbox=bbox,
            roi_margin=args.door_roi_margin,
        )
        event = {
            "type": "door_roi_detection",
            "detector": "sam3d",
            "frame_time": frame_time,
            "image_width": frame.size[0],
            "image_height": frame.size[1],
            "found": bbox is not None,
            "bbox": bbox,
            "roi_margin": args.door_roi_margin,
            "prompt": args.sam3d_door_prompt,
            "prompt_variants": args.sam3d_door_prompt_variants,
            "thresholds": args.sam3d_door_thresholds,
            "command": command,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "raw_json": str(raw_json_path),
            "debug_json": str(output_dir / "door_roi_bbox.json"),
            "debug_image": str(output_dir / "door_roi_overlay.jpg") if bbox is not None else None,
        }
        flow_events.append(event)
        if bbox is None:
            reason = "SAM3D door command did not return a valid bbox."
            if args.sam3d_allow_full_frame_fallback:
                print(f"{reason} Using full frames.")
                return None
            raise RuntimeError(reason)

        print(
            "SAM3D door ROI detected "
            f"bbox=({bbox['x1']:.1f},{bbox['y1']:.1f},"
            f"{bbox['x2']:.1f},{bbox['y2']:.1f})"
        )
        return bbox
    finally:
        if frame is not None:
            close_frames([frame])
        cleanup_memory()


def detect_front_door_roi_vlm(
    *,
    cap: Any,
    frame_time: float,
    output_dir: Path,
    args: argparse.Namespace,
    backend: str,
    model_family: str,
    model_name: str,
    model: Any,
    processor: Any,
    openai_client: Any,
    flow_events: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    frame: Optional[Image.Image] = None
    try:
        frame, frame_error = read_frame_at_time(
            cap=cap,
            timestamp=frame_time,
            max_image_size=args.door_image_size,
        )
        if frame_error is not None or frame is None:
            event = {
                "type": "door_roi_detection",
                "detector": "vlm",
                "frame_time": frame_time,
                "found": False,
                "reasoning": frame_error or "Could not decode first frame.",
            }
            flow_events.append(event)
            print(f"Door ROI detection skipped: {event['reasoning']}")
            return None

        parsed = ask_json_probe(
            backend=backend,
            model_family=model_family,
            model_name=model_name,
            model=model,
            processor=processor,
            openai_client=openai_client,
            content=build_door_bbox_content(frame, frame_time),
            schema_name="front_door_bbox",
            schema=door_bbox_schema(),
            max_new_tokens=min(args.max_new_tokens, 160),
            detail=args.openai_detail,
        )
        bbox = normalize_door_bbox(
            parsed,
            image_width=frame.size[0],
            image_height=frame.size[1],
        )
        save_door_bbox_debug(
            output_dir=output_dir,
            frame=frame,
            frame_time=frame_time,
            parsed=parsed,
            bbox=bbox,
            roi_margin=args.door_roi_margin,
        )
        event = {
            "type": "door_roi_detection",
            "detector": "vlm",
            "frame_time": frame_time,
            "image_width": frame.size[0],
            "image_height": frame.size[1],
            "found": bbox is not None,
            "bbox": bbox,
            "roi_margin": args.door_roi_margin,
            "confidence": coerce_confidence(parsed.get("confidence")),
            "reasoning": str(parsed.get("reasoning", "")),
            "raw_response": str(parsed.get("raw_response", "")),
            "debug_json": str(output_dir / "door_roi_bbox.json"),
            "debug_image": str(output_dir / "door_roi_overlay.jpg") if bbox is not None else None,
        }
        flow_events.append(event)
        if bbox is None:
            print("Door ROI detection did not find a valid bbox; using full frames.")
            return None
        print(
            "Door ROI detected "
            f"bbox=({bbox['x1']:.1f},{bbox['y1']:.1f},"
            f"{bbox['x2']:.1f},{bbox['y2']:.1f}) "
            f"conf={event['confidence']:.2f}"
        )
        return bbox
    finally:
        if frame is not None:
            close_frames([frame])
        cleanup_memory()


def detect_front_door_roi(
    *,
    cap: Any,
    frame_time: float,
    output_dir: Path,
    args: argparse.Namespace,
    backend: str,
    model_family: str,
    model_name: str,
    model: Any,
    processor: Any,
    openai_client: Any,
    flow_events: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    detector = "none" if args.disable_door_roi else args.door_roi_detector
    if detector == "none":
        flow_events.append(
            {
                "type": "door_roi_detection",
                "detector": "none",
                "frame_time": frame_time,
                "found": False,
                "disabled": True,
                "reasoning": "Door ROI localization disabled; using full frames.",
            }
        )
        return None

    explicit_cache_path = (
        Path(args.door_roi_bbox_path).expanduser()
        if args.door_roi_bbox_path
        else None
    )
    cached_bbox = load_cached_door_roi(
        output_dir=output_dir,
        flow_events=flow_events,
        cache_path=explicit_cache_path,
    )
    if cached_bbox is not None:
        frame, frame_error = read_frame_at_time(
            cap=cap,
            timestamp=frame_time,
            max_image_size=args.sam3d_door_image_size,
        )
        if frame is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                cleanup_stale_door_debug_images(output_dir)
                overlay = draw_labeled_bbox(frame, cached_bbox, "front door")
                overlay.save(output_dir / "door_roi_overlay.jpg", format="JPEG", quality=92)
            finally:
                close_frames([frame])
        elif frame_error is not None:
            print(f"Could not save cached door ROI overlay: {frame_error}")
        return cached_bbox

    if detector == "sam3d":
        return detect_front_door_roi_sam3d(
            cap=cap,
            frame_time=frame_time,
            output_dir=output_dir,
            args=args,
            flow_events=flow_events,
        )
    if detector == "vlm":
        return detect_front_door_roi_vlm(
            cap=cap,
            frame_time=frame_time,
            output_dir=output_dir,
            args=args,
            backend=backend,
            model_family=model_family,
            model_name=model_name,
            model=model,
            processor=processor,
            openai_client=openai_client,
            flow_events=flow_events,
        )
    raise ValueError(f"Unsupported door ROI detector: {detector}")


def detect_door_change_window(
    *,
    cap: Any,
    window_start: float,
    window_duration: float,
    door_roi_bbox: Optional[Dict[str, float]],
    args: argparse.Namespace,
    backend: str,
    model_family: str,
    model_name: str,
    model: Any,
    processor: Any,
    openai_client: Any,
    flow_events: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    frames: List[Image.Image] = []
    contact_sheet: Optional[Image.Image] = None
    frame_offsets: List[float] = []
    try:
        frames, frame_offsets, frame_error = sample_time_window_frames(
            cap=cap,
            start_time=window_start,
            duration=window_duration,
            num_frames=args.door_change_frames,
            max_image_size=args.door_image_size,
            roi_bbox=door_roi_bbox,
            roi_margin=args.door_roi_margin,
            roi_mode="masked_overlay",
            roi_label="front door",
        )
        if frame_error is not None:
            print(f"[{window_start:.2f}s] Door change window skipped: {frame_error}")
            return None, None

        contact_sheet = build_frame_contact_sheet(frames)
        sheet_rows, sheet_cols = contact_sheet_grid(len(frames))
        parsed = ask_json_probe(
            backend=backend,
            model_family=model_family,
            model_name=model_name,
            model=model,
            processor=processor,
            openai_client=openai_client,
            content=build_door_change_content(
                contact_sheet=contact_sheet,
                frame_count=len(frames),
                clip_start_time=window_start,
                clip_duration=window_duration,
                door_annotation_label="front door" if door_roi_bbox is not None else None,
            ),
            schema_name="door_status_change",
            schema=door_change_schema(),
            max_new_tokens=min(args.max_new_tokens, 160),
            detail=args.openai_detail,
        )
        status_changed = bool_from_probe(parsed, "status_changed")
        transition = normalize_transition(parsed.get("transition"))
        change_frame_index = normalize_change_frame_index(
            parsed,
            frame_count=len(frames),
            status_changed=status_changed,
        )
        offset = frame_index_to_offset(change_frame_index, frame_offsets)
        confidence = coerce_confidence(parsed.get("confidence"))
        event = {
            "type": "door_change_window",
            "window_start_time": window_start,
            "window_end_time": window_start + window_duration,
            "window_duration": window_duration,
            "window_frame_count": len(frames),
            "window_frame_offsets": frame_offsets,
            "door_contact_sheet_path": None,
            "door_contact_sheet_size": list(contact_sheet.size),
            "door_contact_sheet_grid": [sheet_rows, sheet_cols],
            "door_roi_bbox": door_roi_bbox,
            "door_roi_mode": "masked_overlay" if door_roi_bbox is not None else "none",
            "status_changed": status_changed,
            "transition": transition,
            "change_frame_index": change_frame_index,
            "change_time_offset": offset,
            "change_time": window_start + offset,
            "confidence": confidence,
            "reasoning": str(parsed.get("reasoning", "")),
        }
        flow_events.append(event)
        print(
            f"[{window_start:.2f}-{window_start + window_duration:.2f}s] "
            f"door_change={status_changed} transition={transition} "
            f"t={window_start + offset:.2f}s conf={confidence:.2f}"
        )
        if status_changed and transition in {"closed_to_open", "open_to_closed"}:
            return event.copy(), event
        return None, event
    finally:
        if contact_sheet is not None:
            close_frames([contact_sheet])
        close_frames(frames)
        cleanup_memory()


def verify_door_change_pin(
    *,
    cap: Any,
    candidate_pin: Dict[str, Any],
    video_duration: float,
    door_roi_bbox: Optional[Dict[str, float]],
    args: argparse.Namespace,
    backend: str,
    model_family: str,
    model_name: str,
    model: Any,
    processor: Any,
    openai_client: Any,
    flow_events: List[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    radius = args.door_change_verification_radius
    candidate_time = coerce_float(candidate_pin.get("change_time"), 0.0)
    verification_start = max(0.0, candidate_time - radius)
    verification_end = min(video_duration, candidate_time + radius)
    verification_duration = max(0.0, verification_end - verification_start)
    frames: List[Image.Image] = []
    contact_sheet: Optional[Image.Image] = None
    frame_offsets: List[float] = []

    try:
        candidate_transition = str(candidate_pin.get("transition", ""))
        frames, frame_offsets, frame_error = sample_time_window_frames(
            cap=cap,
            start_time=verification_start,
            duration=verification_duration,
            num_frames=args.door_change_verification_frames,
            max_image_size=args.door_image_size,
            roi_bbox=door_roi_bbox,
            roi_margin=args.door_roi_margin,
            roi_mode="masked_overlay",
            roi_label="front door",
        )
        if frame_error is not None:
            event = {
                "type": "door_change_verification",
                "verified": False,
                "candidate_transition": candidate_transition,
                "candidate_change_time": candidate_time,
                "verification_start_time": verification_start,
                "verification_end_time": verification_end,
                "verification_duration": verification_duration,
                "verification_frame_count": 0,
                "door_roi_bbox": door_roi_bbox,
                "door_roi_mode": "masked_overlay" if door_roi_bbox is not None else "none",
                "reasoning": frame_error,
            }
            flow_events.append(event)
            print(
                f"  Door verification failed at t={candidate_time:.2f}s: "
                f"{frame_error}"
            )
            return False, event

        contact_sheet = build_frame_contact_sheet(frames)
        sheet_rows, sheet_cols = contact_sheet_grid(len(frames))
        parsed = ask_json_probe(
            backend=backend,
            model_family=model_family,
            model_name=model_name,
            model=model,
            processor=processor,
            openai_client=openai_client,
            content=build_door_change_verification_content(
                contact_sheet=contact_sheet,
                frame_count=len(frames),
                clip_start_time=verification_start,
                clip_duration=verification_duration,
                candidate_transition=candidate_transition,
                candidate_change_time=candidate_time,
                door_annotation_label="front door" if door_roi_bbox is not None else None,
            ),
            schema_name="door_status_change_verification",
            schema=door_change_schema(),
            max_new_tokens=min(args.max_new_tokens, 160),
            detail=args.openai_detail,
        )
        status_changed = bool_from_probe(parsed, "status_changed")
        transition = normalize_transition(parsed.get("transition"))
        change_frame_index = normalize_change_frame_index(
            parsed,
            frame_count=len(frames),
            status_changed=status_changed,
        )
        offset = frame_index_to_offset(change_frame_index, frame_offsets)
        confidence = coerce_confidence(parsed.get("confidence"))
        verified = (
            status_changed
            and transition in {"closed_to_open", "open_to_closed"}
            and transition == candidate_transition
        )
        contact_sheet_path: Optional[Path] = None
        if verified:
            contact_sheet_path = save_door_contact_sheet(
                args=args,
                contact_sheet=contact_sheet,
                frames=frames,
                sheet_kind="verification",
                start_time=verification_start,
                end_time=verification_end,
                selected_frame_index=change_frame_index,
            )
        event = {
            "type": "door_change_verification",
            "verified": verified,
            "candidate_transition": candidate_transition,
            "candidate_change_time": candidate_time,
            "verification_start_time": verification_start,
            "verification_end_time": verification_end,
            "verification_duration": verification_duration,
            "verification_frame_count": len(frames),
            "verification_frame_offsets": frame_offsets,
            "door_contact_sheet_path": str(contact_sheet_path) if contact_sheet_path else None,
            "door_contact_sheet_size": list(contact_sheet.size),
            "door_contact_sheet_grid": [sheet_rows, sheet_cols],
            "door_roi_bbox": door_roi_bbox,
            "door_roi_mode": "masked_overlay" if door_roi_bbox is not None else "none",
            "status_changed": status_changed,
            "transition": transition,
            "change_frame_index": change_frame_index,
            "change_time_offset": offset,
            "change_time": verification_start + offset,
            "confidence": confidence,
            "reasoning": str(parsed.get("reasoning", "")),
        }
        flow_events.append(event)
        print(
            f"  Door verification t={candidate_time:.2f}s "
            f"candidate={candidate_transition} answer={transition} "
            f"changed={status_changed} verified={verified} conf={confidence:.2f}"
        )
        return verified, event
    finally:
        if contact_sheet is not None:
            close_frames([contact_sheet])
        close_frames(frames)
        cleanup_memory()


def door_bbox_to_pixels_for_frame(
    bbox: Optional[Dict[str, float]],
    frame_width: int,
    frame_height: int,
    margin: float = 0.0,
) -> Tuple[int, int, int, int]:
    if bbox is None:
        return 0, 0, frame_width, frame_height

    if all(key in bbox for key in ("x1_norm", "y1_norm", "x2_norm", "y2_norm")):
        x1 = coerce_float(bbox.get("x1_norm"), 0.0) * frame_width
        y1 = coerce_float(bbox.get("y1_norm"), 0.0) * frame_height
        x2 = coerce_float(bbox.get("x2_norm"), 1.0) * frame_width
        y2 = coerce_float(bbox.get("y2_norm"), 1.0) * frame_height
    else:
        x1 = coerce_float(bbox.get("x1"), 0.0)
        y1 = coerce_float(bbox.get("y1"), 0.0)
        x2 = coerce_float(bbox.get("x2"), float(frame_width))
        y2 = coerce_float(bbox.get("y2"), float(frame_height))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(0.0, margin) * box_width
    pad_y = max(0.0, margin) * box_height
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(frame_width, int(round(x2 + pad_x)))
    bottom = min(frame_height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return 0, 0, frame_width, frame_height
    return left, top, right, bottom


def normalize_door_status_label(label: Any, class_id: int) -> str:
    text = str(label).strip().lower()
    if "open" in text:
        return "open"
    if "closed" in text or "close" in text:
        return "closed"
    if class_id == 1:
        return "open"
    if class_id == 0:
        return "closed"
    return "unknown"


def classify_door_yolo_batch(model: Any, crops: List[Any]) -> List[Dict[str, Any]]:
    results = model.predict(crops, verbose=False)
    names = getattr(model, "names", {}) or {}
    predictions: List[Dict[str, Any]] = []
    for result in results:
        probs = getattr(result, "probs", None)
        if probs is None:
            predictions.append(
                {
                    "status": "unknown",
                    "class_id": -1,
                    "label": "unknown",
                    "confidence": 0.0,
                }
            )
            continue
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        label = names.get(class_id, class_id)
        predictions.append(
            {
                "status": normalize_door_status_label(label, class_id),
                "class_id": class_id,
                "label": str(label),
                "confidence": confidence,
            }
        )
    return predictions


def scan_door_changes_yolo(
    *,
    cap: Any,
    door_roi_bbox: Optional[Dict[str, float]],
    args: argparse.Namespace,
    video_duration: float,
    processing_end_time: float,
    flow_events: List[Dict[str, Any]],
    on_pin: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO door tracking requires ultralytics. Install it with "
            "`pip install ultralytics`, or run with --door_track vlm."
        ) from exc

    model_path = Path(args.door_yolo_model)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO door model does not exist: {model_path}")

    model = YOLO(str(model_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(args.st_time * fps)))
    end_frame = int(round(processing_end_time * fps))
    if total_frames > 0:
        end_frame = min(end_frame, total_frames)
    if end_frame <= start_frame:
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError(f"Could not decode frame at {args.st_time:.2f}s.")

    frame_height, frame_width = first_frame.shape[:2]
    left, top, right, bottom = door_bbox_to_pixels_for_frame(
        door_roi_bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        margin=args.door_yolo_crop_margin,
    )

    pins: List[Dict[str, Any]] = []
    batch_frames: List[int] = []
    batch_times: List[float] = []
    batch_crops: List[Any] = []
    batch_size = max(1, int(args.door_yolo_batch_size))
    min_consecutive = max(1, int(args.door_yolo_min_consecutive_frames))
    min_confidence = coerce_confidence(args.door_yolo_min_confidence)

    stable_status: Optional[str] = None
    previous_prediction: Optional[Dict[str, Any]] = None
    candidate_status: Optional[str] = None
    candidate_start_frame: Optional[int] = None
    candidate_start_time: Optional[float] = None
    candidate_confidences: List[float] = []
    processed_frames = 0

    def consume_predictions(
        frame_indices: List[int],
        times: List[float],
        crops: List[Any],
    ) -> None:
        nonlocal stable_status
        nonlocal previous_prediction
        nonlocal candidate_status
        nonlocal candidate_start_frame
        nonlocal candidate_start_time
        nonlocal candidate_confidences
        nonlocal processed_frames

        predictions = classify_door_yolo_batch(model, crops)
        for frame_index, frame_time, prediction in zip(
            frame_indices,
            times,
            predictions,
        ):
            status = prediction["status"]
            confidence = coerce_confidence(prediction.get("confidence"))
            if confidence < min_confidence:
                status = (
                    previous_prediction.get("effective_status")
                    if previous_prediction is not None
                    else stable_status
                ) or "unknown"
                prediction = {
                    **prediction,
                    "effective_status": status,
                    "low_confidence_reused_previous_status": True,
                    "min_confidence": min_confidence,
                }
            else:
                prediction = {
                    **prediction,
                    "effective_status": status,
                    "low_confidence_reused_previous_status": False,
                    "min_confidence": min_confidence,
                }
            if status not in {"open", "closed"}:
                previous_prediction = prediction
                processed_frames += 1
                continue

            if stable_status is None:
                stable_status = status
                previous_prediction = prediction
                processed_frames += 1
                continue

            if status == stable_status:
                candidate_status = None
                candidate_start_frame = None
                candidate_start_time = None
                candidate_confidences = []
                previous_prediction = prediction
                processed_frames += 1
                continue

            if status != candidate_status:
                candidate_status = status
                candidate_start_frame = frame_index
                candidate_start_time = frame_time
                candidate_confidences = [coerce_confidence(prediction["confidence"])]
            else:
                candidate_confidences.append(
                    coerce_confidence(prediction["confidence"])
                )

            if (
                candidate_status is not None
                and candidate_start_frame is not None
                and candidate_start_time is not None
                and len(candidate_confidences) >= min_consecutive
            ):
                transition = f"{stable_status}_to_{candidate_status}"
                if transition in {"closed_to_open", "open_to_closed"}:
                    confidence = (
                        sum(candidate_confidences) / len(candidate_confidences)
                    )
                    pin = {
                        "type": "door_change_yolo",
                        "detector": "yolo",
                        "model": str(model_path),
                        "transition": transition,
                        "status_changed": True,
                        "change_frame_index": candidate_start_frame,
                        "change_time": candidate_start_time,
                        "confidence": confidence,
                        "stable_from_status": stable_status,
                        "stable_to_status": candidate_status,
                        "min_consecutive_frames": min_consecutive,
                        "door_roi_bbox": door_roi_bbox,
                        "door_roi_mode": "crop" if door_roi_bbox is not None else "full_frame",
                        "reasoning": (
                            f"YOLO door classifier changed from {stable_status} "
                            f"to {candidate_status} for at least "
                            f"{min_consecutive} consecutive frame(s)."
                        ),
                    }
                    pins.append(pin)
                    flow_events.append(pin.copy())
                    print(
                        "  YOLO door change "
                        f"{transition} at {candidate_start_time:.2f}s "
                        f"conf={confidence:.2f}"
                    )
                    if on_pin is not None:
                        on_pin(pin.copy())
                stable_status = candidate_status
                candidate_status = None
                candidate_start_frame = None
                candidate_start_time = None
                candidate_confidences = []

            previous_prediction = prediction
            processed_frames += 1

    current_frame = start_frame
    frame = first_frame
    while current_frame < end_frame:
        if frame is None:
            break
        crop = frame[top:bottom, left:right]
        batch_frames.append(current_frame)
        batch_times.append(current_frame / fps)
        batch_crops.append(crop)
        if len(batch_crops) >= batch_size:
            consume_predictions(batch_frames, batch_times, batch_crops)
            batch_frames.clear()
            batch_times.clear()
            batch_crops.clear()
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)

        current_frame += 1
        if current_frame >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break

    if batch_crops:
        consume_predictions(batch_frames, batch_times, batch_crops)

    flow_events.append(
        {
            "type": "door_track_yolo_summary",
            "model": str(model_path),
            "processing_start_time": args.st_time,
            "processing_end_time": processing_end_time,
            "video_duration": video_duration,
            "fps": fps,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "processed_frame_count": processed_frames,
            "pin_count": len(pins),
            "door_roi_bbox": door_roi_bbox,
            "door_roi_crop_pixels": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            "min_consecutive_frames": min_consecutive,
            "min_confidence": min_confidence,
            "crop_margin": args.door_yolo_crop_margin,
        }
    )
    return pins


def bbox_iou(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_bbox = left.get("bbox") if isinstance(left.get("bbox"), dict) else left
    right_bbox = right.get("bbox") if isinstance(right.get("bbox"), dict) else right
    ax1 = coerce_float(left_bbox.get("x1"), 0.0)
    ay1 = coerce_float(left_bbox.get("y1"), 0.0)
    ax2 = coerce_float(left_bbox.get("x2"), 0.0)
    ay2 = coerce_float(left_bbox.get("y2"), 0.0)
    bx1 = coerce_float(right_bbox.get("x1"), 0.0)
    by1 = coerce_float(right_bbox.get("y1"), 0.0)
    bx2 = coerce_float(right_bbox.get("x2"), 0.0)
    by2 = coerce_float(right_bbox.get("y2"), 0.0)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_left + area_right - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def run_sam3_passenger_batch(
    *,
    image_paths: List[Path],
    args: argparse.Namespace,
    raw_json_path: Path,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("sam3_door_detector.py")),
        "--model",
        "facebook/sam3",
        "--images",
        *[str(path) for path in image_paths],
        "--prompt",
        args.sam3d_passenger_prompt,
        "--thresholds",
        args.sam3d_passenger_thresholds,
        "--all-instances",
        "--track-video",
        "--min-box-area-ratio",
        str(args.sam3d_passenger_min_box_area_ratio),
        "--max-width-height-ratio",
        str(args.sam3d_passenger_max_width_height_ratio),
        "--output",
        str(raw_json_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        text=True,
        capture_output=True,
        timeout=args.sam3d_passenger_timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SAM3 passenger tracking failed with exit code "
            f"{completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return read_json_payload(raw_json_path)


def build_sam3_passenger_sheet(
    *,
    frames: List[Image.Image],
    stop_id: int,
    args: argparse.Namespace,
) -> Tuple[Image.Image, Path, Dict[str, Any]]:
    output_dir = Path(args.output_dir) / "passenger_sam3d_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / f"stop_{stop_id:04d}_passenger_detections_sheet.jpg"
    annotated_frames: List[Image.Image] = []
    try:
        with tempfile.TemporaryDirectory(prefix="passenger_sam3d_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_paths: List[Path] = []
            for frame_index, frame in enumerate(frames, start=1):
                image_path = tmp_path / f"frame_{frame_index:02d}.jpg"
                frame.save(image_path, format="JPEG", quality=92)
                image_paths.append(image_path)

            batch_size = max(1, int(args.sam3d_passenger_batch_size))
            image_results: List[Dict[str, Any]] = []
            batch_metadata: List[Dict[str, Any]] = []
            previous_global_detections: List[Dict[str, Any]] = []
            next_global_id = 1

            for batch_index, batch_start in enumerate(
                range(0, len(image_paths), batch_size),
                start=1,
            ):
                batch_paths = image_paths[batch_start : batch_start + batch_size]
                raw_json_path = tmp_path / f"detections_batch_{batch_index:04d}.json"
                payload = run_sam3_passenger_batch(
                    image_paths=batch_paths,
                    args=args,
                    raw_json_path=raw_json_path,
                )
                batch_results = (
                    payload.get("images", []) if isinstance(payload, dict) else []
                )
                normalized_batch_results: List[Dict[str, Any]] = []
                for local_index, image_path in enumerate(batch_paths):
                    image_result = (
                        batch_results[local_index]
                        if local_index < len(batch_results)
                        and isinstance(batch_results[local_index], dict)
                        else {}
                    )
                    normalized_batch_results.append(
                        {
                            "image": str(image_path),
                            "frame_index": local_index,
                            "found": False,
                            "detections": [],
                            **image_result,
                        }
                    )
                batch_results = normalized_batch_results
                id_map: Dict[int, int] = {}
                first_batch_detections = []
                if batch_results and isinstance(batch_results[0], dict):
                    first_batch_detections = [
                        item
                        for item in batch_results[0].get("detections", [])
                        if isinstance(item, dict)
                    ]

                used_previous_ids = set()
                for detection in first_batch_detections:
                    local_id = coerce_int(detection.get("object_id"), -1)
                    if local_id < 0 or local_id in id_map:
                        continue
                    best_match = None
                    best_iou = 0.0
                    for previous_detection in previous_global_detections:
                        previous_id = coerce_int(previous_detection.get("object_id"), -1)
                        if previous_id in used_previous_ids:
                            continue
                        iou = bbox_iou(detection, previous_detection)
                        if iou > best_iou:
                            best_match = previous_id
                            best_iou = iou
                    if best_match is not None and best_iou >= 0.20:
                        id_map[local_id] = best_match
                        used_previous_ids.add(best_match)

                for image_result in batch_results:
                    if not isinstance(image_result, dict):
                        continue
                    image_result = dict(image_result)
                    image_result["global_frame_index"] = (
                        batch_start + coerce_int(image_result.get("frame_index"), 0)
                    )
                    detections: List[Dict[str, Any]] = []
                    for detection in image_result.get("detections", []):
                        if not isinstance(detection, dict):
                            continue
                        detection = dict(detection)
                        local_id = coerce_int(detection.get("object_id"), -1)
                        if local_id < 0:
                            continue
                        if local_id not in id_map:
                            id_map[local_id] = next_global_id
                            next_global_id += 1
                        detection["local_object_id"] = local_id
                        detection["object_id"] = id_map[local_id]
                        detection["sam3_batch_index"] = batch_index
                        detections.append(detection)
                    image_result["detections"] = detections
                    image_result["found"] = bool(detections)
                    image_results.append(image_result)

                last_batch_detections: List[Dict[str, Any]] = []
                for image_result in reversed(batch_results):
                    if isinstance(image_result, dict) and image_result.get("detections"):
                        for detection in image_result.get("detections", []):
                            if not isinstance(detection, dict):
                                continue
                            local_id = coerce_int(detection.get("object_id"), -1)
                            if local_id in id_map:
                                last_batch_detections.append(
                                    {**detection, "object_id": id_map[local_id]}
                                )
                        break
                previous_global_detections = last_batch_detections
                batch_metadata.append(
                    {
                        "batch_index": batch_index,
                        "start_frame_index": batch_start + 1,
                        "end_frame_index": batch_start + len(batch_paths),
                        "frame_count": len(batch_paths),
                        "id_map": {str(k): v for k, v in sorted(id_map.items())},
                        "source": payload.get("source") if isinstance(payload, dict) else "",
                        "threshold": payload.get("threshold") if isinstance(payload, dict) else None,
                        "postprocessing": (
                            payload.get("postprocessing", {})
                            if isinstance(payload, dict)
                            else {}
                        ),
                    }
                )

        tracked_ids = set()
        frame_detections: List[Dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            image_result = (
                image_results[frame_index]
                if frame_index < len(image_results)
                and isinstance(image_results[frame_index], dict)
                else {}
            )
            boxes: List[Dict[str, float]] = []
            instances: List[Dict[str, Any]] = []
            for detection in image_result.get("detections", []):
                if not isinstance(detection, dict):
                    continue
                bbox = normalize_door_bbox(
                    detection,
                    image_width=frame.size[0],
                    image_height=frame.size[1],
                )
                if bbox is None:
                    continue
                object_id = coerce_int(detection.get("object_id"), -1)
                tracked_ids.add(object_id)
                boxes.append({**bbox, "object_id": object_id})
                box_width = max(0.0, bbox["x2"] - bbox["x1"])
                box_height = max(0.0, bbox["y2"] - bbox["y1"])
                instances.append(
                    {
                        "object_id": object_id,
                        "bbox": bbox,
                        "box_area": box_width * box_height,
                        "confidence": coerce_confidence(
                            detection.get("confidence")
                        ),
                    }
                )
            annotated_frames.append(
                draw_tracked_passenger_bboxes(frame, boxes)
            )
            frame_detections.append(
                {
                    "frame_index": frame_index + 1,
                    "instances": instances,
                }
            )

        sheet = build_frame_contact_sheet(annotated_frames)
        sheet.save(sheet_path, format="JPEG", quality=92)
        debug_metadata = {
            "tracking": True,
            "tracker_backend": "sam3",
            "prompt": args.sam3d_passenger_prompt,
            "thresholds": args.sam3d_passenger_thresholds,
            "frame_sampling": {
                "mode": "original_frame_stride",
                "frame_stride": args.passenger_split_frame_stride,
                "frame_count": len(frames),
            },
            "sam3_batching": {
                "batch_size": max(1, int(args.sam3d_passenger_batch_size)),
                "batch_count": len(batch_metadata),
                "batches": batch_metadata,
                "id_stitching": "bbox_iou_boundary_match",
                "id_stitching_iou_threshold": 0.40,
            },
            "tracked_object_ids": sorted(item for item in tracked_ids if item >= 0),
            "frame_detections": frame_detections,
        }
        return sheet, sheet_path, debug_metadata
    finally:
        close_frames(annotated_frames)
        cleanup_memory()


def write_pil_frames_to_video(
    *,
    frames: List[Image.Image],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    if cv2 is None or np is None:
        return "YOLO passenger tracking requires OpenCV and NumPy."
    if not frames:
        return "No frames available for YOLO passenger tracking."

    first = frames[0].convert("RGB")
    width, height = first.size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        return f"Could not open video writer for {output_path}."

    try:
        for frame in frames:
            rgb = frame.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height))
            writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return None


def build_yolo_passenger_sheet(
    *,
    frames: List[Image.Image],
    stop_id: int,
    args: argparse.Namespace,
) -> Tuple[Image.Image, Path, Dict[str, Any]]:
    output_dir = Path(args.output_dir) / "passenger_yolo_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / f"stop_{stop_id:04d}_passenger_detections_sheet.jpg"
    annotated_frames: List[Image.Image] = []
    try:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO passenger tracking requires ultralytics. Install it with "
                "`pip install ultralytics`, or run with --track sam3."
            ) from exc

        with tempfile.TemporaryDirectory(prefix="passenger_yolo_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / f"stop_{stop_id:04d}_passenger_frames.mp4"
            video_error = write_pil_frames_to_video(
                frames=frames,
                output_path=video_path,
            )
            if video_error is not None:
                raise RuntimeError(video_error)

            model = YOLO(args.yolo_passenger_model)
            results = model.track(
                source=str(video_path),
                tracker=args.yolo_passenger_tracker,
                conf=args.yolo_passenger_conf,
                iou=args.yolo_passenger_iou,
                classes=[0],
                persist=True,
                stream=False,
                verbose=False,
            )

        tracked_ids = set()
        frame_detections: List[Dict[str, Any]] = []
        frame_results = list(results) if results is not None else []
        min_box_area_ratio = args.sam3d_passenger_min_box_area_ratio
        max_width_height_ratio = args.sam3d_passenger_max_width_height_ratio

        for frame_index, frame in enumerate(frames):
            result = frame_results[frame_index] if frame_index < len(frame_results) else None
            boxes_for_draw: List[Dict[str, float]] = []
            instances: List[Dict[str, Any]] = []
            frame_width, frame_height = frame.size
            frame_area = float(frame_width * frame_height)

            result_boxes = getattr(result, "boxes", None) if result is not None else None
            track_ids = getattr(result_boxes, "id", None) if result_boxes is not None else None
            xyxy = getattr(result_boxes, "xyxy", None) if result_boxes is not None else None
            confidences = getattr(result_boxes, "conf", None) if result_boxes is not None else None
            classes = getattr(result_boxes, "cls", None) if result_boxes is not None else None

            if track_ids is not None and xyxy is not None:
                ids_array = tensor_to_numpy(track_ids).reshape(-1)
                xyxy_array = tensor_to_numpy(xyxy).reshape(-1, 4)
                conf_array = (
                    tensor_to_numpy(confidences).reshape(-1)
                    if confidences is not None
                    else np.ones(len(ids_array), dtype=np.float32)
                )
                cls_array = (
                    tensor_to_numpy(classes).reshape(-1)
                    if classes is not None
                    else np.zeros(len(ids_array), dtype=np.float32)
                )

                for det_index, track_id in enumerate(ids_array):
                    if det_index >= len(xyxy_array):
                        continue
                    class_id = int(round(float(cls_array[det_index])))
                    if class_id != 0:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in xyxy_array[det_index]]
                    x1 = max(0.0, min(float(frame_width - 1), x1))
                    x2 = max(0.0, min(float(frame_width - 1), x2))
                    y1 = max(0.0, min(float(frame_height - 1), y1))
                    y2 = max(0.0, min(float(frame_height - 1), y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    box_width = x2 - x1
                    box_height = y2 - y1
                    width_height_ratio = box_width / box_height if box_height > 0 else 0.0
                    box_area_ratio = (
                        box_width * box_height / frame_area
                        if frame_area > 0
                        else 0.0
                    )
                    if (
                        width_height_ratio > max_width_height_ratio
                        or box_area_ratio < min_box_area_ratio
                    ):
                        continue

                    object_id = int(round(float(track_id)))
                    bbox = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "x1_norm": x1 / frame_width,
                        "y1_norm": y1 / frame_height,
                        "x2_norm": x2 / frame_width,
                        "y2_norm": y2 / frame_height,
                    }
                    confidence = coerce_confidence(conf_array[det_index])
                    tracked_ids.add(object_id)
                    boxes_for_draw.append({**bbox, "object_id": object_id})
                    instances.append(
                        {
                            "object_id": object_id,
                            "bbox": bbox,
                            "box_area": box_width * box_height,
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": "person",
                            "width_height_ratio": width_height_ratio,
                            "box_area_ratio": box_area_ratio,
                        }
                    )

            annotated_frames.append(draw_tracked_passenger_bboxes(frame, boxes_for_draw))
            frame_detections.append(
                {
                    "frame_index": frame_index + 1,
                    "instances": instances,
                }
            )

        sheet = build_frame_contact_sheet(annotated_frames)
        sheet.save(sheet_path, format="JPEG", quality=92)
        debug_metadata = {
            "tracking": True,
            "tracker_backend": "yolo",
            "detector": "YOLO11",
            "model": args.yolo_passenger_model,
            "tracker": args.yolo_passenger_tracker,
            "confidence_threshold": args.yolo_passenger_conf,
            "nms_iou_threshold": args.yolo_passenger_iou,
            "class_filter": {"class_id": 0, "class_name": "person"},
            "postprocessing": {
                "minimum_box_area_ratio": min_box_area_ratio,
                "maximum_width_height_ratio": max_width_height_ratio,
            },
            "frame_sampling": {
                "mode": "original_frame_stride",
                "frame_stride": args.passenger_split_frame_stride,
                "frame_count": len(frames),
            },
            "tracked_object_ids": sorted(item for item in tracked_ids if item >= 0),
            "frame_detections": frame_detections,
        }
        return sheet, sheet_path, debug_metadata
    finally:
        close_frames(annotated_frames)
        cleanup_memory()


def build_passenger_tracking_sheet(
    *,
    frames: List[Image.Image],
    stop_id: int,
    args: argparse.Namespace,
) -> Tuple[Image.Image, Path, Dict[str, Any]]:
    if args.track == "yolo":
        return build_yolo_passenger_sheet(
            frames=frames,
            stop_id=stop_id,
            args=args,
        )
    return build_sam3_passenger_sheet(
        frames=frames,
        stop_id=stop_id,
        args=args,
    )


def split_payment_activated_interval(
    *,
    cap: Any,
    stop_id: int,
    stop_start: float,
    stop_end: float,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stop_duration = max(0.0, stop_end - stop_start)
    frames: List[Image.Image] = []
    contact_sheet: Optional[Image.Image] = None
    frame_offsets: List[float] = []
    try:
        frames, frame_offsets, frame_error = sample_time_window_frames_by_stride(
            cap=cap,
            start_time=stop_start,
            end_time=stop_end,
            frame_stride=args.passenger_split_frame_stride,
            max_image_size=args.passenger_split_image_size,
            lighting_preprocess=not args.no_lighting_preprocess,
        )
        if frame_error is not None:
            split_event = {
                "type": "passenger_split",
                "stop_id": stop_id,
                "start_time": stop_start,
                "end_time": stop_end,
                "duration": stop_duration,
                "frame_count": 0,
                "frame_offsets": [],
                "frame_sampling": {
                    "mode": "original_frame_stride",
                    "frame_stride": args.passenger_split_frame_stride,
                },
                "frame_preprocessing": {
                    "lighting": "shadow_boost_highlight_suppression"
                    if not args.no_lighting_preprocess
                    else "disabled",
                },
                "passenger_count": 0,
                "split_points": [],
                "clips": [],
                "confidence": 0.0,
                "reasoning": frame_error,
            }
            return [], split_event

        contact_sheet, passenger_sheet_path, passenger_tracker_debug = (
            build_passenger_tracking_sheet(
                frames=frames,
                stop_id=stop_id,
                args=args,
            )
        )
        sheet_rows, sheet_cols = contact_sheet_grid(len(frames))
        (
            rule_main_passenger_ids_raw,
            rule_main_passenger_ids_repaired,
            rule_candidate_transitions,
            _rule_split_points,
            rule_main_passenger_id_repair,
        ) = derive_largest_box_passenger_changes(
            passenger_tracker_debug.get("frame_detections", []),
            frame_offsets=frame_offsets,
            stop_duration=stop_duration,
        )
        clips, id_run_segments, split_points = build_payment_clips_from_main_id_runs(
            stop_start=stop_start,
            stop_end=stop_end,
            main_passenger_ids=rule_main_passenger_ids_repaired,
            frame_offsets=frame_offsets,
            min_duration=args.min_clip_duration,
            frame_merge_distance=args.passenger_split_merge_frame_distance,
        )
        passenger_clip_sheet_dir = Path(args.output_dir) / "passenger_clip_sheets"
        frame_detections = passenger_tracker_debug.get("frame_detections", [])
        for clip in clips:
            passenger_index = coerce_int(clip.get("passenger_index"), 0)
            sheet_path = (
                passenger_clip_sheet_dir
                / f"stop_{stop_id:04d}_passenger_{passenger_index:02d}_"
                f"{clip['start_time']:.2f}_{clip['end_time']:.2f}.jpg"
            )
            sheet_error = save_passenger_clip_annotation_sheet(
                frames=frames,
                frame_detections=frame_detections
                if isinstance(frame_detections, list)
                else [],
                clip=clip,
                sheet_path=sheet_path,
            )
            if sheet_error is None:
                clip["passenger_sheet_path"] = str(sheet_path)
            else:
                clip["passenger_sheet_error"] = sheet_error
        selected_split_frame_indices = [
            coerce_int(segment.get("start_frame_index"), 0)
            for segment in id_run_segments
            if coerce_int(segment.get("start_frame_index"), 0) > 0
        ]
        print(
            "  Repaired-ID passenger clip runs: "
            f"{selected_split_frame_indices or 'none'}"
        )
        highlighted_sheet = highlight_contact_sheet_frames(
            contact_sheet=contact_sheet,
            frames=frames,
            selected_frame_indices=selected_split_frame_indices,
        )
        highlighted_sheet.save(passenger_sheet_path, format="JPEG", quality=92)
        if highlighted_sheet is not contact_sheet:
            contact_sheet.close()
            contact_sheet = highlighted_sheet
        passenger_count = len(clips)
        passenger_split_method = "repaired_main_id_runs"
        final_confidence = 1.0
        final_reasoning = (
            "Built passenger clips from contiguous non-empty repaired main "
            "passenger ID runs."
        )
        final_raw_response = ""

        split_event = {
            "type": "passenger_split",
            "stop_id": stop_id,
            "start_time": stop_start,
            "end_time": stop_end,
            "duration": stop_duration,
            "frame_count": len(frames),
            "frame_offsets": frame_offsets,
            "frame_sampling": {
                "mode": "original_frame_stride",
                "frame_stride": args.passenger_split_frame_stride,
            },
            "frame_preprocessing": {
                "lighting": "shadow_boost_highlight_suppression"
                if not args.no_lighting_preprocess
                else "disabled",
            },
            "passenger_contact_sheet_size": list(contact_sheet.size),
            "passenger_contact_sheet_grid": [sheet_rows, sheet_cols],
            "passenger_contact_sheet_source": f"{args.track}_tracking_overlay",
            "passenger_contact_sheet_path": str(passenger_sheet_path),
            "passenger_clip_source": (
                f"repaired_main_id_runs_from_{args.track}_tracking"
            ),
            "passenger_activity_classification": {
                "enabled": False,
                "input": None,
                "labels": [],
            },
            "selected_split_frame_indices": selected_split_frame_indices,
            "passenger_tracker": args.track,
            "passenger_tracker_debug": passenger_tracker_debug,
            "passenger_sam3d_debug": passenger_tracker_debug
            if args.track == "sam3"
            else None,
            "passenger_split_method": passenger_split_method,
            "id_run_merge_frame_distance": args.passenger_split_merge_frame_distance,
            "rule_main_passenger_ids_raw": rule_main_passenger_ids_raw,
            "rule_main_passenger_ids_repaired": rule_main_passenger_ids_repaired,
            "rule_main_passenger_id_repair": rule_main_passenger_id_repair,
            "rule_candidate_transitions": rule_candidate_transitions,
            "id_run_segments": id_run_segments,
            "passenger_count": passenger_count,
            "split_points": [
                {
                    **point,
                    "absolute_time": stop_start + point["time_offset"],
                }
                for point in split_points
            ],
            "clips": clips,
            "confidence": final_confidence,
            "reasoning": final_reasoning,
            "raw_response": final_raw_response,
        }
        return clips, split_event
    finally:
        if contact_sheet is not None:
            close_frames([contact_sheet])
        close_frames(frames)
        cleanup_memory()


def process_streaming_video(args: argparse.Namespace, clip_callback=None) -> List[Dict[str, Any]]:
    require_common_dependencies()
    effective_door_roi_detector = (
        "none" if args.disable_door_roi else args.door_roi_detector
    )
    needs_vlm_runtime = (
        args.door_track == "vlm" or effective_door_roi_detector == "vlm"
    )
    default_model_name = args.model_name or args.model
    clip_model_name = ""
    clip_backend = "none"
    if needs_vlm_runtime:
        clip_model_name, clip_backend = resolve_model_and_backend(
            args.model_clip or default_model_name,
            args.backend,
        )
        if clip_backend == "openai":
            require_openai_dependencies()
        if clip_backend == "local":
            require_local_dependencies(clip_model_name)

    if args.dataset_root is None:
        raise ValueError("--dataset_root must point to the continuous input video.")
    video_path = Path(args.dataset_root)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"--dataset_root must be a video file, got: {video_path}")
    if args.door_change_window <= 0:
        raise ValueError("--door_change_window must be greater than 0.")
    if args.door_yolo_batch_size <= 0:
        raise ValueError("--door_yolo_batch_size must be greater than 0.")
    if args.door_yolo_min_consecutive_frames <= 0:
        raise ValueError("--door_yolo_min_consecutive_frames must be greater than 0.")
    if args.door_yolo_crop_margin < 0:
        raise ValueError("--door_yolo_crop_margin must be greater than or equal to 0.")
    if not 0.0 <= args.door_yolo_min_confidence <= 1.0:
        raise ValueError("--door_yolo_min_confidence must be between 0 and 1.")
    if args.door_track == "yolo":
        door_yolo_model = Path(args.door_yolo_model)
        if not door_yolo_model.exists():
            raise FileNotFoundError(f"--door_yolo_model does not exist: {door_yolo_model}")
    if args.door_change_overlap < 0:
        raise ValueError("--door_change_overlap must be greater than or equal to 0.")
    if args.door_change_overlap >= args.door_change_window:
        raise ValueError("--door_change_overlap must be smaller than --door_change_window.")
    if args.door_change_frames <= 0:
        raise ValueError("--door_change_frames must be greater than 0.")
    if args.door_change_merge_tolerance < 0:
        raise ValueError("--door_change_merge_tolerance must be greater than or equal to 0.")
    if args.door_change_verification_radius <= 0:
        raise ValueError("--door_change_verification_radius must be greater than 0.")
    if args.door_change_verification_frames <= 0:
        raise ValueError("--door_change_verification_frames must be greater than 0.")
    if args.door_image_size < 0:
        raise ValueError("--door_image_size must be greater than or equal to 0.")
    if args.sam3d_door_image_size < 0:
        raise ValueError("--sam3d_door_image_size must be greater than or equal to 0.")
    if args.door_roi_margin < 0:
        raise ValueError("--door_roi_margin must be greater than or equal to 0.")
    if args.sam3d_door_timeout <= 0:
        raise ValueError("--sam3d_door_timeout must be greater than 0.")
    if args.sam3d_passenger_timeout <= 0:
        raise ValueError("--sam3d_passenger_timeout must be greater than 0.")
    if args.sam3d_passenger_batch_size <= 0:
        raise ValueError("--sam3d_passenger_batch_size must be greater than 0.")
    if not 0.0 <= args.sam3d_passenger_min_box_area_ratio <= 1.0:
        raise ValueError(
            "--sam3d_passenger_min_box_area_ratio must be between 0 and 1."
        )
    if args.sam3d_passenger_max_width_height_ratio <= 0:
        raise ValueError(
            "--sam3d_passenger_max_width_height_ratio must be greater than 0."
        )
    if not 0.0 <= args.yolo_passenger_conf <= 1.0:
        raise ValueError("--yolo_passenger_conf must be between 0 and 1.")
    if not 0.0 <= args.yolo_passenger_iou <= 1.0:
        raise ValueError("--yolo_passenger_iou must be between 0 and 1.")
    if args.track == "yolo" and (cv2 is None or np is None):
        raise ImportError(
            "YOLO passenger tracking requires OpenCV and NumPy. Install them "
            "or run with --track sam3."
        )
    if args.passenger_split_image_size < 0:
        raise ValueError("--passenger_split_image_size must be greater than or equal to 0.")
    if args.passenger_split_frame_stride <= 0:
        raise ValueError("--passenger_split_frame_stride must be greater than 0.")
    if args.passenger_split_merge_frame_distance < -1:
        raise ValueError(
            "--passenger_split_merge_frame_distance must be greater than or equal to -1."
        )
    if not args.no_lighting_preprocess and np is None:
        raise ImportError(
            "Lighting preprocessing requires numpy. Install it or pass "
            "--no_lighting_preprocess."
        )
    if args.st_time < 0:
        raise ValueError("--st_time must be greater than or equal to 0.")
    if args.ed_time is not None and args.ed_time <= args.st_time:
        raise ValueError("--ed_time must be greater than --st_time when provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_passenger_clips(output_dir)

    openai_client = None
    if clip_backend == "openai":
        openai_client = OpenAI(api_key=get_openai_api_key(args.api_key))

    local_runtimes: Dict[str, Tuple[Any, Any, str]] = {}

    def build_runtime(model_name: str, backend: str) -> Dict[str, Any]:
        runtime = {
            "model_name": model_name,
            "backend": backend,
            "model_family": "openai",
            "model": None,
            "processor": None,
            "openai_client": openai_client if backend == "openai" else None,
        }
        if backend == "local":
            if model_name not in local_runtimes:
                model_family = resolve_local_model_family(model_name)
                loaded_model, loaded_processor = load_model_and_processor(
                    model_name,
                    prefer_4bit=not args.no_4bit,
                )
                local_runtimes[model_name] = (
                    loaded_model,
                    loaded_processor,
                    model_family,
                )
            (
                runtime["model"],
                runtime["processor"],
                runtime["model_family"],
            ) = local_runtimes[model_name]
        return runtime

    if needs_vlm_runtime:
        clip_runtime = build_runtime(clip_model_name, clip_backend)
        print(
            "Using clip model "
            f"backend={clip_runtime['backend']}, "
            f"family={clip_runtime['model_family']}, "
            f"model={clip_runtime['model_name']}"
        )
    else:
        clip_runtime = {
            "model_name": "",
            "backend": "none",
            "model_family": "none",
            "model": None,
            "processor": None,
            "openai_client": None,
        }
        print("Using YOLO door tracker; VLM door status model is not loaded.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0
    print(
        f"Streaming {video_path} "
        f"({duration:.2f}s, {fps:.2f} FPS, {total_frames} frames)"
    )
    if args.st_time >= duration:
        raise ValueError(
            f"--st_time {args.st_time:.2f}s is beyond video duration {duration:.2f}s."
        )
    processing_end_time = duration
    if args.ed_time is not None:
        if args.ed_time > duration:
            raise ValueError(
                f"--ed_time {args.ed_time:.2f}s is beyond video duration {duration:.2f}s."
            )
        processing_end_time = args.ed_time
    print(
        f"Processing interval: {args.st_time:.2f}s-{processing_end_time:.2f}s "
        f"({processing_end_time - args.st_time:.2f}s)"
    )
    print(
        "Passenger lighting preprocessing: "
        f"{'enabled' if not args.no_lighting_preprocess else 'disabled'}"
    )
    print(f"Passenger tracker: {args.track}")
    print(f"Door tracker: {args.door_track}")

    results: List[Dict[str, Any]] = []
    flow_events: List[Dict[str, Any]] = []
    processed_stops: List[Dict[str, Any]] = []

    def write_flow_events() -> None:
        flow_path = output_dir / "flow_events.json"
        with flow_path.open("w", encoding="utf-8") as f:
            json.dump(flow_events, f, indent=2, ensure_ascii=False)

    def process_stop(stop_id: int, stop: Dict[str, Any]) -> None:
        stop_start = stop["start_time"]
        stop_end = stop["end_time"]
        print(f"Processing stop #{stop_id}: {stop_start:.2f}s-{stop_end:.2f}s")

        clips, split_event = split_payment_activated_interval(
            cap=cap,
            stop_id=stop_id,
            stop_start=stop_start,
            stop_end=stop_end,
            args=args,
        )
        saved_clips = clips
        if not saved_clips:
            print(
                f"  Stop #{stop_id}: detected 0 passenger clip(s); "
                "skipping stop record."
            )
            write_flow_events()
            return

        split_event["saved_clip_count"] = len(saved_clips)
        split_event["saved_clips"] = saved_clips
        split_event["saved_clip_policy"] = "save_all_rule_based_clips"
        flow_events.append(split_event)
        flow_events.append(
            {
                "type": "stop",
                "stop_id": stop_id,
                "start_time": stop_start,
                "end_time": stop_end,
                "payment_activated_interval": {
                    "start_time": stop_start,
                    "end_time": stop_end,
                    "duration": stop_end - stop_start,
                },
                "open_pin": stop.get("open_pin"),
                "close_pin": stop.get("close_pin"),
                "end_reason": stop.get("end_reason"),
                "passenger_split": split_event,
                "candidate_clips": clips,
                "clips": saved_clips,
            }
        )
        processed_stops.append(stop)
        print(
            f"  Stop #{stop_id}: saved {len(saved_clips)} rule-based "
            f"passenger clip(s)"
        )

        for local_clip_index, clip in enumerate(saved_clips, start=1):
            global_clip_index = len(results) + 1
            clip_path = (
                output_dir
                / "passenger_clips"
                / f"stop_{stop_id:04d}_passenger_{local_clip_index:02d}_"
                f"{clip['start_time']:.2f}_{clip['end_time']:.2f}.mp4"
            )
            clip_error = save_video_clip(
                cap=cap,
                start_time=clip["start_time"],
                end_time=clip["end_time"],
                output_path=clip_path,
                lighting_preprocess=not args.no_lighting_preprocess,
            )
            if clip_error is None:
                clip["clip_path"] = str(clip_path)
                print(f"  Saved passenger clip: {clip_path}")
            else:
                clip["clip_save_error"] = clip_error
                print(f"  Could not save passenger clip: {clip_error}")

            result = {
                "video_name": f"clip_{global_clip_index:04d}",
                "video_path": str(video_path),
                "clip_id": global_clip_index,
                "stop_id": stop_id,
                "stop_start_time": stop_start,
                "stop_end_time": stop_end,
                "open_pin": stop.get("open_pin"),
                "close_pin": stop.get("close_pin"),
                "passenger_index": clip.get("passenger_index"),
                "main_passenger_id": clip.get("main_passenger_id"),
                "passenger_activity": None,
                "start_frame_index": clip.get("start_frame_index"),
                "end_frame_index": clip.get("end_frame_index"),
                "clip_start_time": clip["start_time"],
                "clip_end_time": clip["end_time"],
                "clip_duration": clip["end_time"] - clip["start_time"],
                "passenger_split_points": split_event.get("split_points", []),
                "saved_clip_path": clip.get("clip_path"),
                "passenger_sheet_path": clip.get("passenger_sheet_path"),
                "passenger_sheet_format": clip.get("passenger_sheet_format"),
                "passenger_sheet_frame_indices": clip.get(
                    "passenger_sheet_frame_indices"
                ),
                "passenger_sheet_error": clip.get("passenger_sheet_error"),
                "clip_save_error": clip.get("clip_save_error"),
            }
            results.append(result)
            if clip_callback is not None:
                clip_callback(result, split_event)
            write_results(results, output_dir)
        write_flow_events()

    door_roi_bbox = detect_front_door_roi(
        cap=cap,
        frame_time=args.st_time,
        output_dir=output_dir,
        args=args,
        backend=clip_runtime["backend"],
        model_family=clip_runtime["model_family"],
        model_name=clip_runtime["model_name"],
        model=clip_runtime["model"],
        processor=clip_runtime["processor"],
        openai_client=clip_runtime["openai_client"],
        flow_events=flow_events,
    )
    write_flow_events()

    try:
        raw_pins: List[Dict[str, Any]] = []
        door_pins: List[Dict[str, Any]] = []
        active_open_pin: Optional[Dict[str, Any]] = None
        stop_id = 0

        def accept_pin(accepted_pin: Dict[str, Any]) -> None:
            nonlocal active_open_pin
            nonlocal stop_id
            if (
                door_pins
                and accepted_pin["transition"] == door_pins[-1]["transition"]
                and abs(accepted_pin["change_time"] - door_pins[-1]["change_time"])
                <= args.door_change_merge_tolerance
            ):
                if accepted_pin.get("confidence", 0.0) > door_pins[-1].get("confidence", 0.0):
                    door_pins[-1] = accepted_pin
                    if accepted_pin["transition"] == "closed_to_open":
                        active_open_pin = accepted_pin
                return

            door_pins.append(accepted_pin)
            if accepted_pin["transition"] == "closed_to_open":
                active_open_pin = accepted_pin
            elif (
                accepted_pin["transition"] == "open_to_closed"
                and active_open_pin is not None
                and accepted_pin["change_time"] > active_open_pin["change_time"]
            ):
                stop_id += 1
                process_stop(
                    stop_id,
                    {
                        "start_time": active_open_pin["change_time"],
                        "end_time": accepted_pin["change_time"],
                        "open_pin": active_open_pin,
                        "close_pin": accepted_pin,
                    },
                )
                active_open_pin = None

        if args.door_track == "yolo":
            raw_pins = scan_door_changes_yolo(
                cap=cap,
                door_roi_bbox=door_roi_bbox,
                args=args,
                video_duration=duration,
                processing_end_time=processing_end_time,
                flow_events=flow_events,
                on_pin=accept_pin,
            )
        else:
            step = args.door_change_window - args.door_change_overlap
            if step <= 0:
                raise ValueError(
                    "--door_change_overlap must be smaller than --door_change_window."
                )

            window_start = args.st_time
            while window_start < processing_end_time:
                window_duration = min(
                    args.door_change_window,
                    processing_end_time - window_start,
                )
                if window_duration <= 0:
                    break

                pin, _ = detect_door_change_window(
                    cap=cap,
                    window_start=window_start,
                    window_duration=window_duration,
                    door_roi_bbox=door_roi_bbox,
                    args=args,
                    backend=clip_runtime["backend"],
                    model_family=clip_runtime["model_family"],
                    model_name=clip_runtime["model_name"],
                    model=clip_runtime["model"],
                    processor=clip_runtime["processor"],
                    openai_client=clip_runtime["openai_client"],
                    flow_events=flow_events,
                )
                if pin is None:
                    window_start += step
                    continue

                raw_pins.append(pin.copy())
                verified, verification_event = verify_door_change_pin(
                    cap=cap,
                    candidate_pin=pin,
                    video_duration=processing_end_time,
                    door_roi_bbox=door_roi_bbox,
                    args=args,
                    backend=clip_runtime["backend"],
                    model_family=clip_runtime["model_family"],
                    model_name=clip_runtime["model_name"],
                    model=clip_runtime["model"],
                    processor=clip_runtime["processor"],
                    openai_client=clip_runtime["openai_client"],
                    flow_events=flow_events,
                )
                if not verified:
                    window_start += step
                    continue

                accepted_pin = pin.copy()
                accepted_pin["verification"] = verification_event
                accept_pin(accepted_pin)
                window_start += step

        if (
            active_open_pin is not None
            and processing_end_time > active_open_pin["change_time"]
        ):
            stop_id += 1
            process_stop(
                stop_id,
                {
                    "start_time": active_open_pin["change_time"],
                    "end_time": processing_end_time,
                    "open_pin": active_open_pin,
                    "close_pin": None,
                    "end_reason": "processing_end"
                    if args.ed_time is not None
                    else "video_end",
                },
            )

        flow_events.append(
            {
                "type": "door_change_summary",
                "door_tracker": args.door_track,
                "raw_pin_count": len(raw_pins),
                "accepted_pin_count": len(door_pins),
                "verified_pin_count": len(door_pins)
                if args.door_track == "vlm"
                else None,
                "processing_start_time": args.st_time,
                "processing_end_time": processing_end_time,
                "merged_pins": door_pins,
                "stop_intervals": processed_stops,
            }
        )
        print(
            f"Door change scan found {len(raw_pins)} raw pin(s), "
            f"{len(door_pins)} accepted merged pin(s), "
            f"{len(processed_stops)} stop interval(s)."
        )
        write_flow_events()
    finally:
        cap.release()
        cleanup_memory()

    write_results(results, output_dir)
    flow_path = output_dir / "flow_events.json"
    with flow_path.open("w", encoding="utf-8") as f:
        json.dump(flow_events, f, indent=2, ensure_ascii=False)

    return results


def main() -> int:
    args = parse_args()
    try:
        results = process_streaming_video(args)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    print(f"Done. Wrote {len(results)} passenger clip record(s) to:")
    print(f"  {output_dir / 'results.json'}")
    print(f"  {output_dir / 'results.csv'}")
    if (output_dir / "flow_events.json").exists():
        print(f"  {output_dir / 'flow_events.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
