from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from model.base.selection import (
    FrameSelectionResult,
    QWEN_VISION_FACTOR,
    _open_video_for_sampling,
    _resize_frame,
)


CLIP_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_device(
    device: str | torch.device | None,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_clip_dtype(
    clip_dtype: str | torch.dtype | None,
) -> tuple[torch.dtype | None, str]:
    if clip_dtype is None:
        return None, "default"
    if isinstance(clip_dtype, torch.dtype):
        return clip_dtype, str(clip_dtype).replace("torch.", "")

    normalized = str(clip_dtype).strip().lower()
    if normalized == "default":
        return None, "default"
    resolved = CLIP_DTYPE_MAP.get(normalized)
    if resolved is None:
        available = ", ".join(sorted(CLIP_DTYPE_MAP))
        raise ValueError(
            f"Unsupported `clip_dtype`: {clip_dtype}. Available: {available}."
        )
    return resolved, normalized


@lru_cache(maxsize=16)
def _load_queries_cached(
    resolved_path: str,
    modified_time_ns: int,
) -> tuple[str, ...]:
    del modified_time_ns
    path = Path(resolved_path)
    queries = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not queries:
        raise ValueError(f"Query file does not contain any non-empty lines: {path}")
    return queries


def _load_queries(query_file: str | Path) -> tuple[str, ...]:
    path = Path(query_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Query file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Query file path must point to a file: {path}")
    stat = path.stat()
    return _load_queries_cached(
        str(path.resolve()),
        stat.st_mtime_ns,
    )


@lru_cache(maxsize=4)
def _load_clip_components(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
) -> tuple[CLIPImageProcessor, Any, Any, Any]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)

    if clip_dtype is None:
        text_model = text_model.to(device_key)
        vision_model = vision_model.to(device_key)
    else:
        text_model = text_model.to(device=device_key, dtype=clip_dtype)
        vision_model = vision_model.to(device=device_key, dtype=clip_dtype)

    text_model.eval()
    vision_model.eval()
    return image_processor, tokenizer, text_model, vision_model


def preload_qframe_lite_selection(
    *,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    **_: Any,
) -> None:
    resolved_device = _resolve_device(device)
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    _load_clip_components(
        clip_model_name,
        str(resolved_device),
        clip_dtype_key,
    )


def _aggregate_query_scores(
    scores: torch.Tensor,
    *,
    aggregation: str,
) -> torch.Tensor:
    normalized = aggregation.strip().lower()
    if normalized == "max":
        return scores.max(dim=-1).values
    if normalized == "mean":
        return scores.mean(dim=-1)
    raise ValueError(
        f"Unsupported query aggregation: {aggregation}. Use `max` or `mean`."
    )


def _sample_candidate_frames(
    video_path: str,
    *,
    candidate_frames: int,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> tuple[list[np.ndarray], list[int], dict[str, Any]]:
    if candidate_frames <= 0:
        raise ValueError(
            f"`candidate_frames` must be positive, got {candidate_frames}."
        )

    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)
    effective_candidates = min(candidate_frames, total_frames)
    if effective_candidates <= 0:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")

    candidate_indices = (
        np.linspace(0, total_frames - 1, effective_candidates)
        .round()
        .astype(int)
        .tolist()
    )
    target_set = set(candidate_indices)
    frame_idx = 0
    sampled_indices: list[int] = []
    frames: list[np.ndarray] = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx in target_set:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = _resize_frame(
                    frame_rgb,
                    max_side=max_side,
                    ensure_qwen_compatibility=ensure_qwen_compatibility,
                    qwen_factor=qwen_factor,
                )
                frames.append(frame_rgb)
                sampled_indices.append(frame_idx)
            frame_idx += 1
    finally:
        cap.release()
        if transcoded_path is not None:
            transcoded_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    if len(frames) != len(candidate_indices):
        raise RuntimeError(
            f"Expected {len(candidate_indices)} candidate frames, but got {len(frames)}."
        )

    base_height, base_width = frames[0].shape[:2]
    normalized_frames: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(
                frame,
                (base_width, base_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(frame)

    metadata = {
        "video_path": video_path,
        "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
        "total_frames": total_frames,
        "fps": fps if fps > 0 else None,
        "candidate_frame_count": len(normalized_frames),
        "candidate_indices": sampled_indices,
        "frame_shape": list(normalized_frames[0].shape),
        "ensure_qwen_compatibility": ensure_qwen_compatibility,
        "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
    }
    return normalized_frames, sampled_indices, metadata


def _encode_queries(
    queries: tuple[str, ...],
    *,
    tokenizer: Any,
    text_model: Any,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        text_inputs = tokenizer(
            list(queries),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        text_features = text_model(**text_inputs).text_embeds
        return F.normalize(text_features, dim=-1)


def _encode_frames(
    frames: list[np.ndarray],
    *,
    image_processor: CLIPImageProcessor,
    vision_model: Any,
    device: torch.device,
    clip_dtype: torch.dtype | None,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            pixel_values = image_processor(
                images=batch_frames,
                return_tensors="pt",
            )["pixel_values"]
            if clip_dtype is None:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            else:
                pixel_values = pixel_values.to(
                    device=device,
                    dtype=clip_dtype,
                    non_blocking=True,
                )
            image_features = vision_model(pixel_values=pixel_values).image_embeds
            embeddings.append(F.normalize(image_features, dim=-1))
    return torch.cat(embeddings, dim=0)


def qframe_lite_selection(
    video_path: str,
    *,
    query_file: str,
    num_frames: int = 8,
    candidate_frames: int = 32,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    aggregation: str = "max",
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> FrameSelectionResult:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")

    frames, candidate_indices, metadata = _sample_candidate_frames(
        video_path,
        candidate_frames=candidate_frames,
        max_side=max_side,
        ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
    )
    queries = _load_queries(query_file)
    resolved_device = _resolve_device(device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    image_processor, tokenizer, text_model, vision_model = _load_clip_components(
        clip_model_name,
        str(resolved_device),
        clip_dtype_key,
    )

    query_embeddings = _encode_queries(
        queries,
        tokenizer=tokenizer,
        text_model=text_model,
        device=resolved_device,
    )
    frame_embeddings = _encode_frames(
        frames,
        image_processor=image_processor,
        vision_model=vision_model,
        device=resolved_device,
        clip_dtype=resolved_clip_dtype,
        batch_size=batch_size,
    )
    frame_query_scores = frame_embeddings @ query_embeddings.transpose(0, 1).contiguous()
    relevance_scores = _aggregate_query_scores(
        frame_query_scores,
        aggregation=aggregation,
    )

    keep_count = min(num_frames, len(candidate_indices))
    selected_candidate_positions = torch.topk(
        relevance_scores,
        k=keep_count,
        largest=True,
        sorted=False,
    ).indices.tolist()
    selected_candidate_positions = sorted(
        selected_candidate_positions,
        key=lambda idx: candidate_indices[idx],
    )

    selected_frames = np.stack(
        [frames[idx] for idx in selected_candidate_positions],
        axis=0,
    )
    selected_indices = [candidate_indices[idx] for idx in selected_candidate_positions]
    selected_scores = [
        float(relevance_scores[idx].detach().cpu().item())
        for idx in selected_candidate_positions
    ]

    return FrameSelectionResult(
        frames=torch.from_numpy(selected_frames),
        metadata={
            **metadata,
            "selector_type": "qframe_lite_selection",
            "query_file": str(Path(query_file).expanduser()),
            "queries": list(queries),
            "query_count": len(queries),
            "aggregation": aggregation,
            "clip_model_name": clip_model_name,
            "clip_dtype": clip_dtype_key,
            "selected_indices": selected_indices,
            "selected_frame_count": len(selected_indices),
            "selected_scores": selected_scores,
            "candidate_scores": [
                float(score)
                for score in relevance_scores.detach().cpu().tolist()
            ],
        },
    )


qframe_lite_selection.preload = preload_qframe_lite_selection
