from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

from ..base.selection import FrameSelectionResult, PatchSelectionResult
from .cilp_model import CLIPTextModel, CLIPVisionModel_v2


SUPPORTED_QWEN_BACKENDS = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
SUPPORTED_QUERY_AGGREGATIONS = {"max", "mean"}
SUPPORTED_SCORE_POOLING = {"naive_mean", "sliding_window"}
CLIP_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_device(
    device: str | torch.device | None,
    *,
    reference_tensor: torch.Tensor,
) -> torch.device:
    if device is None:
        return reference_tensor.device
    return torch.device(device)


def _resolve_model_device(model: Any) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _resolve_device_key(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return str(device)


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


def _resolve_query_aggregation(aggregation: str) -> str:
    normalized = str(aggregation).strip().lower()
    if normalized not in SUPPORTED_QUERY_AGGREGATIONS:
        available = ", ".join(sorted(SUPPORTED_QUERY_AGGREGATIONS))
        raise ValueError(
            f"Unsupported `aggregation`: {aggregation}. Available: {available}."
        )
    return normalized


def _resolve_score_pooling(score_pooling: str) -> str:
    normalized = str(score_pooling).strip().lower()
    if normalized == "merge":
        normalized = "naive_mean"
    if normalized not in SUPPORTED_SCORE_POOLING:
        available = ", ".join(sorted(SUPPORTED_SCORE_POOLING))
        raise ValueError(
            f"Unsupported `score_pooling`: {score_pooling}. Available: {available}."
        )
    return normalized


def _resolve_spatial_merge_size(
    model: Any,
    *,
    default: int,
) -> int:
    candidate_paths = (
        ("model", "visual", "spatial_merge_size"),
        ("visual", "spatial_merge_size"),
        ("config", "vision_config", "spatial_merge_size"),
    )
    for path in candidate_paths:
        current = model
        found = True
        for attr in path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and current is not None:
            value = int(current)
            if value > 0:
                return value
    return int(default)


def _resolve_temporal_patch_size(
    model: Any,
    *,
    frame_count: int,
    grid_t: int,
    default: int,
) -> int:
    candidate_paths = (
        ("model", "visual", "temporal_patch_size"),
        ("visual", "temporal_patch_size"),
        ("config", "vision_config", "temporal_patch_size"),
        ("config", "temporal_patch_size"),
    )
    for path in candidate_paths:
        current = model
        found = True
        for attr in path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and current is not None:
            value = int(current)
            if value > 0:
                return value

    if grid_t > 0 and frame_count % grid_t == 0:
        inferred = frame_count // grid_t
        if inferred > 0:
            return inferred
    return int(default)


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
    return _load_queries_cached(str(path.resolve()), stat.st_mtime_ns)


def _coerce_video_frames(frame_selection: FrameSelectionResult) -> torch.Tensor:
    frames = frame_selection.frames
    if frames is None:
        raise ValueError("Frame selection did not provide video frames.")
    if not torch.is_tensor(frames):
        raise TypeError("Frame selection frames must be a torch.Tensor.")
    if frames.ndim != 4:
        raise ValueError(
            "Expected sampled video frames with shape (T, H, W, C), "
            f"but got {tuple(frames.shape)}."
        )
    return frames


def _extract_video_grid(
    *,
    model_inputs: dict[str, Any],
    extraction_metadata: dict[str, Any],
) -> tuple[int, int, int]:
    video_grid_thw = model_inputs.get("video_grid_thw")
    if torch.is_tensor(video_grid_thw):
        if video_grid_thw.ndim == 2:
            if video_grid_thw.shape[0] != 1:
                raise ValueError(
                    "TRIPS patch selection currently expects a single video per prompt."
                )
            row = video_grid_thw[0]
        elif video_grid_thw.ndim == 1 and video_grid_thw.numel() == 3:
            row = video_grid_thw
        else:
            raise ValueError(
                "Expected `video_grid_thw` to have shape (1, 3) or (3,), "
                f"but got {tuple(video_grid_thw.shape)}."
            )
        return int(row[0]), int(row[1]), int(row[2])

    metadata_grid = extraction_metadata.get("video_grid_thw")
    if isinstance(metadata_grid, (list, tuple)) and len(metadata_grid) == 3:
        return int(metadata_grid[0]), int(metadata_grid[1]), int(metadata_grid[2])

    raise ValueError("`video_grid_thw` is required for TRIPS patch selection.")


def _prepare_frame_arrays(frames: torch.Tensor) -> list[np.ndarray]:
    frames_cpu = frames.detach().cpu()
    return [frame.numpy() for frame in frames_cpu]


@lru_cache(maxsize=4)
def _load_trips_components(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
) -> tuple[CLIPImageProcessor, Any, CLIPVisionModel_v2, CLIPTextModel]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel_v2(clip_model_name)
    text_model = CLIPTextModel(clip_model_name)
    if clip_dtype is None:
        vision_model = vision_model.to(device_key)
        text_model = text_model.to(device_key)
    else:
        vision_model = vision_model.to(device=device_key, dtype=clip_dtype)
        text_model = text_model.to(device=device_key, dtype=clip_dtype)
    vision_model.eval()
    text_model.eval()
    return image_processor, tokenizer, vision_model, text_model


def preload_trips_patch_selection(
    *,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    model: Any,
    **_: Any,
) -> None:
    preload_device = (
        torch.device(device) if device is not None else _resolve_model_device(model)
    )
    device_key = _resolve_device_key(preload_device)
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    _load_trips_components(clip_model_name, device_key, clip_dtype_key)


def _encode_text_queries(
    *,
    queries: tuple[str, ...],
    tokenizer: Any,
    text_model: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        text_inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        return F.normalize(text_model(**text_inputs), dim=-1)


def _aggregate_query_scores(
    scores: torch.Tensor,
    *,
    aggregation: str,
) -> torch.Tensor:
    if aggregation == "max":
        return scores.max(dim=-1).values
    return scores.mean(dim=-1)


def _compute_clip_score_maps(
    *,
    frame_arrays: list[np.ndarray],
    image_processor: CLIPImageProcessor,
    vision_model: CLIPVisionModel_v2,
    text_embeddings_t: torch.Tensor,
    aggregation: str,
    batch_size: int,
    device: torch.device,
    clip_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    score_chunks: list[torch.Tensor] = []
    clip_grid: tuple[int, int] | None = None
    patch_size = int(vision_model.config.patch_size)

    with torch.inference_mode():
        for start in range(0, len(frame_arrays), batch_size):
            batch_frames = frame_arrays[start : start + batch_size]
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

            patch_embeddings, _ = vision_model(pixel_values)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            patch_scores = patch_embeddings @ text_embeddings_t
            patch_scores = _aggregate_query_scores(
                patch_scores,
                aggregation=aggregation,
            )

            clip_h = int(pixel_values.shape[-2] // patch_size)
            clip_w = int(pixel_values.shape[-1] // patch_size)
            if clip_h * clip_w != int(patch_scores.shape[1]):
                raise ValueError(
                    "Failed to infer CLIP patch grid from processor output: "
                    f"expected {clip_h}x{clip_w}={clip_h * clip_w} patches, "
                    f"but got {int(patch_scores.shape[1])}."
                )

            if clip_grid is None:
                clip_grid = (clip_h, clip_w)
            elif clip_grid != (clip_h, clip_w):
                raise ValueError(
                    "CLIP processor returned inconsistent patch grids across batches: "
                    f"{clip_grid} vs {(clip_h, clip_w)}."
                )

            score_chunks.append(
                patch_scores.view(patch_scores.shape[0], clip_h, clip_w)
            )

    if clip_grid is None:
        raise ValueError("No frames were provided for TRIPS scoring.")

    return torch.cat(score_chunks, dim=0), clip_grid


def _resize_score_maps(
    score_maps: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    resized = F.interpolate(
        score_maps.unsqueeze(1),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1)


def _align_temporal_score_maps(
    score_maps: torch.Tensor,
    *,
    grid_t: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    if temporal_patch_size <= 1:
        if int(score_maps.shape[0]) != int(grid_t):
            raise ValueError(
                "Temporal score map count does not match video grid: "
                f"maps={int(score_maps.shape[0])}, grid_t={grid_t}."
            )
        return score_maps

    expected_frames = int(grid_t * temporal_patch_size)
    if int(score_maps.shape[0]) != expected_frames:
        raise ValueError(
            "Temporal score map count does not match the inferred temporal patch size: "
            f"maps={int(score_maps.shape[0])}, expected={expected_frames}."
        )
    return score_maps.view(
        grid_t,
        temporal_patch_size,
        score_maps.shape[1],
        score_maps.shape[2],
    ).mean(dim=1)


def _pool_scores_naive_mean(
    score_maps: torch.Tensor,
    *,
    merge_size: int,
) -> torch.Tensor:
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")

    _, raw_height, raw_width = score_maps.shape
    if raw_height % merge_size != 0 or raw_width % merge_size != 0:
        raise ValueError(
            "Raw Qwen spatial grid must be divisible by spatial_merge_size: "
            f"grid=({raw_height}, {raw_width}), merge_size={merge_size}."
        )

    pooled = F.avg_pool2d(
        score_maps.unsqueeze(1),
        kernel_size=merge_size,
        stride=merge_size,
    )
    return pooled.squeeze(1)


def _compute_window_score_maps(
    score_maps: torch.Tensor,
    *,
    window_size: int,
    window_stride: int,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError(f"`window_size` must be positive, got {window_size}.")
    if window_stride <= 0:
        raise ValueError(f"`window_stride` must be positive, got {window_stride}.")

    unfolded = F.unfold(
        score_maps.unsqueeze(1),
        kernel_size=window_size,
        stride=window_stride,
    )
    patch_count = window_size * window_size
    return unfolded.view(score_maps.shape[0], patch_count, -1).mean(dim=1)


def _pool_scores_sliding_window(
    score_maps: torch.Tensor,
    *,
    raw_height: int,
    raw_width: int,
    merge_size: int,
    window_size: int,
    window_stride: int,
) -> torch.Tensor:
    if raw_height % merge_size != 0 or raw_width % merge_size != 0:
        raise ValueError(
            "Raw Qwen spatial grid must be divisible by spatial_merge_size: "
            f"grid=({raw_height}, {raw_width}), merge_size={merge_size}."
        )

    window_scores = _compute_window_score_maps(
        score_maps,
        window_size=window_size,
        window_stride=window_stride,
    )
    window_h = (raw_height - window_size) // window_stride + 1
    window_w = (raw_width - window_size) // window_stride + 1
    if window_h <= 0 or window_w <= 0:
        raise ValueError(
            "Sliding-window setup does not fit the token grid: "
            f"raw_grid=({raw_height}, {raw_width}), "
            f"window_size={window_size}, window_stride={window_stride}."
        )

    window_scores = window_scores.view(score_maps.shape[0], window_h, window_w)
    sum_scores = F.avg_pool2d(
        window_scores.unsqueeze(1),
        kernel_size=merge_size,
        stride=merge_size,
    ).squeeze(1)
    coverage = F.avg_pool2d(
        torch.ones_like(window_scores).unsqueeze(1),
        kernel_size=merge_size,
        stride=merge_size,
    ).squeeze(1)
    if torch.any(coverage <= 0):
        raise ValueError("Encountered a merged token without sliding-window coverage.")
    return sum_scores / coverage


def _resolve_attentive_budget(
    *,
    token_count: int,
    keep_ratio: float,
    attentive_budget: int | None,
    max_attentive_budget: int | None,
) -> int:
    if attentive_budget is not None:
        resolved_budget = int(attentive_budget)
        if resolved_budget <= 0:
            raise ValueError(
                f"`attentive_budget` must be positive when provided, got {attentive_budget}."
            )
    else:
        if keep_ratio <= 0.0 or keep_ratio >= 1.0:
            raise ValueError(f"`keep_ratio` must be in (0, 1), got {keep_ratio}.")
        resolved_budget = max(1, int(math.ceil(token_count * keep_ratio)))

    if max_attentive_budget is not None:
        max_budget = int(max_attentive_budget)
        if max_budget <= 0:
            raise ValueError(
                "`max_attentive_budget` must be positive when provided, "
                f"got {max_attentive_budget}."
            )
        resolved_budget = min(resolved_budget, max_budget)

    if resolved_budget >= token_count:
        resolved_budget = token_count - 1
    if resolved_budget <= 0:
        raise ValueError(
            "TRIPS needs at least one attentive token and one inattentive token to fuse. "
            f"Got token_count={token_count}, resolved_budget={resolved_budget}."
        )
    return resolved_budget


def _build_fused_token(
    *,
    inattentive_features: torch.Tensor,
    inattentive_scores: torch.Tensor,
    fuse_strategy: str,
) -> torch.Tensor:
    strategy = str(fuse_strategy).strip().lower()
    if strategy == "mean":
        return inattentive_features.mean(dim=0, keepdim=True)
    if strategy == "score_weighted_mean":
        weights = torch.softmax(inattentive_scores.float(), dim=0).to(
            device=inattentive_features.device,
            dtype=inattentive_features.dtype,
        )
        return (inattentive_features * weights.unsqueeze(-1)).sum(
            dim=0,
            keepdim=True,
        )
    raise ValueError(
        f"Unsupported `fuse_strategy`: {fuse_strategy}. "
        "Available: mean, score_weighted_mean."
    )


def _resolve_fuse_scope(fuse_scope: str) -> str:
    normalized = str(fuse_scope).strip().lower()
    if normalized not in {"global", "framewise"}:
        raise ValueError(
            f"Unsupported `fuse_scope`: {fuse_scope}. Available: global, framewise."
        )
    return normalized


def _resolve_two_stage_thresholds(
    *,
    low_threshold: float | None,
    high_threshold: float | None,
) -> tuple[float | None, float | None]:
    if low_threshold is None and high_threshold is None:
        return None, None
    if low_threshold is None or high_threshold is None:
        raise ValueError(
            "Set both `patch_score_threshold_low` and `patch_score_threshold_high`, "
            "or leave both unset."
        )

    resolved_low = float(low_threshold)
    resolved_high = float(high_threshold)
    if resolved_high < resolved_low:
        raise ValueError(
            "`patch_score_threshold_high` must be greater than or equal to "
            "`patch_score_threshold_low`."
        )
    return resolved_low, resolved_high


def trips_patch_selection(
    video_features: torch.Tensor,
    *,
    frame_selection: FrameSelectionResult,
    model_inputs: dict[str, Any],
    extraction_metadata: dict[str, Any],
    model: Any,
    backend: str,
    query_file: str,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    keep_ratio: float = 0.25,
    attentive_budget: int | None = None,
    max_attentive_budget: int | None = None,
    patch_score_threshold: float | None = None,
    patch_score_threshold_low: float | None = None,
    patch_score_threshold_high: float | None = None,
    aggregation: str = "max",
    score_pooling: str = "naive_mean",
    window_size: int = 4,
    window_stride: int = 2,
    batch_size: int = 8,
    spatial_merge_size: int | None = None,
    fuse_strategy: str = "score_weighted_mean",
    fuse_scope: str = "global",
    device: str | torch.device | None = None,
    **_: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "TRIPS patch selection currently supports only Qwen backends, "
            f"got `{backend}`."
        )

    if video_features.ndim != 2:
        raise ValueError(
            "Expected extracted video features with shape (N, D), "
            f"but got {tuple(video_features.shape)}."
        )

    resolved_aggregation = _resolve_query_aggregation(aggregation)
    resolved_score_pooling = _resolve_score_pooling(score_pooling)
    resolved_fuse_scope = _resolve_fuse_scope(fuse_scope)
    resolved_low_threshold, resolved_high_threshold = _resolve_two_stage_thresholds(
        low_threshold=patch_score_threshold_low,
        high_threshold=patch_score_threshold_high,
    )
    frames = _coerce_video_frames(frame_selection)
    frame_count = int(frames.shape[0])
    grid_t, raw_grid_h, raw_grid_w = _extract_video_grid(
        model_inputs=model_inputs,
        extraction_metadata=extraction_metadata,
    )
    merge_size = int(
        spatial_merge_size
        if spatial_merge_size is not None
        else _resolve_spatial_merge_size(model, default=2)
    )
    temporal_patch_size = _resolve_temporal_patch_size(
        model,
        frame_count=frame_count,
        grid_t=grid_t,
        default=1,
    )

    selector_device = _resolve_device(device, reference_tensor=video_features)
    selector_device_key = _resolve_device_key(selector_device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    queries = _load_queries(query_file)
    image_processor, tokenizer, vision_model, text_model = _load_trips_components(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
    )
    text_embeddings = _encode_text_queries(
        queries=queries,
        tokenizer=tokenizer,
        text_model=text_model,
        device=torch.device(selector_device_key),
    )
    frame_arrays = _prepare_frame_arrays(frames)
    clip_score_maps, clip_grid = _compute_clip_score_maps(
        frame_arrays=frame_arrays,
        image_processor=image_processor,
        vision_model=vision_model,
        text_embeddings_t=text_embeddings.transpose(0, 1).contiguous(),
        aggregation=resolved_aggregation,
        batch_size=batch_size,
        device=selector_device,
        clip_dtype=resolved_clip_dtype,
    )

    raw_score_maps = _resize_score_maps(
        clip_score_maps,
        target_height=raw_grid_h,
        target_width=raw_grid_w,
    )
    aligned_score_maps = _align_temporal_score_maps(
        raw_score_maps,
        grid_t=grid_t,
        temporal_patch_size=temporal_patch_size,
    )

    if resolved_score_pooling == "naive_mean":
        token_scores = _pool_scores_naive_mean(
            aligned_score_maps,
            merge_size=merge_size,
        )
    else:
        token_scores = _pool_scores_sliding_window(
            aligned_score_maps,
            raw_height=raw_grid_h,
            raw_width=raw_grid_w,
            merge_size=merge_size,
            window_size=window_size,
            window_stride=window_stride,
        )

    flat_scores = token_scores.reshape(-1)
    expected_tokens = int(video_features.shape[0])
    if int(flat_scores.numel()) != expected_tokens:
        raise ValueError(
            "TRIPS score count does not match extracted video feature count: "
            f"scores={int(flat_scores.numel())}, features={expected_tokens}."
        )
    if expected_tokens < 2:
        raise ValueError(
            "TRIPS fusion requires at least two video tokens, "
            f"but got {expected_tokens}."
        )

    resolved_attentive_budget = _resolve_attentive_budget(
        token_count=expected_tokens,
        keep_ratio=keep_ratio,
        attentive_budget=attentive_budget,
        max_attentive_budget=max_attentive_budget,
    )
    effective_attentive_budget = resolved_attentive_budget
    if resolved_low_threshold is not None and resolved_high_threshold is not None:
        high_conf_mask = flat_scores >= resolved_high_threshold
        eligible_mask = flat_scores >= resolved_low_threshold
        high_conf_indices = torch.nonzero(high_conf_mask, as_tuple=False).flatten()
        eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        eligible_token_count = int(eligible_indices.numel())
        high_conf_token_count = int(high_conf_indices.numel())

        if high_conf_token_count >= effective_attentive_budget:
            high_conf_scores = flat_scores[high_conf_indices]
            attentive_indices = high_conf_indices[
                torch.topk(
                    high_conf_scores,
                    k=effective_attentive_budget,
                    dim=0,
                    largest=True,
                    sorted=False,
                ).indices
            ]
        else:
            remaining_budget = effective_attentive_budget - high_conf_token_count
            mid_conf_mask = eligible_mask & ~high_conf_mask
            mid_conf_indices = torch.nonzero(mid_conf_mask, as_tuple=False).flatten()
            mid_conf_take = min(remaining_budget, int(mid_conf_indices.numel()))
            if mid_conf_take > 0:
                mid_conf_scores = flat_scores[mid_conf_indices]
                selected_mid_indices = mid_conf_indices[
                    torch.topk(
                        mid_conf_scores,
                        k=mid_conf_take,
                        dim=0,
                        largest=True,
                        sorted=False,
                    ).indices
                ]
                attentive_indices = torch.cat(
                    [high_conf_indices, selected_mid_indices],
                    dim=0,
                )
            else:
                attentive_indices = high_conf_indices

            effective_attentive_budget = int(attentive_indices.numel())
    else:
        eligible_mask = torch.ones_like(flat_scores, dtype=torch.bool)
        if patch_score_threshold is not None:
            eligible_mask = flat_scores >= float(patch_score_threshold)

        eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        eligible_token_count = int(eligible_indices.numel())
        effective_attentive_budget = min(resolved_attentive_budget, eligible_token_count)

        if effective_attentive_budget > 0:
            eligible_scores = flat_scores[eligible_indices]
            attentive_indices = eligible_indices[
                torch.topk(
                    eligible_scores,
                    k=effective_attentive_budget,
                    dim=0,
                    largest=True,
                    sorted=False,
                ).indices
            ]
        else:
            attentive_indices = torch.empty(
                0,
                device=flat_scores.device,
                dtype=torch.long,
            )

    attentive_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    attentive_mask[attentive_indices] = True
    inattentive_mask = ~attentive_mask
    inattentive_indices = torch.nonzero(inattentive_mask, as_tuple=False).flatten()
    if int(inattentive_indices.numel()) == 0:
        raise RuntimeError("TRIPS expected inattentive tokens to fuse, but none remained.")

    fused_anchor_indices: list[int] = []
    fused_feature_map: dict[int, torch.Tensor] = {}
    framewise_fused_metadata: list[dict[str, Any]] = []
    if resolved_fuse_scope == "global":
        fused_anchor_index = int(inattentive_indices[0].item())
        fused_feature_map[fused_anchor_index] = _build_fused_token(
            inattentive_features=video_features[
                inattentive_indices.to(video_features.device)
            ],
            inattentive_scores=flat_scores[inattentive_indices].to(video_features.device),
            fuse_strategy=fuse_strategy,
        )
        fused_anchor_indices.append(fused_anchor_index)
    else:
        tokens_per_frame = (raw_grid_h // merge_size) * (raw_grid_w // merge_size)
        for frame_idx in range(grid_t):
            frame_start = frame_idx * tokens_per_frame
            frame_end = frame_start + tokens_per_frame
            frame_mask = (inattentive_indices >= frame_start) & (inattentive_indices < frame_end)
            frame_inattentive_indices = inattentive_indices[frame_mask]
            if int(frame_inattentive_indices.numel()) == 0:
                continue

            fused_anchor_index = int(frame_inattentive_indices[0].item())
            fused_feature_map[fused_anchor_index] = _build_fused_token(
                inattentive_features=video_features[
                    frame_inattentive_indices.to(video_features.device)
                ],
                inattentive_scores=flat_scores[frame_inattentive_indices].to(
                    video_features.device
                ),
                fuse_strategy=fuse_strategy,
            )
            fused_anchor_indices.append(fused_anchor_index)
            framewise_fused_metadata.append(
                {
                    "frame_index": int(frame_idx),
                    "fused_anchor_index": fused_anchor_index,
                    "inattentive_token_count": int(frame_inattentive_indices.numel()),
                }
            )

    selected_indices = torch.cat(
        [
            attentive_indices.to(video_features.device, dtype=torch.long),
            torch.tensor(
                fused_anchor_indices,
                device=video_features.device,
                dtype=torch.long,
            ),
        ],
        dim=0,
    )
    sort_order = torch.argsort(selected_indices)
    selected_indices = selected_indices[sort_order]

    feature_rows: list[torch.Tensor] = []
    for index in selected_indices.detach().cpu().tolist():
        if int(index) in fused_feature_map:
            feature_rows.append(fused_feature_map[int(index)])
        else:
            feature_rows.append(video_features[int(index)].unsqueeze(0))
    selected_features = torch.cat(feature_rows, dim=0)

    attentive_scores = flat_scores[attentive_indices].detach().cpu()
    fused_score = flat_scores[inattentive_indices].mean().detach().cpu()
    attentive_score_min = (
        float(attentive_scores.min().item())
        if attentive_scores.numel() > 0
        else None
    )
    attentive_score_max = (
        float(attentive_scores.max().item())
        if attentive_scores.numel() > 0
        else None
    )
    metadata = {
        "selector_type": "trips_patch_selection",
        "selector_variant": "paper_style_keep_and_fuse",
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "aggregation": resolved_aggregation,
        "score_pooling": resolved_score_pooling,
        "keep_ratio": float(keep_ratio),
        "attentive_budget": int(effective_attentive_budget),
        "requested_attentive_budget": int(resolved_attentive_budget),
        "max_attentive_budget": (
            int(max_attentive_budget) if max_attentive_budget is not None else None
        ),
        "patch_score_threshold": (
            float(patch_score_threshold) if patch_score_threshold is not None else None
        ),
        "patch_score_threshold_low": resolved_low_threshold,
        "patch_score_threshold_high": resolved_high_threshold,
        "eligible_token_count": eligible_token_count,
        "fused_token_count": len(fused_anchor_indices),
        "selected_token_count": int(selected_indices.numel()),
        "video_token_count": expected_tokens,
        "inattentive_token_count": int(inattentive_indices.numel()),
        "fuse_strategy": str(fuse_strategy).strip().lower(),
        "fuse_scope": resolved_fuse_scope,
        "spatial_merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "frame_count": frame_count,
        "temporal_grid_t": grid_t,
        "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        "raw_grid_thw": [grid_t, raw_grid_h, raw_grid_w],
        "merged_grid_thw": [
            grid_t,
            raw_grid_h // merge_size,
            raw_grid_w // merge_size,
        ],
        "attentive_indices": [
            int(index)
            for index in attentive_indices.sort().values.detach().cpu().tolist()
        ],
        "inattentive_indices_preview": [
            int(index) for index in inattentive_indices[:16].detach().cpu().tolist()
        ],
        "fused_anchor_indices": [int(index) for index in fused_anchor_indices],
        "attentive_score_min": attentive_score_min,
        "attentive_score_max": attentive_score_max,
        "fused_score_mean": float(fused_score.item()),
    }
    if resolved_fuse_scope == "global":
        metadata["fused_anchor_index"] = fused_anchor_indices[0]
    else:
        metadata["framewise_fused"] = framewise_fused_metadata
    if resolved_score_pooling == "sliding_window":
        metadata["window_size"] = int(window_size)
        metadata["window_stride"] = int(window_stride)

    return PatchSelectionResult(
        selected_indices=selected_indices,
        selected_features=selected_features,
        metadata=metadata,
    )


trips_patch_selection.preload = preload_trips_patch_selection
