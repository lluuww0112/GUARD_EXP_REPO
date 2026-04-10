from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

from ...base.selection import FrameSelectionResult, PatchSelectionResult
from .cilp_model import CLIPTextModel, CLIPVisionModel


SUPPORTED_QWEN_BACKENDS = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
SUPPORTED_SELECTION_MODES = {"naive_mean", "sliding_window"}
SELECTION_MODE_ALIASES = {
    "merge": "naive_mean",
}
CLIP_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_device(
    device: str | torch.device | None,
    reference_tensor: torch.Tensor,
) -> torch.device:
    if device is None:
        return reference_tensor.device
    return torch.device(device)


def _resolve_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)

    try:
        first_parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")
    return first_parameter.device


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


def _resolve_selection_mode(selection_mode: str) -> str:
    normalized = str(selection_mode).strip().lower()
    normalized = SELECTION_MODE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_SELECTION_MODES:
        available = ", ".join(sorted(SUPPORTED_SELECTION_MODES))
        raise ValueError(
            f"Unsupported `selection_mode`: {selection_mode}. Available: {available}."
        )
    return normalized


def _resolve_device_key(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return str(device)


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
    return _load_queries_cached(
        str(path.resolve()),
        stat.st_mtime_ns,
    )


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
    model_inputs: dict[str, Any],
    extraction_metadata: dict[str, Any],
) -> tuple[int, int, int]:
    video_grid_thw = model_inputs.get("video_grid_thw")
    if torch.is_tensor(video_grid_thw):
        if video_grid_thw.ndim == 2:
            if video_grid_thw.shape[0] != 1:
                raise ValueError(
                    "Patch selection currently expects a single video per prompt."
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

    raise ValueError("`video_grid_thw` is required for Qwen patch selection.")


@lru_cache(maxsize=4)
def _load_maskclip_components(
    clip_model_name: str,
    device_type: str,
    clip_dtype_key: str,
) -> tuple[CLIPImageProcessor, Any, CLIPVisionModel, CLIPTextModel]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel(clip_model_name)
    text_model = CLIPTextModel(clip_model_name)
    if clip_dtype is None:
        vision_model = vision_model.to(device_type)
        text_model = text_model.to(device_type)
    else:
        vision_model = vision_model.to(device=device_type, dtype=clip_dtype)
        text_model = text_model.to(device=device_type, dtype=clip_dtype)
    vision_model.eval()
    text_model.eval()
    return image_processor, tokenizer, vision_model, text_model


def preload_maskclip_patch_selection(
    *,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    model: Any,
    **_: Any,
) -> None:
    preload_device = (
        torch.device(device)
        if device is not None
        else _resolve_model_device(model)
    )
    device_key = _resolve_device_key(preload_device)
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    _load_maskclip_components(clip_model_name, device_key, clip_dtype_key)


def _encode_text_queries(
    queries: tuple[str, ...],
    *,
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
        text_embeddings = text_model(**text_inputs)
        return F.normalize(text_embeddings, dim=-1)


@lru_cache(maxsize=16)
def _load_text_embeddings(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
    queries: tuple[str, ...],
) -> torch.Tensor:
    _, tokenizer, _, text_model = _load_maskclip_components(
        clip_model_name,
        device_key,
        clip_dtype_key,
    )
    return _encode_text_queries(
        queries,
        tokenizer=tokenizer,
        text_model=text_model,
        device=torch.device(device_key),
    )


def _prepare_frame_arrays(frames: torch.Tensor) -> list[Any]:
    frames_cpu = frames.detach()
    if frames_cpu.device.type != "cpu":
        frames_cpu = frames_cpu.cpu()
    if not frames_cpu.is_contiguous():
        frames_cpu = frames_cpu.contiguous()
    return list(frames_cpu.numpy())


def _resolve_patch_scoring_frames(
    frame_selection: FrameSelectionResult,
) -> tuple[torch.Tensor, dict[str, int] | None]:
    frames = _coerce_video_frames(frame_selection)
    duplication_metadata = frame_selection.metadata.get("frame_duplication")
    if not isinstance(duplication_metadata, dict):
        return frames, None
    if not bool(duplication_metadata.get("applied", False)):
        return frames, None

    duplicate_factor = int(duplication_metadata.get("duplicate_factor", 1) or 1)
    original_num_frames = int(
        duplication_metadata.get("original_num_frames", 0) or 0
    )
    if duplicate_factor <= 1 or original_num_frames <= 0:
        return frames, None
    if int(frames.shape[0]) != original_num_frames * duplicate_factor:
        return frames, None

    return (
        frames[::duplicate_factor],
        {
            "duplicate_factor": duplicate_factor,
            "original_num_frames": original_num_frames,
        },
    )


def _expand_scores_for_frame_duplication(
    scores: torch.Tensor,
    duplication_info: dict[str, int] | None,
) -> torch.Tensor:
    if duplication_info is None:
        return scores
    duplicate_factor = int(duplication_info.get("duplicate_factor", 1))
    if duplicate_factor <= 1:
        return scores
    return torch.repeat_interleave(scores, repeats=duplicate_factor, dim=0)


def _aggregate_query_scores(
    scores: torch.Tensor,
    *,
    aggregation: str,
) -> torch.Tensor:
    aggregation = aggregation.lower().strip()
    if aggregation == "max":
        return scores.max(dim=-1).values
    if aggregation == "mean":
        return scores.mean(dim=-1)
    raise ValueError(
        f"Unsupported query aggregation: {aggregation}. Use `max` or `mean`."
    )


def _compute_dense_patch_score_maps(
    frame_arrays: list[Any],
    *,
    image_processor: CLIPImageProcessor,
    vision_model: CLIPVisionModel,
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

            patch_embeddings = vision_model(pixel_values)
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

            score_chunks.append(patch_scores.view(patch_scores.shape[0], clip_h, clip_w))

    if clip_grid is None:
        raise ValueError("No frames were provided for dense patch scoring.")
    return torch.cat(score_chunks, dim=0), clip_grid


def _align_temporal_score_maps(
    score_maps: torch.Tensor,
    *,
    grid_t: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    frame_count = int(score_maps.shape[0])
    if grid_t <= 0:
        raise ValueError(f"`grid_t` must be positive, got {grid_t}.")
    if temporal_patch_size <= 0:
        raise ValueError(
            f"`temporal_patch_size` must be positive, got {temporal_patch_size}."
        )

    if frame_count == grid_t:
        return score_maps

    expected_frame_count = grid_t * temporal_patch_size
    if frame_count != expected_frame_count:
        raise ValueError(
            "Frame count and Qwen temporal grid are incompatible even after temporal "
            "patch alignment: "
            f"frames={frame_count}, video_grid_thw[0]={grid_t}, "
            f"temporal_patch_size={temporal_patch_size}."
        )

    reshaped = score_maps.view(
        grid_t,
        temporal_patch_size,
        score_maps.shape[1],
        score_maps.shape[2],
    )
    return reshaped.mean(dim=1)


def _resize_score_maps(
    score_maps: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    if score_maps.shape[-2:] == (target_height, target_width):
        return score_maps
    return F.interpolate(
        score_maps.unsqueeze(1),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _compute_qwen_merge_mean_scores(
    raw_score_maps: torch.Tensor,
    *,
    merge_size: int,
) -> torch.Tensor:
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")

    raw_height = int(raw_score_maps.shape[-2])
    raw_width = int(raw_score_maps.shape[-1])
    if raw_height % merge_size != 0 or raw_width % merge_size != 0:
        raise ValueError(
            "Raw Qwen spatial grid must be divisible by spatial_merge_size: "
            f"grid=({raw_height}, {raw_width}), merge_size={merge_size}."
        )

    return F.avg_pool2d(
        raw_score_maps.unsqueeze(1),
        kernel_size=merge_size,
        stride=merge_size,
    ).squeeze(1)


def _compute_window_score_maps(
    raw_score_maps: torch.Tensor,
    *,
    window_size: int,
    window_stride: int,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError(f"`window_size` must be positive, got {window_size}.")
    if window_stride <= 0:
        raise ValueError(f"`window_stride` must be positive, got {window_stride}.")
    if raw_score_maps.shape[-2] < window_size or raw_score_maps.shape[-1] < window_size:
        raise ValueError(
            "Raw patch grid is smaller than the requested sliding window: "
            f"grid={tuple(raw_score_maps.shape[-2:])}, window_size={window_size}."
        )

    return F.avg_pool2d(
        raw_score_maps.unsqueeze(1),
        kernel_size=window_size,
        stride=window_stride,
    ).squeeze(1)


def _compute_sliding_window_merged_scores(
    window_score_maps: torch.Tensor,
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

    window_scores = window_score_maps.unsqueeze(1)
    overlap_kernel = window_scores.new_ones((1, 1, window_size, window_size))
    raw_score_sums = F.conv_transpose2d(
        window_scores,
        overlap_kernel,
        stride=window_stride,
    )
    raw_overlap_counts = F.conv_transpose2d(
        torch.ones_like(window_scores),
        overlap_kernel,
        stride=window_stride,
    )

    output_height = int(raw_score_sums.shape[-2])
    output_width = int(raw_score_sums.shape[-1])
    pad_height = raw_height - output_height
    pad_width = raw_width - output_width
    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            "Sliding-window reconstruction exceeded the expected raw grid size: "
            f"reconstructed=({output_height}, {output_width}), "
            f"expected=({raw_height}, {raw_width})."
        )
    if pad_height > 0 or pad_width > 0:
        padding = (0, pad_width, 0, pad_height)
        raw_score_sums = F.pad(raw_score_sums, padding)
        raw_overlap_counts = F.pad(raw_overlap_counts, padding)

    pooled_score_sums = F.avg_pool2d(
        raw_score_sums,
        kernel_size=merge_size,
        stride=merge_size,
    )
    pooled_overlap_counts = F.avg_pool2d(
        raw_overlap_counts,
        kernel_size=merge_size,
        stride=merge_size,
    )
    if torch.any(pooled_overlap_counts <= 0):
        raise ValueError("Encountered a merged token without any overlapping sliding windows.")

    return (pooled_score_sums / pooled_overlap_counts).squeeze(1)


def _select_topk_per_frame(
    merged_scores: torch.Tensor,
    *,
    keep_ratio: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"`keep_ratio` must be in (0, 1], got {keep_ratio}.")

    merged_h = int(merged_scores.shape[1])
    merged_w = int(merged_scores.shape[2])
    tokens_per_frame = merged_h * merged_w
    keep_per_frame = max(1, int(math.ceil(tokens_per_frame * keep_ratio)))

    frame_scores = merged_scores.reshape(merged_scores.shape[0], -1)
    topk_indices = torch.topk(
        frame_scores,
        k=keep_per_frame,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    selected_per_frame = torch.sort(topk_indices, dim=1).values
    frame_offsets = (
        torch.arange(
            merged_scores.shape[0],
            device=merged_scores.device,
            dtype=selected_per_frame.dtype,
        ).unsqueeze(1)
        * tokens_per_frame
    )
    selected_indices = (selected_per_frame + frame_offsets).reshape(-1)

    min_scores = frame_scores.min(dim=1).values
    max_scores = frame_scores.max(dim=1).values
    frame_metadata = [
        {
            "frame_index": int(frame_idx),
            "tokens_per_frame": tokens_per_frame,
            "keep_count": keep_per_frame,
            "score_min": float(min_scores[frame_idx].item()),
            "score_max": float(max_scores[frame_idx].item()),
        }
        for frame_idx in range(int(merged_scores.shape[0]))
    ]

    return selected_indices, frame_metadata


def maskclip_patch_selection(
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
    keep_ratio: float = 0.5,
    selection_mode: str = "naive_mean",
    window_size: int = 2,
    window_stride: int = 1,
    batch_size: int = 8,
    aggregation: str = "max",
    spatial_merge_size: int | None = None,
    device: str | torch.device | None = None,
    **_: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "MaskCLIP patch selection currently supports only Qwen backends, "
            f"got `{backend}`."
        )

    resolved_selection_mode = _resolve_selection_mode(selection_mode)
    frames = _coerce_video_frames(frame_selection)
    frame_count = int(frames.shape[0])
    scoring_frames, duplication_info = _resolve_patch_scoring_frames(frame_selection)
    grid_t, raw_grid_h, raw_grid_w = _extract_video_grid(
        model_inputs=model_inputs,
        extraction_metadata=extraction_metadata,
    )

    merge_size = int(
        spatial_merge_size
        if spatial_merge_size is not None
        else _resolve_spatial_merge_size(model, default=2)
    )
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")
    temporal_patch_size = _resolve_temporal_patch_size(
        model,
        frame_count=frame_count,
        grid_t=grid_t,
        default=1,
    )

    selector_device = _resolve_device(device, video_features)
    selector_device_key = _resolve_device_key(selector_device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    queries = _load_queries(query_file)
    image_processor, _, vision_model, _ = _load_maskclip_components(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
    )

    text_embeddings = _load_text_embeddings(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
        queries,
    )
    frame_arrays = _prepare_frame_arrays(scoring_frames)
    dense_score_maps, clip_grid = _compute_dense_patch_score_maps(
        frame_arrays,
        image_processor=image_processor,
        vision_model=vision_model,
        text_embeddings_t=text_embeddings.transpose(0, 1).contiguous(),
        aggregation=aggregation,
        batch_size=batch_size,
        device=selector_device,
        clip_dtype=resolved_clip_dtype,
    )
    dense_score_maps = _expand_scores_for_frame_duplication(
        dense_score_maps,
        duplication_info,
    )
    raw_score_maps = _resize_score_maps(
        dense_score_maps,
        target_height=raw_grid_h,
        target_width=raw_grid_w,
    )
    raw_score_maps = _align_temporal_score_maps(
        raw_score_maps,
        grid_t=grid_t,
        temporal_patch_size=temporal_patch_size,
    )

    mode_metadata: dict[str, Any] = {}
    if resolved_selection_mode == "naive_mean":
        merged_scores = _compute_qwen_merge_mean_scores(
            raw_score_maps,
            merge_size=merge_size,
        )
        mode_metadata["score_reduction"] = "mean_within_qwen_merge_unit"
    else:
        window_score_maps = _compute_window_score_maps(
            raw_score_maps,
            window_size=window_size,
            window_stride=window_stride,
        )
        merged_scores = _compute_sliding_window_merged_scores(
            window_score_maps,
            raw_height=raw_grid_h,
            raw_width=raw_grid_w,
            merge_size=merge_size,
            window_size=window_size,
            window_stride=window_stride,
        )
        mode_metadata["window_size"] = int(window_size)
        mode_metadata["window_stride"] = int(window_stride)

    selected_indices, frame_metadata = _select_topk_per_frame(
        merged_scores,
        keep_ratio=keep_ratio,
    )

    expected_tokens = int(video_features.shape[0])
    if int(merged_scores.numel()) != expected_tokens:
        raise ValueError(
            "Merged token score count does not match extracted video feature count: "
            f"scores={int(merged_scores.numel())}, features={expected_tokens}."
        )

    selected_indices = selected_indices.to(
        device=video_features.device,
        dtype=torch.long,
    )
    metadata = {
        "selector_type": "maskclip_patch_selection",
        "selection_mode": resolved_selection_mode,
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "aggregation": aggregation,
        "keep_ratio": float(keep_ratio),
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
        "selected_token_count": int(selected_indices.numel()),
        "video_token_count": expected_tokens,
        "per_frame": frame_metadata,
        **mode_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


maskclip_patch_selection.preload = preload_maskclip_patch_selection
