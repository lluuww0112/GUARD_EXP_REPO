from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

from ...base.selection import FrameSelectionResult, PatchSelectionResult
from .cilp_model import CLIPTextModel, CLIPVisionModel_v2
from .selection_v1 import (
    _aggregate_query_scores,
    _align_temporal_score_maps,
    _coerce_video_frames,
    _compute_qwen_merge_mean_scores,
    _compute_sliding_window_merged_scores,
    _compute_window_score_maps,
    _expand_scores_for_frame_duplication,
    _extract_video_grid,
    _load_queries,
    _prepare_frame_arrays,
    _resolve_patch_scoring_frames,
    _resolve_clip_dtype,
    _resolve_device,
    _resolve_device_key,
    _resolve_model_device,
    _resolve_selection_mode,
    _resolve_spatial_merge_size,
    _resolve_temporal_patch_size,
    _resize_score_maps,
    SUPPORTED_QWEN_BACKENDS,
)


@lru_cache(maxsize=4)
def _load_maskclip_components_v2(
    clip_model_name: str,
    device_type: str,
    clip_dtype_key: str,
) -> tuple[CLIPImageProcessor, Any, CLIPVisionModel_v2, CLIPTextModel]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel_v2(clip_model_name)
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
    _load_maskclip_components_v2(clip_model_name, device_key, clip_dtype_key)


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
def _load_text_embeddings_v2(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
    queries: tuple[str, ...],
) -> torch.Tensor:
    _, tokenizer, _, text_model = _load_maskclip_components_v2(
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


def _compute_dense_patch_score_maps_and_frame_scores(
    frame_arrays: list[Any],
    *,
    image_processor: CLIPImageProcessor,
    vision_model: CLIPVisionModel_v2,
    text_embeddings_t: torch.Tensor,
    aggregation: str,
    batch_size: int,
    device: torch.device,
    clip_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    score_chunks: list[torch.Tensor] = []
    frame_score_chunks: list[torch.Tensor] = []
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

            patch_embeddings, image_latents = vision_model(pixel_values)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            image_latents = F.normalize(image_latents, dim=-1)

            patch_scores = patch_embeddings @ text_embeddings_t
            patch_scores = _aggregate_query_scores(
                patch_scores,
                aggregation=aggregation,
            )

            frame_scores = image_latents @ text_embeddings_t
            frame_scores = _aggregate_query_scores(
                frame_scores,
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
            frame_score_chunks.append(frame_scores)

    if clip_grid is None:
        raise ValueError("No frames were provided for dense patch scoring.")

    return (
        torch.cat(score_chunks, dim=0),
        torch.cat(frame_score_chunks, dim=0),
        clip_grid,
    )


def _resolve_total_budget(
    *,
    merged_scores: torch.Tensor,
    keep_ratio: float,
    total_budget: int | None,
) -> int:
    total_tokens = int(merged_scores.numel())
    if total_budget is not None:
        resolved_total_budget = int(total_budget)
        if resolved_total_budget <= 0:
            raise ValueError(
                f"`total_budget` must be positive when provided, got {total_budget}."
            )
    else:
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"`keep_ratio` must be in (0, 1], got {keep_ratio}.")
        resolved_total_budget = max(1, int(math.ceil(total_tokens * keep_ratio)))

    if resolved_total_budget > total_tokens:
        raise ValueError(
            "Requested total patch budget exceeds the available token count: "
            f"budget={resolved_total_budget}, total_tokens={total_tokens}."
        )
    return resolved_total_budget


def _allocate_budget_with_softmax(
    frame_scores: torch.Tensor,
    *,
    total_budget: int,
    tokens_per_frame: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"`temperature` must be positive, got {temperature}.")
    if total_budget <= 0:
        raise ValueError(f"`total_budget` must be positive, got {total_budget}.")
    if tokens_per_frame <= 0:
        raise ValueError(f"`tokens_per_frame` must be positive, got {tokens_per_frame}.")

    scores = frame_scores.to(dtype=torch.float32)
    capacities = torch.full_like(scores, fill_value=tokens_per_frame, dtype=torch.long)
    if total_budget > int(capacities.sum().item()):
        raise ValueError(
            "Requested total patch budget exceeds frame capacities: "
            f"budget={total_budget}, capacity={int(capacities.sum().item())}."
        )

    quotas = torch.zeros_like(scores, dtype=torch.float32)
    remaining_budget = float(total_budget)
    active_mask = capacities > 0

    while remaining_budget > 1e-6:
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            raise RuntimeError("Failed to allocate the requested patch budget.")

        active_scores = scores[active_indices] / temperature
        active_weights = torch.softmax(active_scores, dim=0)
        active_capacities = capacities[active_indices].to(dtype=torch.float32)
        tentative = active_weights * remaining_budget
        saturated = tentative >= active_capacities

        if torch.any(saturated):
            saturated_indices = active_indices[saturated]
            quotas[saturated_indices] = capacities[saturated_indices].to(
                dtype=torch.float32
            )
            remaining_budget -= float(capacities[saturated_indices].sum().item())
            active_mask[saturated_indices] = False
            continue

        quotas[active_indices] = tentative
        remaining_budget = 0.0

    allocated = torch.floor(quotas).to(dtype=torch.long)
    remaining_integer_budget = total_budget - int(allocated.sum().item())
    if remaining_integer_budget > 0:
        fractional = quotas - allocated.to(dtype=quotas.dtype)
        available_mask = allocated < capacities
        candidate_indices = torch.nonzero(available_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            raise RuntimeError("No frames left to receive the remaining patch budget.")

        candidate_fractional = fractional[candidate_indices]
        candidate_scores = scores[candidate_indices]
        ranking = torch.argsort(
            candidate_fractional + candidate_scores * 1e-6,
            descending=True,
        )
        ranked_indices = candidate_indices[ranking]
        allocated[ranked_indices[:remaining_integer_budget]] += 1

    if int(allocated.sum().item()) != total_budget:
        raise RuntimeError(
            "Allocated patch budget does not match the requested total budget: "
            f"allocated={int(allocated.sum().item())}, requested={total_budget}."
        )

    final_weights = torch.softmax(scores / temperature, dim=0)
    return allocated, quotas, final_weights


def _select_topk_with_variable_budget(
    merged_scores: torch.Tensor,
    *,
    frame_scores: torch.Tensor,
    allocated_budget: torch.Tensor,
    raw_budget: torch.Tensor,
    frame_weights: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    merged_h = int(merged_scores.shape[1])
    merged_w = int(merged_scores.shape[2])
    tokens_per_frame = merged_h * merged_w
    frame_token_scores = merged_scores.reshape(merged_scores.shape[0], -1)

    min_scores = frame_token_scores.min(dim=1).values
    max_scores = frame_token_scores.max(dim=1).values

    selected_chunks: list[torch.Tensor] = []
    frame_metadata: list[dict[str, Any]] = []
    for frame_idx in range(int(merged_scores.shape[0])):
        keep_count = int(allocated_budget[frame_idx].item())
        if keep_count > tokens_per_frame:
            raise ValueError(
                "Allocated frame budget exceeds the per-frame token count: "
                f"frame_index={frame_idx}, keep_count={keep_count}, "
                f"tokens_per_frame={tokens_per_frame}."
            )

        if keep_count > 0:
            topk_indices = torch.topk(
                frame_token_scores[frame_idx],
                k=keep_count,
                dim=0,
                largest=True,
                sorted=False,
            ).indices
            topk_indices = torch.sort(topk_indices).values
            frame_offset = frame_idx * tokens_per_frame
            selected_chunks.append(topk_indices + frame_offset)

        frame_metadata.append(
            {
                "frame_index": int(frame_idx),
                "tokens_per_frame": tokens_per_frame,
                "keep_count": keep_count,
                "score_min": float(min_scores[frame_idx].item()),
                "score_max": float(max_scores[frame_idx].item()),
                "frame_importance_score": float(frame_scores[frame_idx].item()),
                "softmax_weight": float(frame_weights[frame_idx].item()),
                "raw_budget": float(raw_budget[frame_idx].item()),
            }
        )

    if not selected_chunks:
        raise ValueError("Patch selector produced an empty selection.")
    return torch.cat(selected_chunks, dim=0), frame_metadata


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
    total_budget: int | None = None,
    temperature: float = 1.0,
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
    image_processor, _, vision_model, _ = _load_maskclip_components_v2(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
    )

    text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
        queries,
    )
    frame_arrays = _prepare_frame_arrays(scoring_frames)
    dense_score_maps, image_frame_scores, clip_grid = (
        _compute_dense_patch_score_maps_and_frame_scores(
            frame_arrays,
            image_processor=image_processor,
            vision_model=vision_model,
            text_embeddings_t=text_embeddings.transpose(0, 1).contiguous(),
            aggregation=aggregation,
            batch_size=batch_size,
            device=selector_device,
            clip_dtype=resolved_clip_dtype,
        )
    )
    dense_score_maps = _expand_scores_for_frame_duplication(
        dense_score_maps,
        duplication_info,
    )
    image_frame_scores = _expand_scores_for_frame_duplication(
        image_frame_scores,
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
    image_frame_scores = _align_temporal_score_maps(
        image_frame_scores.view(frame_count, 1, 1),
        grid_t=grid_t,
        temporal_patch_size=temporal_patch_size,
    ).flatten()

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

    expected_tokens = int(video_features.shape[0])
    if int(merged_scores.numel()) != expected_tokens:
        raise ValueError(
            "Merged token score count does not match extracted video feature count: "
            f"scores={int(merged_scores.numel())}, features={expected_tokens}."
        )

    total_budget_value = _resolve_total_budget(
        merged_scores=merged_scores,
        keep_ratio=keep_ratio,
        total_budget=total_budget,
    )
    tokens_per_frame = int(merged_scores.shape[1] * merged_scores.shape[2])
    allocated_budget, raw_budget, frame_weights = _allocate_budget_with_softmax(
        image_frame_scores,
        total_budget=total_budget_value,
        tokens_per_frame=tokens_per_frame,
        temperature=temperature,
    )
    selected_indices, frame_metadata = _select_topk_with_variable_budget(
        merged_scores,
        frame_scores=image_frame_scores,
        allocated_budget=allocated_budget,
        raw_budget=raw_budget,
        frame_weights=frame_weights,
    )

    selected_indices = selected_indices.to(
        device=video_features.device,
        dtype=torch.long,
    )
    metadata = {
        "selector_type": "maskclip_patch_selection_v2",
        "selection_mode": resolved_selection_mode,
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "aggregation": aggregation,
        "keep_ratio": float(keep_ratio),
        "total_budget": int(total_budget_value),
        "temperature": float(temperature),
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
        "frame_importance_scores": [float(score) for score in image_frame_scores.tolist()],
        "frame_softmax_weights": [float(weight) for weight in frame_weights.tolist()],
        "frame_raw_budgets": [float(budget) for budget in raw_budget.tolist()],
        "frame_allocated_budgets": [int(budget) for budget in allocated_budget.tolist()],
        "per_frame": frame_metadata,
        **mode_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


maskclip_patch_selection.preload = preload_maskclip_patch_selection
