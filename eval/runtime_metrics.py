from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_runtime_metrics(vlm: Any) -> dict[str, int | bool | None]:
    timing_info = getattr(vlm, "last_timing_info", {}) or {}
    patch_info = getattr(vlm, "last_patch_selection_info", {}) or {}
    selector_metadata = patch_info.get("selector_metadata")
    if not isinstance(selector_metadata, Mapping):
        selector_metadata = {}

    input_sequence_length = _coerce_int(
        timing_info.get("input_sequence_length")
    )
    if input_sequence_length is None:
        input_sequence_length = _coerce_int(
            patch_info.get("input_length_after")
        )

    input_sequence_length_before_patch = _coerce_int(
        patch_info.get("input_length_before")
    )
    selected_video_tokens = _coerce_int(
        patch_info.get("selected_video_tokens")
    )
    original_video_tokens = _coerce_int(
        patch_info.get("original_video_tokens")
    )
    reallocated_patch_count = _coerce_int(
        patch_info.get("reallocated_token_count")
    )
    if reallocated_patch_count is None:
        reallocated_patch_count = _coerce_int(
            selector_metadata.get("reallocated_token_count")
        )

    return {
        "patch_selection_applied": bool(patch_info.get("applied", False)),
        "llm_input_sequence_length": input_sequence_length,
        "llm_input_sequence_length_before_patch_selection": input_sequence_length_before_patch,
        "original_video_token_count": original_video_tokens,
        "selected_video_token_count": selected_video_tokens,
        "reallocated_patch_count": reallocated_patch_count,
    }


def init_runtime_metric_totals() -> dict[str, float | int]:
    return {
        "input_sequence_length_sum": 0.0,
        "input_sequence_length_count": 0,
        "reallocated_patch_sum": 0.0,
        "reallocated_patch_count": 0,
    }


def update_runtime_metric_totals(
    totals: dict[str, float | int],
    metrics: Mapping[str, int | bool | None],
) -> None:
    input_sequence_length = metrics.get("llm_input_sequence_length")
    if isinstance(input_sequence_length, int):
        totals["input_sequence_length_sum"] += float(input_sequence_length)
        totals["input_sequence_length_count"] += 1

    reallocated_patch_count = metrics.get("reallocated_patch_count")
    if isinstance(reallocated_patch_count, int):
        totals["reallocated_patch_sum"] += float(reallocated_patch_count)
        totals["reallocated_patch_count"] += 1


def summarize_runtime_metric_totals(
    totals: Mapping[str, float | int],
) -> dict[str, float | int | None]:
    input_count = int(totals.get("input_sequence_length_count", 0))
    reallocation_count = int(totals.get("reallocated_patch_count", 0))
    input_sum = float(totals.get("input_sequence_length_sum", 0.0))
    reallocation_sum = float(totals.get("reallocated_patch_sum", 0.0))

    return {
        "avg_llm_input_sequence_length": (
            input_sum / input_count if input_count > 0 else None
        ),
        "llm_input_sequence_length_samples": input_count,
        "avg_reallocated_patch_count": (
            reallocation_sum / reallocation_count if reallocation_count > 0 else None
        ),
        "reallocated_patch_samples": reallocation_count,
    }
