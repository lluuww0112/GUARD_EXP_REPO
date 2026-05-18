from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _coerce_average_metric(
    value: float | int | None,
) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_runtime_metrics(vlm: Any) -> dict[str, Any]:
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
    original_llm_input_sequence_length = input_sequence_length_before_patch
    if original_llm_input_sequence_length is None:
        original_llm_input_sequence_length = input_sequence_length

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

    llm_generate_flops = _coerce_float(
        timing_info.get("llm_generate_flops")
    )
    llm_generate_gflops = _coerce_float(
        timing_info.get("llm_generate_gflops")
    )
    if llm_generate_gflops is None and llm_generate_flops is not None:
        llm_generate_gflops = llm_generate_flops / 1e9

    top_flop_ops = timing_info.get("llm_generate_top_flop_ops")
    if not isinstance(top_flop_ops, list):
        top_flop_ops = None

    latency_seconds = _coerce_float(
        timing_info.get("latency_seconds")
    )
    llm_prefill_seconds = _coerce_float(
        timing_info.get("llm_prefill_seconds")
    )
    gpu_total_memory_bytes = _coerce_int(
        timing_info.get("gpu_total_memory_bytes")
    )
    gpu_peak_memory_allocated_bytes = _coerce_int(
        timing_info.get("gpu_peak_memory_allocated_bytes")
    )
    gpu_peak_memory_reserved_bytes = _coerce_int(
        timing_info.get("gpu_peak_memory_reserved_bytes")
    )
    gpu_memory_allocated_bytes = _coerce_int(
        timing_info.get("gpu_memory_allocated_bytes")
    )
    gpu_memory_reserved_bytes = _coerce_int(
        timing_info.get("gpu_memory_reserved_bytes")
    )

    return {
        "patch_selection_applied": bool(patch_info.get("applied", False)),
        "llm_input_sequence_length": input_sequence_length,
        "original_llm_input_sequence_length": original_llm_input_sequence_length,
        "llm_input_sequence_length_before_patch_selection": input_sequence_length_before_patch,
        "visual_token_count": selected_video_tokens,
        "original_visual_token_count": original_video_tokens,
        "original_video_token_count": original_video_tokens,
        "selected_video_token_count": selected_video_tokens,
        "reallocated_patch_count": reallocated_patch_count,
        "llm_generate_flops_profiled": bool(
            timing_info.get("llm_generate_flops_profiled", False)
        ),
        "llm_generate_flops": llm_generate_flops,
        "llm_generate_gflops": llm_generate_gflops,
        "llm_generate_profiled_ops": _coerce_int(
            timing_info.get("llm_generate_profiled_ops")
        ),
        "llm_generate_profiled_flop_ops": _coerce_int(
            timing_info.get("llm_generate_profiled_flop_ops")
        ),
        "llm_generate_top_flop_ops": top_flop_ops,
        "llm_generate_flops_error": timing_info.get("llm_generate_flops_error"),
        "latency_seconds": latency_seconds,
        "llm_prefill_seconds": llm_prefill_seconds,
        "gpu_device": timing_info.get("gpu_device"),
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "gpu_memory_allocated_bytes": gpu_memory_allocated_bytes,
        "gpu_memory_reserved_bytes": gpu_memory_reserved_bytes,
        "gpu_peak_memory_allocated_bytes": gpu_peak_memory_allocated_bytes,
        "gpu_peak_memory_reserved_bytes": gpu_peak_memory_reserved_bytes,
        "gpu_memory_error": timing_info.get("gpu_memory_error"),
    }


def init_runtime_metric_totals() -> dict[str, float | int]:
    return {
        "visual_token_sum": 0.0,
        "visual_token_count": 0,
        "input_sequence_length_sum": 0.0,
        "input_sequence_length_count": 0,
        "original_visual_token_sum": 0.0,
        "original_visual_token_count": 0,
        "original_input_sequence_length_sum": 0.0,
        "original_input_sequence_length_count": 0,
        "reallocated_patch_sum": 0.0,
        "reallocated_patch_count": 0,
        "llm_generate_flops_sum": 0.0,
        "llm_generate_flops_count": 0,
        "latency_seconds_sum": 0.0,
        "latency_seconds_count": 0,
        "llm_prefill_seconds_sum": 0.0,
        "llm_prefill_seconds_count": 0,
        "gpu_total_memory_bytes_max": 0,
        "gpu_total_memory_bytes_count": 0,
        "gpu_memory_allocated_bytes_sum": 0.0,
        "gpu_memory_allocated_bytes_max": 0,
        "gpu_memory_allocated_bytes_count": 0,
        "gpu_peak_memory_allocated_bytes_max": 0,
        "gpu_peak_memory_allocated_bytes_count": 0,
        "gpu_peak_memory_reserved_bytes_max": 0,
        "gpu_peak_memory_reserved_bytes_count": 0,
    }


def update_runtime_metric_totals(
    totals: dict[str, float | int],
    metrics: Mapping[str, Any],
) -> None:
    visual_token_count = metrics.get("visual_token_count")
    if isinstance(visual_token_count, int):
        totals["visual_token_sum"] += float(visual_token_count)
        totals["visual_token_count"] += 1

    input_sequence_length = metrics.get("llm_input_sequence_length")
    if isinstance(input_sequence_length, int):
        totals["input_sequence_length_sum"] += float(input_sequence_length)
        totals["input_sequence_length_count"] += 1

    original_visual_token_count = metrics.get("original_visual_token_count")
    if isinstance(original_visual_token_count, int):
        totals["original_visual_token_sum"] += float(original_visual_token_count)
        totals["original_visual_token_count"] += 1

    original_input_sequence_length = metrics.get("original_llm_input_sequence_length")
    if isinstance(original_input_sequence_length, int):
        totals["original_input_sequence_length_sum"] += float(original_input_sequence_length)
        totals["original_input_sequence_length_count"] += 1

    reallocated_patch_count = metrics.get("reallocated_patch_count")
    if isinstance(reallocated_patch_count, int):
        totals["reallocated_patch_sum"] += float(reallocated_patch_count)
        totals["reallocated_patch_count"] += 1

    llm_generate_flops = _coerce_float(metrics.get("llm_generate_flops"))
    if llm_generate_flops is not None:
        totals["llm_generate_flops_sum"] += llm_generate_flops
        totals["llm_generate_flops_count"] += 1

    latency_seconds = _coerce_float(metrics.get("latency_seconds"))
    if latency_seconds is not None:
        totals["latency_seconds_sum"] += latency_seconds
        totals["latency_seconds_count"] += 1

    llm_prefill_seconds = _coerce_float(metrics.get("llm_prefill_seconds"))
    if llm_prefill_seconds is not None:
        totals["llm_prefill_seconds_sum"] += llm_prefill_seconds
        totals["llm_prefill_seconds_count"] += 1

    gpu_total_memory_bytes = _coerce_int(metrics.get("gpu_total_memory_bytes"))
    if gpu_total_memory_bytes is not None:
        totals["gpu_total_memory_bytes_max"] = max(
            int(totals.get("gpu_total_memory_bytes_max", 0)),
            gpu_total_memory_bytes,
        )
        totals["gpu_total_memory_bytes_count"] += 1

    gpu_memory_allocated_bytes = _coerce_int(
        metrics.get("gpu_memory_allocated_bytes")
    )
    if gpu_memory_allocated_bytes is not None:
        totals["gpu_memory_allocated_bytes_sum"] += float(gpu_memory_allocated_bytes)
        totals["gpu_memory_allocated_bytes_max"] = max(
            int(totals.get("gpu_memory_allocated_bytes_max", 0)),
            gpu_memory_allocated_bytes,
        )
        totals["gpu_memory_allocated_bytes_count"] += 1

    gpu_peak_memory_allocated_bytes = _coerce_int(
        metrics.get("gpu_peak_memory_allocated_bytes")
    )
    if gpu_peak_memory_allocated_bytes is not None:
        totals["gpu_peak_memory_allocated_bytes_max"] = max(
            int(totals.get("gpu_peak_memory_allocated_bytes_max", 0)),
            gpu_peak_memory_allocated_bytes,
        )
        totals["gpu_peak_memory_allocated_bytes_count"] += 1

    gpu_peak_memory_reserved_bytes = _coerce_int(
        metrics.get("gpu_peak_memory_reserved_bytes")
    )
    if gpu_peak_memory_reserved_bytes is not None:
        totals["gpu_peak_memory_reserved_bytes_max"] = max(
            int(totals.get("gpu_peak_memory_reserved_bytes_max", 0)),
            gpu_peak_memory_reserved_bytes,
        )
        totals["gpu_peak_memory_reserved_bytes_count"] += 1


def summarize_runtime_metric_totals(
    totals: Mapping[str, float | int],
) -> dict[str, float | int | None]:
    visual_token_count = int(totals.get("visual_token_count", 0))
    input_count = int(totals.get("input_sequence_length_count", 0))
    original_visual_token_count = int(totals.get("original_visual_token_count", 0))
    original_input_count = int(totals.get("original_input_sequence_length_count", 0))
    reallocation_count = int(totals.get("reallocated_patch_count", 0))
    flops_count = int(totals.get("llm_generate_flops_count", 0))
    latency_count = int(totals.get("latency_seconds_count", 0))
    prefill_count = int(totals.get("llm_prefill_seconds_count", 0))
    gpu_total_memory_count = int(totals.get("gpu_total_memory_bytes_count", 0))
    gpu_alloc_count = int(totals.get("gpu_memory_allocated_bytes_count", 0))
    gpu_peak_alloc_count = int(totals.get("gpu_peak_memory_allocated_bytes_count", 0))
    gpu_peak_reserved_count = int(totals.get("gpu_peak_memory_reserved_bytes_count", 0))
    visual_token_sum = float(totals.get("visual_token_sum", 0.0))
    input_sum = float(totals.get("input_sequence_length_sum", 0.0))
    original_visual_token_sum = float(totals.get("original_visual_token_sum", 0.0))
    original_input_sum = float(totals.get("original_input_sequence_length_sum", 0.0))
    reallocation_sum = float(totals.get("reallocated_patch_sum", 0.0))
    flops_sum = float(totals.get("llm_generate_flops_sum", 0.0))
    latency_sum = float(totals.get("latency_seconds_sum", 0.0))
    prefill_sum = float(totals.get("llm_prefill_seconds_sum", 0.0))
    gpu_total_memory_max = int(totals.get("gpu_total_memory_bytes_max", 0))
    gpu_alloc_sum = float(totals.get("gpu_memory_allocated_bytes_sum", 0.0))
    gpu_alloc_max = int(totals.get("gpu_memory_allocated_bytes_max", 0))
    gpu_peak_alloc_max = int(totals.get("gpu_peak_memory_allocated_bytes_max", 0))
    gpu_peak_reserved_max = int(totals.get("gpu_peak_memory_reserved_bytes_max", 0))

    return {
        "avg_visual_token_count": (
            visual_token_sum / visual_token_count if visual_token_count > 0 else None
        ),
        "visual_token_samples": visual_token_count,
        "avg_llm_input_sequence_length": (
            input_sum / input_count if input_count > 0 else None
        ),
        "llm_input_sequence_length_samples": input_count,
        "avg_original_visual_token_count": (
            original_visual_token_sum / original_visual_token_count
            if original_visual_token_count > 0 else None
        ),
        "original_visual_token_samples": original_visual_token_count,
        "avg_original_llm_input_sequence_length": (
            original_input_sum / original_input_count
            if original_input_count > 0 else None
        ),
        "original_llm_input_sequence_length_samples": original_input_count,
        "avg_reallocated_patch_count": (
            reallocation_sum / reallocation_count if reallocation_count > 0 else None
        ),
        "reallocated_patch_samples": reallocation_count,
        "avg_llm_generate_flops": (
            flops_sum / flops_count if flops_count > 0 else None
        ),
        "avg_llm_generate_gflops": (
            flops_sum / flops_count / 1e9 if flops_count > 0 else None
        ),
        "llm_generate_flops_samples": flops_count,
        "avg_latency_seconds": (
            latency_sum / latency_count if latency_count > 0 else None
        ),
        "latency_seconds_samples": latency_count,
        "avg_llm_prefill_seconds": (
            prefill_sum / prefill_count if prefill_count > 0 else None
        ),
        "llm_prefill_seconds_samples": prefill_count,
        "gpu_total_memory_bytes": (
            gpu_total_memory_max if gpu_total_memory_count > 0 else None
        ),
        "gpu_total_memory_samples": gpu_total_memory_count,
        "avg_gpu_memory_allocated_bytes": (
            gpu_alloc_sum / gpu_alloc_count if gpu_alloc_count > 0 else None
        ),
        "max_gpu_memory_allocated_bytes": (
            gpu_alloc_max if gpu_alloc_count > 0 else None
        ),
        "gpu_memory_allocated_samples": gpu_alloc_count,
        "max_gpu_peak_memory_allocated_bytes": (
            gpu_peak_alloc_max if gpu_peak_alloc_count > 0 else None
        ),
        "gpu_peak_memory_allocated_samples": gpu_peak_alloc_count,
        "max_gpu_peak_memory_reserved_bytes": (
            gpu_peak_reserved_max if gpu_peak_reserved_count > 0 else None
        ),
        "gpu_peak_memory_reserved_samples": gpu_peak_reserved_count,
    }


def _format_average_metric(value: float | int | None, *, samples: int) -> str:
    resolved_value = _coerce_average_metric(value)
    if resolved_value is None:
        return "N/A"
    return f"{resolved_value:.2f} (n={samples})"


def _format_seconds_metric(value: float | int | None, *, samples: int) -> str:
    resolved_value = _coerce_average_metric(value)
    if resolved_value is None:
        return "N/A"
    return f"{resolved_value:.4f}s (n={samples})"


def _format_bytes_metric(value: float | int | None, *, samples: int) -> str:
    resolved_value = _coerce_average_metric(value)
    if resolved_value is None:
        return "N/A"
    return f"{resolved_value / (1024 ** 3):.2f} GiB (n={samples})"


def format_runtime_summary_lines(
    summary: Mapping[str, float | int | None],
) -> list[tuple[str, str]]:
    return [
        (
            "Avg Visual",
            _format_average_metric(
                summary.get("avg_visual_token_count"),
                samples=int(summary.get("visual_token_samples", 0) or 0),
            ),
        ),
        (
            "Avg LLM Seq",
            _format_average_metric(
                summary.get("avg_llm_input_sequence_length"),
                samples=int(summary.get("llm_input_sequence_length_samples", 0) or 0),
            ),
        ),
        (
            "Avg Orig Vis",
            _format_average_metric(
                summary.get("avg_original_visual_token_count"),
                samples=int(summary.get("original_visual_token_samples", 0) or 0),
            ),
        ),
        (
            "Avg Orig Seq",
            _format_average_metric(
                summary.get("avg_original_llm_input_sequence_length"),
                samples=int(summary.get("original_llm_input_sequence_length_samples", 0) or 0),
            ),
        ),
        (
            "Avg Realloc",
            _format_average_metric(
                summary.get("avg_reallocated_patch_count"),
                samples=int(summary.get("reallocated_patch_samples", 0) or 0),
            ),
        ),
        (
            "Avg LLM GFLOPs",
            _format_average_metric(
                summary.get("avg_llm_generate_gflops"),
                samples=int(summary.get("llm_generate_flops_samples", 0) or 0),
            ),
        ),
        (
            "Avg Latency",
            _format_seconds_metric(
                summary.get("avg_latency_seconds"),
                samples=int(summary.get("latency_seconds_samples", 0) or 0),
            ),
        ),
        (
            "Avg Prefill",
            _format_seconds_metric(
                summary.get("avg_llm_prefill_seconds"),
                samples=int(summary.get("llm_prefill_seconds_samples", 0) or 0),
            ),
        ),
        (
            "GPU Total Mem",
            _format_bytes_metric(
                summary.get("gpu_total_memory_bytes"),
                samples=int(summary.get("gpu_total_memory_samples", 0) or 0),
            ),
        ),
        (
            "Avg GPU Alloc",
            _format_bytes_metric(
                summary.get("avg_gpu_memory_allocated_bytes"),
                samples=int(summary.get("gpu_memory_allocated_samples", 0) or 0),
            ),
        ),
        (
            "Max GPU Alloc",
            _format_bytes_metric(
                summary.get("max_gpu_memory_allocated_bytes"),
                samples=int(summary.get("gpu_memory_allocated_samples", 0) or 0),
            ),
        ),
        (
            "GPU Peak Alloc",
            _format_bytes_metric(
                summary.get("max_gpu_peak_memory_allocated_bytes"),
                samples=int(summary.get("gpu_peak_memory_allocated_samples", 0) or 0),
            ),
        ),
        (
            "GPU Peak Reserv",
            _format_bytes_metric(
                summary.get("max_gpu_peak_memory_reserved_bytes"),
                samples=int(summary.get("gpu_peak_memory_reserved_samples", 0) or 0),
            ),
        ),
    ]
