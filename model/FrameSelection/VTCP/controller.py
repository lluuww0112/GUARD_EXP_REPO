from __future__ import annotations

import math


def _validate_stride_bounds(
    *,
    min_stride: int,
    max_stride: int,
) -> None:
    if min_stride <= 0:
        raise ValueError(f"`min_stride` must be positive, got {min_stride}.")
    if max_stride < min_stride:
        raise ValueError(
            "`max_stride` must be greater than or equal to `min_stride`, "
            f"got min={min_stride}, max={max_stride}."
        )


def _clamp_stride(
    stride: int | float,
    *,
    min_stride: int,
    max_stride: int,
) -> int:
    return max(min_stride, min(int(stride), max_stride))


class RenoStrideController:
    def __init__(
        self,
        additive_step: int = 1,
        decrease_factor: float = 0.5,
        score_threshold: float = 0.5,
        slow_start_threshold: int = 8,
        slow_start_multiplier: float = 2.0,
        min_stride: int = 1,
        max_stride: int = 32,
    ) -> None:
        if additive_step <= 0:
            raise ValueError(
                f"`additive_step` must be positive, got {additive_step}."
            )
        if not 0.0 < decrease_factor < 1.0:
            raise ValueError(
                "`decrease_factor` must be in the open interval (0, 1), "
                f"got {decrease_factor}."
            )
        if slow_start_multiplier <= 1.0:
            raise ValueError(
                "`slow_start_multiplier` must be greater than 1.0, "
                f"got {slow_start_multiplier}."
            )
        _validate_stride_bounds(
            min_stride=min_stride,
            max_stride=max_stride,
        )
        if not min_stride <= slow_start_threshold <= max_stride:
            raise ValueError(
                "`slow_start_threshold` must be within "
                f"[{min_stride}, {max_stride}], got {slow_start_threshold}."
            )

        self.additive_step = int(additive_step)
        self.decrease_factor = float(decrease_factor)
        self.score_threshold = float(score_threshold)
        self.slow_start_threshold = int(slow_start_threshold)
        self.slow_start_multiplier = float(slow_start_multiplier)
        self.min_stride = int(min_stride)
        self.max_stride = int(max_stride)

    def __call__(
        self,
        current_stride: int,
        score: float,
    ) -> int:
        stride = _clamp_stride(
            current_stride,
            min_stride=self.min_stride,
            max_stride=self.max_stride,
        )
        if float(score) >= self.score_threshold:
            next_stride = max(
                self.min_stride,
                int(stride * self.decrease_factor),
            )
        elif stride < self.slow_start_threshold:
            next_stride = max(
                stride + 1,
                int(math.ceil(stride * self.slow_start_multiplier)),
            )
        else:
            next_stride = stride + self.additive_step

        return _clamp_stride(
            next_stride,
            min_stride=self.min_stride,
            max_stride=self.max_stride,
        )
