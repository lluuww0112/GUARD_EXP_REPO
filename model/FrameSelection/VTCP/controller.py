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


class EMAStrideController:
    """Adapt stride continuously using score novelty against an EMA baseline."""

    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 1.0,
        min_stride: int = 1,
        max_stride: int = 32,
        eps: float = 1.0e-6,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f"`alpha` must be in the interval (0, 1], got {alpha}."
            )
        if gamma <= 0.0:
            raise ValueError(f"`gamma` must be positive, got {gamma}.")
        if eps <= 0.0:
            raise ValueError(f"`eps` must be positive, got {eps}.")
        _validate_stride_bounds(
            min_stride=min_stride,
            max_stride=max_stride,
        )

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.min_stride = int(min_stride)
        self.max_stride = int(max_stride)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._ema_score: float | None = None
        self._ema_history: list[float] = []
        self._novelty_history: list[float] = []
        self._target_stride_history: list[float] = []

    def _map_novelty_to_target_stride(
        self,
        novelty: float,
    ) -> float:
        scaled_novelty = math.pow(max(novelty, self.eps), self.gamma)
        stride_span = float(self.max_stride - self.min_stride)
        return float(self.min_stride) + (
            stride_span / (1.0 + scaled_novelty)
        )

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
        score_value = max(float(score), 0.0)

        if self._ema_score is None:
            self._ema_score = max(score_value, self.eps)
            self._ema_history.append(self._ema_score)
            self._novelty_history.append(1.0)
            self._target_stride_history.append(float(stride))
            return stride

        baseline = max(self._ema_score, self.eps)
        novelty = max(score_value / baseline, self.eps)
        target_stride = self._map_novelty_to_target_stride(novelty)
        # Blend toward an absolute target stride instead of multiplying the
        # current stride. This avoids runaway growth and makes recovery from
        # a temporary spike much easier.
        mapped_stride = (
            (1.0 - self.alpha) * float(stride)
            + self.alpha * target_stride
        )
        next_stride = _clamp_stride(
            int(round(mapped_stride)),
            min_stride=self.min_stride,
            max_stride=self.max_stride,
        )

        self._ema_score = (
            self.alpha * score_value
            + (1.0 - self.alpha) * self._ema_score
        )
        self._ema_history.append(self._ema_score)
        self._novelty_history.append(novelty)
        self._target_stride_history.append(target_stride)
        return next_stride

    def get_diagnostics(self) -> dict[str, list[float]]:
        return {
            "ema_history": list(self._ema_history),
            "novelty_history": list(self._novelty_history),
            "target_stride_history": list(self._target_stride_history),
        }
