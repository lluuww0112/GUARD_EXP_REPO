from __future__ import annotations

import functools
from typing import Any, Callable

from model.base.vlm import BaseVLM


class VTCPVLM(BaseVLM):
    def _resolve_preload_hook(self) -> tuple[Callable[..., Any], dict[str, Any]] | None:
        preload_target = super()._resolve_preload_hook()
        if preload_target is not None:
            return preload_target

        if self.frame_selector is None:
            return None

        selector = self.frame_selector
        selector_kwargs: dict[str, Any] = {}
        if isinstance(selector, functools.partial):
            selector_kwargs = dict(selector.keywords or {})
            selector = selector.func

        preload_hook = getattr(self.frame_selector, "preload", None)
        if preload_hook is None:
            preload_hook = getattr(selector, "preload", None)
        if preload_hook is None or not callable(preload_hook):
            return None

        return preload_hook, selector_kwargs
