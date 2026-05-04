from __future__ import annotations

from typing import Any

import torch

from model.base.vlm import BaseVLM, PromptInput


class DPCVLM(BaseVLM):
    def _resolve_shared_clip_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            return torch.device(device)

        try:
            first_parameter = next(self.model.parameters())
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return first_parameter.device

    def answer(
        self,
        video_path: str,
        prompt: PromptInput,
        **frame_selector_kwargs: Any,
    ) -> str:
        shared_kwargs = {
            "device": self._resolve_shared_clip_device(),
            **frame_selector_kwargs,
        }
        return super().answer(
            video_path=video_path,
            prompt=prompt,
            **shared_kwargs,
        )
