from __future__ import annotations

from typing import Any

import torch

from model.base.selection import FrameSelectionResult
from model.FrameSelection.prototype_VTCP.vlm import VTCPVLM


class LlavaOneVisionVTCPVLM(VTCPVLM):
    """VTCP VLM adapter for LLaVA-OneVision video inference."""

    def _build_embedding_encoder(self) -> tuple[Any, Any]:
        processor, embedding_model = super()._build_embedding_encoder()
        if torch.cuda.is_available():
            target_device = torch.device("cuda:0")
            current_device = next(embedding_model.parameters()).device
            if current_device.type != "cuda":
                try:
                    embedding_model.to(target_device)
                except RuntimeError:
                    embedding_model.to(current_device)
        embedding_model.eval()
        return processor, embedding_model

    def _build_chat_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        has_video: bool,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if has_video:
            content.append({"type": "video"})

        text_prompt = user_prompt
        if system_prompt:
            text_prompt = f"{system_prompt}\n\n{user_prompt}"
        content.append({"type": "text", "text": text_prompt})
        return [{"role": "user", "content": content}]

    def _build_model_inputs(
        self,
        prompt_text: str,
        video_tensor: torch.Tensor | None,
        frame_selection: FrameSelectionResult | None = None,
    ) -> dict[str, Any]:
        processor_inputs: dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
        }
        if video_tensor is not None:
            video_np = video_tensor.detach().cpu().numpy()
            processor_inputs["videos"] = list(video_np)

        processor_attempts = [processor_inputs]
        if video_tensor is not None:
            processor_attempts.extend(
                [
                    {**processor_inputs, "videos": [list(video_np)], "text": [prompt_text]},
                    {**processor_inputs, "videos": video_np},
                    {**processor_inputs, "videos": [video_np], "text": [prompt_text]},
                ]
            )

        last_error: Exception | None = None
        inputs = None
        for attempt_kwargs in processor_attempts:
            try:
                inputs = self.processor(**attempt_kwargs)
                break
            except (TypeError, ValueError) as exc:
                last_error = exc

        if inputs is None:
            raise last_error or RuntimeError("Failed to build LLaVA-OneVision inputs.")

        device = self.model.device
        return {
            key: self._move_model_input_to_device(key, value, device=device)
            for key, value in inputs.items()
        }

    def _move_model_input_to_device(
        self,
        key: str,
        value: Any,
        *,
        device: torch.device,
    ) -> Any:
        if not hasattr(value, "to"):
            return value
        if torch.is_tensor(value) and value.is_floating_point():
            model_dtype = next(self.model.parameters()).dtype
            return value.to(device=device, dtype=model_dtype)
        return value.to(device)
