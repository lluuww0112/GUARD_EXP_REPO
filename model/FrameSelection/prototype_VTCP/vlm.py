from __future__ import annotations

from typing import Any

import torch

from model.base.selection import FrameSelectionResult
from model.base.vlm import PromptInput
from model.FrameSelection.MDP3.vlm import MDP3VLM


class VTCPVLM(MDP3VLM):

    def _embed_frame_for_vtcp(self, frames: torch.Tensor) -> torch.Tensor:
        return self._get_image_embeddings(frames)

    def _embed_query_for_vtcp(self, query: str) -> torch.Tensor:
        return self._get_text_embedding(query)

    def answer(
        self,
        video_path: str,
        prompt: PromptInput,
        **frame_selector_kwargs: Any,
    ) -> str:
        if self.frame_selector is None:
            return super(MDP3VLM, self).answer(
                video_path=video_path, prompt=prompt, **frame_selector_kwargs,
            )

        raw_query = ""
        if isinstance(prompt, dict):
            raw_query = str(prompt.get("query", "") or "").strip()

        selector_kwargs = {
            **self.frame_selector_kwargs,
            **frame_selector_kwargs,
            "embed_frame_fn": self._embed_frame_for_vtcp,   # CHANGED
            "embed_query_fn": self._embed_query_for_vtcp,   # CHANGED
            "query": raw_query,
        }
        frame_selection = self._normalize_frame_selection_output(
            self.frame_selector(video_path=video_path, **selector_kwargs),
            video_path=video_path,
        )
        frame_selection = self._duplicate_frame_selection(frame_selection)
        print("=== VTCP Debug ===")
        print("visited_count:", frame_selection.metadata.get("visited_count"))
        print("selected:", frame_selection.metadata.get("sampled_indices"))
        print("threshold_selected:", frame_selection.metadata.get("threshold_selected_indices"))
        print("budget_filled_strict:", frame_selection.metadata.get("budget_filled_strict"))
        print("budget_filled_relaxed:", frame_selection.metadata.get("budget_filled_relaxed"))

        if not isinstance(frame_selection, FrameSelectionResult):
            raise TypeError("VTCP frame selector must return a FrameSelectionResult.")

        self.last_frame_selection = frame_selection
        video_tensor = frame_selection.frames
        prompt_text = self._prepare_text_input(
            prompt,
            has_video=video_tensor is not None,
        )
        self._reset_multimodal_state()
        model_inputs = self._build_model_inputs(
            prompt_text=prompt_text,
            video_tensor=video_tensor,
            frame_selection=frame_selection,
        )
        self.last_timing_info = {}

        if self.patch_selector is not None and video_tensor is not None:
            return self._run_patch_selection_generation(
                prompt=prompt,
                frame_selection=frame_selection,
                model_inputs=model_inputs,
            )

        self.last_patch_selection_info = {
            "applied": False,
            "backend": self.backend,
            "reason": "patch_selector_not_configured",
        }
        return self._run_standard_generation(model_inputs)
