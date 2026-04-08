from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from model.base.selection import FrameSelectionResult
from model.base.vlm import BaseVLM, DTYPE_MAP, PromptInput

# BaseVLM 상속
class MDP3VLM(BaseVLM):
    def __init__(
        self,
        model_id: str,
        frame_selector: Any,
        patch_selector: Any | None = None,
        *,
        embedding_model_id: str = "google/siglip-so400m-patch14-384",
        embedding_dtype: str | torch.dtype | None = None,
        embedding_processor_kwargs: dict[str, Any] | None = None,
        embedding_model_kwargs: dict[str, Any] | None = None,
        embedding_local_model_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_id=model_id,
            frame_selector=frame_selector,
            patch_selector=patch_selector,
            **kwargs,
        )
        self.embedding_model_id = embedding_model_id
        self.embedding_dtype = (
            DTYPE_MAP[embedding_dtype]
            if isinstance(embedding_dtype, str)
            else embedding_dtype
        )
        self.embedding_processor_kwargs = dict(embedding_processor_kwargs or {})
        self.embedding_model_kwargs = dict(embedding_model_kwargs or {})
        self.embedding_local_model_dir = (
            Path(embedding_local_model_dir).expanduser()
            if embedding_local_model_dir is not None
            else None
        )
        self.embedding_processor, self.embedding_model = self._build_embedding_encoder()

    def _resolve_embedding_model_source(self) -> str:
        candidate = Path(self.embedding_model_id).expanduser()
        if candidate.exists():
            return str(candidate.resolve())

        if self.embedding_local_model_dir is None:
            return self.embedding_model_id

        mirrored = self.embedding_local_model_dir / self.embedding_model_id.replace("/", "__")
        if mirrored.exists():
            return str(mirrored.resolve())
        return self.embedding_model_id

    def _build_embedding_encoder(self) -> tuple[Any, Any]:
        source = self._resolve_embedding_model_source() # 원래 임베딩 지우고 새로 넣음 
        processor = AutoProcessor.from_pretrained(
            source,
            **self.embedding_processor_kwargs,
        )
        model_kwargs = dict(self.embedding_model_kwargs)
        if self.embedding_dtype is not None:
            model_kwargs.setdefault("torch_dtype", self.embedding_dtype)
        embedding_model = AutoModel.from_pretrained(source, **model_kwargs)

        device = self.model.device if torch.cuda.is_available() else torch.device("cpu")
        embedding_model.to(device)
        embedding_model.eval()
        return processor, embedding_model

    def _move_to_embedding_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        model_param = next(self.embedding_model.parameters())
        device = model_param.device
        dtype = model_param.dtype
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue

            if torch.is_tensor(value) and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        return moved

    def _coerce_embedding_tensor(
        self,
        output: Any,
        *,
        kind: str,
    ) -> torch.Tensor:
        # Some embedding backbones return a bare tensor, while others return a
        # ModelOutput wrapper. MDP3 expects tensors, so normalize the output
        # shape/type here before downstream scoring and L2 normalization.
        if torch.is_tensor(output):
            return output

        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if torch.is_tensor(value):
                return value

        last_hidden_state = getattr(output, "last_hidden_state", None)
        if torch.is_tensor(last_hidden_state):
            return last_hidden_state.mean(dim=1)

        raise RuntimeError(
            f"Failed to extract a tensor from the {kind} embedding output: "
            f"type={type(output)!r}"
        )

    def _get_image_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        frame_images = [
            Image.fromarray(frame.detach().cpu().numpy())
            for frame in frames
        ]
        image_inputs = self.embedding_processor(
            images=frame_images,
            return_tensors="pt",
        )
        image_inputs = self._move_to_embedding_device(dict(image_inputs))

        with torch.inference_mode():
            if hasattr(self.embedding_model, "get_image_features"):
                image_outputs = self.embedding_model.get_image_features(**image_inputs)
            else:
                image_outputs = self.embedding_model(**image_inputs)

        return self._coerce_embedding_tensor(image_outputs, kind="image")

    def _get_text_embedding(self, query: str) -> torch.Tensor:
        query_text = query.strip() or "Describe the most relevant frames for answering the video question."
        text_inputs = self.embedding_processor(
            text=[query_text],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = self._move_to_embedding_device(dict(text_inputs))

        with torch.inference_mode():
            if hasattr(self.embedding_model, "get_text_features"):
                text_outputs = self.embedding_model.get_text_features(**text_inputs)
            else:
                text_outputs = self.embedding_model(**text_inputs)

        text_features = self._coerce_embedding_tensor(text_outputs, kind="text")
        if text_features.ndim > 1 and text_features.shape[0] == 1:
            return text_features.squeeze(0)
        return text_features

    def _embed_for_mdp3(
        self,
        frames: torch.Tensor,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_embeddings = self._get_image_embeddings(frames)
        query_embedding = self._get_text_embedding(query)
        return frame_embeddings, query_embedding

    def answer(
        self,
        video_path: str,
        prompt: PromptInput,
        **frame_selector_kwargs: Any,
    ) -> str:
        if self.frame_selector is None:
            return super().answer(video_path=video_path, prompt=prompt, **frame_selector_kwargs)

        raw_query = ""
        if isinstance(prompt, dict):
            raw_query = str(prompt.get("query", "") or "").strip()
        selector_kwargs = {
            **self.frame_selector_kwargs,
            **frame_selector_kwargs,
            "embed_fn": self._embed_for_mdp3,
            "query": raw_query,
        }
        frame_selection = self._normalize_frame_selection_output(
            self.frame_selector(video_path=video_path, **selector_kwargs),
            video_path=video_path,
        )
        if not isinstance(frame_selection, FrameSelectionResult):
            raise TypeError("MDP3 frame selector must return a FrameSelectionResult.")

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
