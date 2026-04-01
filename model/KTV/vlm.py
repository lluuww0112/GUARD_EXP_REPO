import inspect
import warnings
from functools import partial
from typing import Any

import torch
from transformers import AutoConfig

from model.base.vlm import BaseVLM, VLMInterface


VIDEO_BACKENDS = {"video_llava", "qwen2_vl"}
VIDEO_MODEL_TYPES = {
    "video_llava",
    "qwen2_vl",
    "qwen2_5_vl",
}


class KTVSelector(VLMInterface):
    def __init__(
        self,
        model_id: str,
        frame_selector,
        token_selector=None,
        backend: str = "auto",
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        dtype: str | torch.dtype | None = None,
        token_selector_kwargs: dict[str, Any] | None = None,
        fallback_to_standard_path: bool = True,
        local_model_dir: str | None = None,
        preload_auxiliary_models: bool = True,
        **frame_selector_kwargs: Any,
    ):
        self.model_id = model_id
        self.backend = backend
        self.processor_kwargs = dict(processor_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.local_model_dir = local_model_dir
        effective_token_selector_kwargs = dict(token_selector_kwargs or {})
        if self.local_model_dir is not None:
            effective_token_selector_kwargs.setdefault(
                "local_model_dir",
                self.local_model_dir,
            )

        vlm_cls = self._select_vlm_cls(model_id=model_id, backend=backend)
        self.impl = vlm_cls(
            model_id=model_id,
            frame_selector=frame_selector,
            token_selector=token_selector,
            backend=backend,
            processor_kwargs=processor_kwargs,
            model_kwargs=model_kwargs,
            generation_kwargs=generation_kwargs,
            dtype=dtype,
            token_selector_kwargs=effective_token_selector_kwargs,
            fallback_to_standard_path=fallback_to_standard_path,
            local_model_dir=local_model_dir,
            **frame_selector_kwargs,
        )

        if preload_auxiliary_models:
            self._preload_auxiliary_models(
                frame_selector=frame_selector,
                token_selector=token_selector,
                token_selector_kwargs=effective_token_selector_kwargs,
            )

    def _unwrap_selector(
        self,
        selector: Any,
    ) -> tuple[Any | None, dict[str, Any]]:
        if selector is None:
            return None, {}
        if isinstance(selector, partial):
            return selector.func, dict(selector.keywords or {})
        return selector, {}

    def _preload_auxiliary_models(
        self,
        *,
        frame_selector: Any,
        token_selector: Any,
        token_selector_kwargs: dict[str, Any] | None,
    ) -> None:
        from . import selection as ktv_selection

        frame_fn, frame_kwargs = self._unwrap_selector(frame_selector)
        token_fn, token_kwargs = self._unwrap_selector(token_selector)
        token_kwargs.update(token_selector_kwargs or {})

        frame_name = getattr(frame_fn, "__name__", "")
        frame_module = getattr(frame_fn, "__module__", "")
        if (
            frame_module == ktv_selection.__name__
            and frame_name in {"remove_duplicate_frames", "dinov2_kmeans_keyframe_selection"}
        ):
            try:
                ktv_selection.preload_dinov2_model(
                    model_id=str(
                        frame_kwargs.get(
                            "model_id",
                            ktv_selection.DEFAULT_DINOV2_MODEL_ID,
                        )
                    ),
                    local_files_only=bool(frame_kwargs.get("local_files_only", False)),
                    dtype=frame_kwargs.get("dtype"),
                    local_model_dir=self.local_model_dir,
                )
            except Exception as exc:
                warnings.warn(
                    "Failed to preload KTV DINOv2 auxiliary model; "
                    f"it will be loaded lazily on first use. reason={exc}",
                    stacklevel=2,
                )

        token_name = getattr(token_fn, "__name__", "")
        token_module = getattr(token_fn, "__module__", "")
        if (
            token_module == ktv_selection.__name__
            and token_name == "query_conditioned_key_patch_selection"
        ):
            try:
                ktv_selection.preload_clip_model(
                    model_id=str(
                        token_kwargs.get(
                            "clip_model_id",
                            ktv_selection.DEFAULT_CLIP_MODEL_ID,
                        )
                    ),
                    local_files_only=bool(
                        token_kwargs.get("clip_local_files_only", False)
                    ),
                    local_model_dir=self.local_model_dir,
                )
            except Exception as exc:
                warnings.warn(
                    "Failed to preload KTV CLIP auxiliary model; "
                    f"it will be loaded lazily on first use. reason={exc}",
                    stacklevel=2,
                )

    def _select_vlm_cls(
        self,
        model_id: str,
        backend: str,
    ):
        if backend in VIDEO_BACKENDS:
            return BaseVLM
        if backend != "auto":
            return TrainFreeVLM

        model_type = self._resolve_model_type(model_id)
        if model_type in VIDEO_MODEL_TYPES:
            return BaseVLM
        model_id_lower = model_id.lower()
        if "video" in model_id_lower or "qwen2-vl" in model_id_lower:
            return BaseVLM
        return TrainFreeVLM

    def _resolve_model_type(self, model_id: str) -> str | None:
        config_kwargs: dict[str, Any] = {}
        for source_kwargs in (self.processor_kwargs, self.model_kwargs):
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "revision",
                "subfolder",
                "token",
                "trust_remote_code",
            ):
                if key in source_kwargs:
                    config_kwargs[key] = source_kwargs[key]

        try:
            config = AutoConfig.from_pretrained(model_id, **config_kwargs)
        except Exception:
            return None

        model_type = getattr(config, "model_type", None)
        return None if model_type is None else str(model_type).lower()

    def build_vlm(
        self,
        model_id: str,
    ):
        return self.impl.build_vlm(model_id)

    def answer(
        self,
        video_path: str,
        prompt: str,
        **frame_selector_kwargs: Any,
    ) -> str:
        effective_frame_selector_kwargs = dict(frame_selector_kwargs)
        if self.local_model_dir is not None:
            effective_frame_selector_kwargs.setdefault(
                "local_model_dir",
                self.local_model_dir,
            )
        return self.impl.answer(
            video_path=video_path,
            prompt=prompt,
            **effective_frame_selector_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.impl, name)


class TrainFreeVLM(BaseVLM):
    def _prepare_image_prompt(self, prompt: str, num_images: int) -> str:
        prompt = prompt.strip()
        if num_images <= 0:
            return prompt

        image_token = getattr(self.processor, "image_token", None)
        if not isinstance(image_token, str) or not image_token:
            raise AttributeError(
                "TrainFreeVLM requires a processor exposing `image_token`."
            )

        token_count = prompt.count(image_token)
        if token_count == 0:
            image_block = "\n".join([image_token] * num_images)
            prompt = f"{image_block}\n{prompt}"
        elif token_count != num_images:
            raise ValueError(
                "Prompt image-token count must match the sampled frame count: "
                f"tokens={token_count}, frames={num_images}."
            )

        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        if "llava" in model_type:
            if not prompt.startswith("USER:"):
                prompt = f"USER: {prompt}"
            if "ASSISTANT:" not in prompt:
                prompt = f"{prompt}\nASSISTANT:"

        return prompt

    def _processor_accepts_argument(self, arg_name: str) -> bool:
        signature = inspect.signature(self.processor.__call__)
        return arg_name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _move_inputs_to_model_device(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        device = self.model.device
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _video_tensor_to_images(self, video_tensor: torch.Tensor) -> list[Any]:
        if video_tensor.ndim != 4:
            raise ValueError(
                "Expected sampled frames with shape (T, H, W, C), "
                f"but got {tuple(video_tensor.shape)}."
            )

        frames = video_tensor.detach().cpu()
        if frames.dtype != torch.uint8:
            if (
                frames.is_floating_point()
                and frames.numel() > 0
                and float(frames.max().item()) <= 1.0
            ):
                frames = frames * 255.0
            frames = frames.clamp(0, 255).to(torch.uint8)

        return [frame.numpy() for frame in frames]

    def _build_model_inputs(
        self,
        prompt: str,
        video_tensor: torch.Tensor | None,
    ) -> dict[str, Any]:
        if video_tensor is None:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
            )
            return self._move_inputs_to_model_device(inputs)

        if not self._processor_accepts_argument("images"):
            raise TypeError(
                "TrainFreeVLM requires a processor that accepts `images=` inputs."
            )

        inputs = self.processor(
            text=prompt,
            images=self._video_tensor_to_images(video_tensor),
            return_tensors="pt",
        )
        return self._move_inputs_to_model_device(inputs)

    def _supports_layered_video_path(
        self,
        model_inputs: dict[str, Any],
    ) -> bool:
        input_ids = model_inputs.get("input_ids")
        pixel_values = model_inputs.get("pixel_values")
        if input_ids is None or pixel_values is None:
            return False

        return input_ids.shape[0] == 1

    def _get_image_feature_getter(self):
        image_feature_getter = getattr(self.model, "get_image_features", None)
        if image_feature_getter is None:
            image_feature_getter = getattr(
                getattr(self.model, "model", None),
                "get_image_features",
                None,
            )
        if image_feature_getter is None:
            raise AttributeError(
                "This image VLM implementation does not expose `get_image_features` "
                "on either the top-level model or its inner `.model`."
            )
        return image_feature_getter

    def _stack_image_features(
        self,
        image_features: Any,
    ) -> torch.Tensor:
        if isinstance(image_features, list):
            if not image_features:
                raise ValueError("Image feature list is empty.")
            return torch.stack(image_features, dim=0)

        if torch.is_tensor(image_features):
            if image_features.ndim == 2:
                return image_features.unsqueeze(0)
            if image_features.ndim == 3:
                return image_features

        raise ValueError(
            "Expected image features as a list of tensors or a tensor with shape "
            f"(T, L, D), but got {type(image_features)!r}."
        )

    def _extract_visual_token_features(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        image_feature_getter = self._get_image_feature_getter()

        feature_kwargs: dict[str, Any] = {"return_dict": True}
        for key in ("image_sizes",):
            if key in model_inputs:
                feature_kwargs[key] = model_inputs[key]

        vision_feature_layer = getattr(self.model.config, "vision_feature_layer", None)
        if vision_feature_layer is not None:
            feature_kwargs["vision_feature_layer"] = vision_feature_layer

        pixel_values = model_inputs["pixel_values"]

        cls_tokens: torch.Tensor | None = None
        used_surrogate_cls = False
        try:
            outputs = image_feature_getter(
                pixel_values=pixel_values,
                vision_feature_select_strategy="full",
                **feature_kwargs,
            )
            image_features = self._stack_image_features(outputs.pooler_output)
            if image_features.ndim != 3 or image_features.shape[1] < 2:
                raise ValueError(
                    "Projected image features with cls tokens must have shape (T, L+1, D)."
                )
            cls_tokens = image_features[:, 0, :]
            patch_tokens = image_features[:, 1:, :]
        except Exception:
            outputs = image_feature_getter(
                pixel_values=pixel_values,
                vision_feature_select_strategy=getattr(
                    self.model.config,
                    "vision_feature_select_strategy",
                    None,
                ),
                **feature_kwargs,
            )
            patch_tokens = self._stack_image_features(outputs.pooler_output)
            if patch_tokens.ndim != 3:
                raise ValueError(
                    "Expected image features with shape (T, L, D), "
                    f"but got {tuple(patch_tokens.shape)}."
                )
            cls_tokens = patch_tokens.mean(dim=1)
            used_surrogate_cls = True

        metadata = {
            "has_cls_token": False,
            "cls_tokens": cls_tokens,
            "num_frames": int(patch_tokens.shape[0]),
            "tokens_per_frame": int(patch_tokens.shape[1]),
            "used_surrogate_cls": used_surrogate_cls,
        }
        return patch_tokens, metadata

    def _call_token_selector(
        self,
        token_features: torch.Tensor,
        *,
        video_tensor: torch.Tensor,
        prompt: str,
        model_inputs: dict[str, Any],
        extraction_metadata: dict[str, Any],
    ) -> Any:
        if self.token_selector is None:
            return None

        selector_kwargs = {
            **self.token_selector_kwargs,
            "token_features": token_features,
            "cls_tokens": extraction_metadata.get("cls_tokens"),
            "has_cls_token": extraction_metadata.get("has_cls_token", False),
            "video_tensor": video_tensor,
            "prompt": prompt,
            "model_inputs": model_inputs,
            "extraction_metadata": extraction_metadata,
            "processor": self.processor,
            "model": self.model,
            "backend": self.backend,
        }

        signature = inspect.signature(self.token_selector)
        accepts_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_keyword:
            return self.token_selector(**selector_kwargs)

        filtered_kwargs = {
            name: value
            for name, value in selector_kwargs.items()
            if name in signature.parameters
        }
        return self.token_selector(**filtered_kwargs)

    def _get_video_token_id(self) -> int | None:
        for attr_name in ("image_token_id", "image_token_index"):
            value = getattr(self.model.config, attr_name, None)
            if value is not None:
                return int(value)

        tokenizer = getattr(self.processor, "tokenizer", None)
        image_token = getattr(self.processor, "image_token", None)
        if tokenizer is not None and isinstance(image_token, str) and image_token:
            token_id = tokenizer.convert_tokens_to_ids(image_token)
            if token_id is not None and token_id >= 0:
                return int(token_id)

        return None

    def answer(
        self,
        video_path: str,
        prompt: str,
        **frame_selector_kwargs: Any,
    ) -> str:
        selector_kwargs = {
            **self.frame_selector_kwargs,
            **frame_selector_kwargs,
        }
        video_tensor = self.frame_selector(video_path=video_path, **selector_kwargs)
        num_images = 0 if video_tensor is None else int(video_tensor.shape[0])
        prompt = self._prepare_image_prompt(prompt, num_images=num_images)
        model_inputs = self._build_model_inputs(prompt=prompt, video_tensor=video_tensor)

        if video_tensor is not None and self._supports_layered_video_path(model_inputs):
            try:
                return self._run_layered_video_path(
                    model_inputs=model_inputs,
                    video_tensor=video_tensor,
                    prompt=prompt,
                )
            except Exception as exc:
                self.last_token_selection_info = {
                    "applied": False,
                    "backend": self.backend,
                    "reason": str(exc),
                }
                if not self.fallback_to_standard_path:
                    raise

                warnings.warn(
                    f"Layered visual-token path failed and standard generation will be used: {exc}",
                    stacklevel=2,
                )

        else:
            self.last_token_selection_info = {
                "applied": False,
                "backend": self.backend,
                "reason": "layered_path_not_supported",
            }

        return self._run_standard_generation(model_inputs)

__all__ = ["BaseVLM", "KTVSelector", "TrainFreeVLM", "VLMInterface"]
