import inspect
import importlib.util
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationMixin,
    PreTrainedModel,
    ProcessorMixin,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)


FrameSelector = Callable[..., torch.Tensor]
TokenSelector = Callable[..., Any]


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

QUANTIZED_DTYPE_KEYS = ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage")
DEFAULT_VISION_SKIP_MODULES = (
    "vision_tower",
    "vision_model",
    "visual",
    "image_tower",
    "video_tower",
)

BACKEND_MODEL_TYPE_HINTS = {
    "video_llava": {"video_llava"},
    "qwen2_vl": {"qwen2_vl", "qwen2_5_vl"},
}

BACKEND_SUGGESTED_MODEL_IDS = {
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
}

SNAPSHOT_DOWNLOAD_KEYS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "token",
)


def _sanitize_model_id_for_path(model_id: str) -> str:
    safe = model_id.replace("\\", "/").strip()
    safe = safe.replace("/", "__").replace(":", "_").replace("@", "_")
    return "".join(
        char if (char.isalnum() or char in {"-", "_", "."}) else "_"
        for char in safe
    )



VLM_BACKENDS = {
    "video_llava": {
        "processor_cls": VideoLlavaProcessor,
        "model_cls": VideoLlavaForConditionalGeneration,
    },
    "qwen2_vl": {
        "processor_cls": Qwen2VLProcessor,
        "model_cls": Qwen2VLForConditionalGeneration,
    },
    "qwen2_5_vl":{
        "processor_cls": AutoProcessor,
        "model_cls": Qwen2_5_VLForConditionalGeneration,
    },
    "qwen3_vl":{
        "processor_cls": AutoProcessor,
        "model_cls": Qwen3VLForConditionalGeneration,
    },
    "auto": {
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForImageTextToText,
    },
}


class VLMInterface(ABC):
    @abstractmethod
    def build_vlm(
        self,
        model_id: str,
    ) -> tuple[ProcessorMixin, PreTrainedModel | GenerationMixin]:
        raise NotImplementedError

    @abstractmethod
    def answer(
        self,
        video_path: str,
        prompt: str,
        **frame_selector_kwargs: Any,
    ) -> str:
        raise NotImplementedError


class BaseVLM(VLMInterface):
    def __init__(
        self,
        model_id: str,
        frame_selector: FrameSelector,
        token_selector: TokenSelector | None = None,
        backend: str = "auto",
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        dtype: str | torch.dtype | None = None,
        quantization: dict[str, Any] | None = None,
        token_selector_kwargs: dict[str, Any] | None = None,
        fallback_to_standard_path: bool = True,
        local_model_dir: str | None = None,
        **frame_selector_kwargs: Any,
    ):
        if backend not in VLM_BACKENDS:
            available = ", ".join(sorted(VLM_BACKENDS))
            raise ValueError(f"Unsupported backend: {backend}. Available: {available}")

        backend_config = VLM_BACKENDS[backend]

        self.frame_selector = frame_selector
        self.token_selector = token_selector
        self.frame_selector_kwargs = frame_selector_kwargs
        self.backend = backend
        self.processor_cls = backend_config["processor_cls"]
        self.model_cls = backend_config["model_cls"]
        self.processor_kwargs = dict(processor_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
        self.dtype = DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
        self.quantization = dict(quantization or {})
        self.token_selector_kwargs = dict(token_selector_kwargs or {})
        self.fallback_to_standard_path = fallback_to_standard_path
        self.local_model_dir = (
            Path(local_model_dir).expanduser()
            if local_model_dir is not None
            else None
        )
        self.resolved_model_source = model_id
        self.last_token_selection_info: dict[str, Any] = {
            "applied": False,
            "backend": backend,
        }

        self.processor: ProcessorMixin
        self.model: PreTrainedModel | GenerationMixin
        self.processor, self.model = self.build_vlm(model_id)

    def build_vlm(
        self,
        model_id: str,
    ) -> tuple[ProcessorMixin, PreTrainedModel | GenerationMixin]:
        use_cuda = torch.cuda.is_available()
        dtype = self.dtype or (torch.float16 if use_cuda else torch.float32)
        model_source = self._resolve_model_source(model_id)
        self._validate_backend_model_type(model_source)
        quantization_kwargs = self._build_quantization_kwargs(use_cuda=use_cuda)

        processor = self.processor_cls.from_pretrained(
            model_source,
            **self.processor_kwargs,
        )
        model_loading_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if use_cuda else None,
            **quantization_kwargs,
            **self.model_kwargs,
        }
        model = self.model_cls.from_pretrained(
            model_source,
            **model_loading_kwargs,
        )

        if not use_cuda:
            model.to("cpu")

        model.eval()
        return processor, model

    def _resolve_dtype(self, dtype_value: Any) -> Any:
        if isinstance(dtype_value, str):
            return DTYPE_MAP.get(dtype_value.lower(), dtype_value)
        return dtype_value

    def _build_quantization_kwargs(
        self,
        *,
        use_cuda: bool,
    ) -> dict[str, Any]:
        if not self.quantization:
            return {}
        if not bool(self.quantization.get("enabled", False)):
            return {}
        if "quantization_config" in self.model_kwargs:
            raise ValueError(
                "Do not set both `vlm.quantization` and `vlm.model_kwargs.quantization_config`."
            )

        if not use_cuda:
            raise RuntimeError(
                "Quantization requires CUDA in this project. "
                "Disable `vlm.quantization.enabled` when running on CPU."
            )
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "`bitsandbytes` is required for quantization. "
                "Install it and retry."
            )

        mode = str(self.quantization.get("mode", "")).lower().strip()
        if mode not in {"4bit", "8bit"}:
            raise ValueError("`vlm.quantization.mode` must be one of: `4bit`, `8bit`.")

        bnb_kwargs = dict(self.quantization.get("kwargs") or {})
        for key in QUANTIZED_DTYPE_KEYS:
            if key in bnb_kwargs:
                bnb_kwargs[key] = self._resolve_dtype(bnb_kwargs[key])

        skip_modules = list(self.quantization.get("skip_modules") or [])
        existing_skip_modules = bnb_kwargs.get("llm_int8_skip_modules") or []
        if isinstance(existing_skip_modules, (list, tuple)):
            skip_modules.extend(existing_skip_modules)
        skip_vision_encoder = bool(self.quantization.get("skip_vision_encoder", True))
        if skip_vision_encoder:
            skip_modules = [*skip_modules, *DEFAULT_VISION_SKIP_MODULES]
        if skip_modules:
            bnb_kwargs["llm_int8_skip_modules"] = sorted(set(skip_modules))

        bnb_kwargs["load_in_4bit"] = mode == "4bit"
        bnb_kwargs["load_in_8bit"] = mode == "8bit"
        quantization_config = BitsAndBytesConfig(**bnb_kwargs)
        return {"quantization_config": quantization_config}

    def _extract_snapshot_download_kwargs(self) -> dict[str, Any]:
        download_kwargs: dict[str, Any] = {}
        for source_kwargs in (self.processor_kwargs, self.model_kwargs):
            for key in SNAPSHOT_DOWNLOAD_KEYS:
                if key in source_kwargs:
                    download_kwargs[key] = source_kwargs[key]
        return download_kwargs

    def _as_existing_local_path(
        self,
        model_id: str,
    ) -> Path | None:
        model_path = Path(model_id).expanduser()
        if model_path.exists():
            return model_path.resolve()
        return None

    def _build_local_model_path(
        self,
        model_id: str,
    ) -> Path | None:
        if self.local_model_dir is None:
            return None

        revision = self._extract_snapshot_download_kwargs().get("revision")
        revision_suffix = ""
        if revision:
            safe_revision = _sanitize_model_id_for_path(str(revision))
            revision_suffix = f"__rev_{safe_revision}"

        return self.local_model_dir / (
            f"{_sanitize_model_id_for_path(model_id)}{revision_suffix}"
        )

    def _is_local_snapshot_ready(
        self,
        local_model_path: Path,
    ) -> bool:
        if not local_model_path.exists() or not local_model_path.is_dir():
            return False
        return (local_model_path / "config.json").exists()

    def _resolve_model_source(
        self,
        model_id: str,
    ) -> str:
        existing_local = self._as_existing_local_path(model_id)
        if existing_local is not None:
            self.resolved_model_source = str(existing_local)
            return self.resolved_model_source

        local_model_path = self._build_local_model_path(model_id)
        if local_model_path is None:
            self.resolved_model_source = model_id
            return self.resolved_model_source

        if self._is_local_snapshot_ready(local_model_path):
            self.resolved_model_source = str(local_model_path)
            return self.resolved_model_source

        download_kwargs = self._extract_snapshot_download_kwargs()
        try:
            local_model_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_model_path),
                **download_kwargs,
            )
            if self._is_local_snapshot_ready(local_model_path):
                self.resolved_model_source = str(local_model_path)
                return self.resolved_model_source
        except Exception as exc:
            warnings.warn(
                "Failed to mirror model snapshot into local_model_dir; "
                f"falling back to default Hugging Face loading path. reason={exc}",
                stacklevel=2,
            )

        self.resolved_model_source = model_id
        return self.resolved_model_source

    def _extract_config_load_kwargs(self) -> dict[str, Any]:
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
        return config_kwargs

    def _validate_backend_model_type(
        self,
        model_id: str,
    ) -> None:
        expected_model_types = BACKEND_MODEL_TYPE_HINTS.get(self.backend)
        if not expected_model_types:
            return

        config_kwargs = self._extract_config_load_kwargs()
        try:
            config = AutoConfig.from_pretrained(model_id, **config_kwargs)
        except Exception:
            return

        model_type = str(getattr(config, "model_type", "")).lower()
        if not model_type or model_type in expected_model_types:
            return

        expected_display = ", ".join(sorted(expected_model_types))
        suggestion = BACKEND_SUGGESTED_MODEL_IDS.get(self.backend)
        suggestion_text = (
            f" For example, use `{suggestion}`."
            if suggestion is not None
            else ""
        )
        raise ValueError(
            f"Incompatible backend/model pair: backend `{self.backend}` expects "
            f"`config.model_type` in {{{expected_display}}}, but `{model_id}` has "
            f"`config.model_type={model_type}`.{suggestion_text}"
        )

    def _prepare_prompt(self, prompt: str, has_video: bool) -> str:
        prompt = prompt.strip()
        if not has_video:
            return prompt

        video_token = getattr(self.processor, "video_token", None)
        if isinstance(video_token, str) and video_token and video_token not in prompt:
            prompt = f"{video_token}\n{prompt}"

        if self.backend == "video_llava":
            if not prompt.startswith("USER:"):
                prompt = f"USER: {prompt}"
            if "ASSISTANT:" not in prompt:
                prompt = f"{prompt}\nASSISTANT:"

        return prompt

    def _build_model_inputs(
        self,
        prompt: str,
        video_tensor: torch.Tensor | None,
    ) -> dict[str, Any]:
        inputs = self.processor(
            text=prompt,
            videos=video_tensor,
            return_tensors="pt",
        )

        device = self.model.device
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _decode_generation_output(
        self,
        output_ids: torch.Tensor,
        prompt_length: int,
    ) -> str:
        if output_ids.ndim == 2 and output_ids.shape[1] >= prompt_length:
            output_ids = output_ids[:, prompt_length:]

        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

    def _run_standard_generation(
        self,
        model_inputs: dict[str, Any],
    ) -> str:
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                **self.generation_kwargs,
            )

        input_ids = model_inputs.get("input_ids")
        prompt_length = input_ids.shape[1] if input_ids is not None else 0
        return self._decode_generation_output(output_ids, prompt_length=prompt_length)

    def _supports_layered_video_path(
        self,
        model_inputs: dict[str, Any],
    ) -> bool:
        if self.backend != "video_llava":
            return False

        input_ids = model_inputs.get("input_ids")
        pixel_values_videos = model_inputs.get("pixel_values_videos")
        if input_ids is None or pixel_values_videos is None:
            return False

        return input_ids.shape[0] == 1

    def _extract_visual_token_features(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.backend != "video_llava":
            raise RuntimeError(
                f"Layered visual-token path is not implemented for backend `{self.backend}`."
            )

        video_feature_getter = getattr(self.model, "get_video_features", None)
        if video_feature_getter is None:
            video_feature_getter = getattr(getattr(self.model, "model", None), "get_video_features", None)
        if video_feature_getter is None:
            raise AttributeError(
                "This Video-LLaVA implementation does not expose `get_video_features` "
                "on either the top-level model or its inner `.model`."
            )

        outputs = video_feature_getter(
            pixel_values_videos=model_inputs["pixel_values_videos"],
            return_dict=True,
        )
        video_features = outputs.pooler_output
        if video_features.ndim != 3:
            raise ValueError(
                "Expected video features with shape (T, L, D), "
                f"but got {tuple(video_features.shape)}."
            )

        metadata = {
            "has_cls_token": True,
            "num_frames": int(video_features.shape[0]),
            "tokens_per_frame": int(video_features.shape[1]),
        }
        return video_features, metadata

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

    def _normalize_token_selection_output(
        self,
        selection_output: Any,
        full_token_features: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        if selection_output is None:
            full_indices = [
                torch.arange(
                    full_token_features.shape[1],
                    device=full_token_features.device,
                    dtype=torch.long,
                )
                for _ in range(full_token_features.shape[0])
            ]
            full_tokens = [
                full_token_features[frame_idx]
                for frame_idx in range(full_token_features.shape[0])
            ]
            return full_indices, full_tokens, {}

        if isinstance(selection_output, dict):
            selected_indices = selection_output.get("selected_indices")
            selected_tokens = selection_output.get("selected_tokens")

            if selected_indices is None and selected_tokens is None:
                raise ValueError(
                    "Token selector output dict must include `selected_indices` "
                    "or `selected_tokens`."
                )

            if selected_indices is None:
                selected_indices = [
                    torch.arange(
                        frame_tokens.shape[0],
                        device=frame_tokens.device,
                        dtype=torch.long,
                    )
                    for frame_tokens in selected_tokens
                ]
                warnings.warn(
                    "Token selector output did not include `selected_indices`; "
                    "falling back to the first k token positions per frame.",
                    stacklevel=2,
                )

            if selected_tokens is None:
                selected_tokens = [
                    full_token_features[frame_idx, frame_indices]
                    for frame_idx, frame_indices in enumerate(selected_indices)
                ]

            return selected_indices, selected_tokens, dict(selection_output)

        if torch.is_tensor(selection_output):
            if selection_output.ndim != 3:
                raise ValueError(
                    "Tensor token selector output must have shape (T, K, D), "
                    f"but got {tuple(selection_output.shape)}."
                )
            selected_tokens = [
                selection_output[frame_idx]
                for frame_idx in range(selection_output.shape[0])
            ]
            selected_indices = [
                torch.arange(
                    frame_tokens.shape[0],
                    device=frame_tokens.device,
                    dtype=torch.long,
                )
                for frame_tokens in selected_tokens
            ]
            warnings.warn(
                "Tensor token selector output did not include source indices; "
                "falling back to the first k token positions per frame.",
                stacklevel=2,
            )
            return selected_indices, selected_tokens, {}

        if isinstance(selection_output, list) and all(
            torch.is_tensor(item) for item in selection_output
        ):
            selected_tokens = selection_output
            selected_indices = [
                torch.arange(
                    frame_tokens.shape[0],
                    device=frame_tokens.device,
                    dtype=torch.long,
                )
                for frame_tokens in selected_tokens
            ]
            warnings.warn(
                "List token selector output did not include source indices; "
                "falling back to the first k token positions per frame.",
                stacklevel=2,
            )
            return selected_indices, selected_tokens, {}

        raise TypeError(
            "Unsupported token selector output type. Expected dict, Tensor, or list[Tensor]."
        )

    def _select_visual_tokens(
        self,
        token_features: torch.Tensor,
        *,
        video_tensor: torch.Tensor,
        prompt: str,
        model_inputs: dict[str, Any],
        extraction_metadata: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        selection_output = self._call_token_selector(
            token_features=token_features,
            video_tensor=video_tensor,
            prompt=prompt,
            model_inputs=model_inputs,
            extraction_metadata=extraction_metadata,
        )
        return self._normalize_token_selection_output(
            selection_output=selection_output,
            full_token_features=token_features,
        )

    def _project_visual_tokens(
        self,
        selected_tokens: list[torch.Tensor],
        extraction_metadata: dict[str, Any],
    ) -> list[torch.Tensor]:
        _ = extraction_metadata
        return selected_tokens

    def _get_video_token_id(self) -> int | None:
        for attr_name in ("video_token_index", "video_token_id"):
            value = getattr(self.model.config, attr_name, None)
            if value is not None:
                return int(value)
        return None

    def _build_generation_inputs_from_selected_tokens(
        self,
        model_inputs: dict[str, Any],
        full_token_features: torch.Tensor,
        selected_indices: list[torch.Tensor],
        projected_tokens: list[torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("`input_ids` is required to rebuild pruned multimodal inputs.")

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        video_token_id = self._get_video_token_id()
        if video_token_id is None:
            raise ValueError("Could not find the video placeholder token id from the model config.")

        video_positions = torch.nonzero(
            input_ids[0] == video_token_id,
            as_tuple=False,
        ).flatten()

        num_frames, tokens_per_frame, _ = full_token_features.shape
        expected_video_tokens = num_frames * tokens_per_frame
        if int(video_positions.numel()) != expected_video_tokens:
            raise ValueError(
                "Video placeholder count does not match extracted token count: "
                f"tokens={int(video_positions.numel())}, features={expected_video_tokens}."
            )

        frame_positions = video_positions.view(num_frames, tokens_per_frame)
        kept_positions = []
        flattened_tokens = []
        keep_counts = []
        for frame_idx, frame_indices in enumerate(selected_indices):
            frame_indices = frame_indices.to(device=frame_positions.device, dtype=torch.long)
            kept_positions.append(frame_positions[frame_idx, frame_indices])
            frame_tokens = projected_tokens[frame_idx].to(
                device=full_token_features.device,
                dtype=full_token_features.dtype,
            )
            flattened_tokens.append(frame_tokens)
            keep_counts.append(int(frame_tokens.shape[0]))

        keep_mask = torch.ones(
            input_ids.shape[1],
            dtype=torch.bool,
            device=input_ids.device,
        )
        keep_mask[video_positions] = False
        keep_mask[torch.cat(kept_positions, dim=0)] = True

        pruned_input_ids = input_ids[:, keep_mask]
        pruned_attention_mask = attention_mask[:, keep_mask]
        pruned_inputs_embeds = self.model.get_input_embeddings()(pruned_input_ids)

        video_mask = pruned_input_ids[0] == video_token_id
        pruned_video_tokens = torch.cat(flattened_tokens, dim=0).to(
            device=pruned_inputs_embeds.device,
            dtype=pruned_inputs_embeds.dtype,
        )
        if int(video_mask.sum().item()) != pruned_video_tokens.shape[0]:
            raise ValueError(
                "Pruned video placeholder count does not match selected token count: "
                f"tokens={int(video_mask.sum().item())}, "
                f"features={pruned_video_tokens.shape[0]}."
            )

        pruned_inputs_embeds[0, video_mask] = pruned_video_tokens
        generation_inputs = {
            "inputs_embeds": pruned_inputs_embeds,
            "attention_mask": pruned_attention_mask,
        }
        metadata = {
            "original_video_tokens": expected_video_tokens,
            "selected_video_tokens": int(pruned_video_tokens.shape[0]),
            "keep_counts": keep_counts,
            "input_length_before": int(input_ids.shape[1]),
            "input_length_after": int(pruned_input_ids.shape[1]),
        }
        return generation_inputs, metadata

    def _run_layered_video_path(
        self,
        model_inputs: dict[str, Any],
        *,
        video_tensor: torch.Tensor,
        prompt: str,
    ) -> str:
        token_features, extraction_metadata = self._extract_visual_token_features(model_inputs)
        selected_indices, selected_tokens, selector_metadata = self._select_visual_tokens(
            token_features=token_features,
            video_tensor=video_tensor,
            prompt=prompt,
            model_inputs=model_inputs,
            extraction_metadata=extraction_metadata,
        )
        projected_tokens = self._project_visual_tokens(
            selected_tokens=selected_tokens,
            extraction_metadata=extraction_metadata,
        )
        generation_inputs, pruning_metadata = self._build_generation_inputs_from_selected_tokens(
            model_inputs=model_inputs,
            full_token_features=token_features,
            selected_indices=selected_indices,
            projected_tokens=projected_tokens,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **generation_inputs,
                **self.generation_kwargs,
            )

        prompt_length = generation_inputs["attention_mask"].shape[1]
        self.last_token_selection_info = {
            "applied": True,
            "backend": self.backend,
            **pruning_metadata,
            "selector_output_keys": (
                sorted(selector_metadata.keys())
                if isinstance(selector_metadata, dict)
                else None
            ),
        }
        return self._decode_generation_output(output_ids, prompt_length=prompt_length)

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
        prompt = self._prepare_prompt(prompt, has_video=video_tensor is not None)
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
