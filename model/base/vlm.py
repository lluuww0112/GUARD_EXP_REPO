from __future__ import annotations

import importlib.util
import inspect
from types import MethodType
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
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
)

from .selection import FrameSelectionResult, PatchSelectionResult


FrameSelector = Callable[..., torch.Tensor | FrameSelectionResult | None]
PatchSelector = Callable[..., Any]
PromptInput = str | Mapping[str, Any]


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
DEFAULT_CUDA_ATTN_IMPLEMENTATION = "eager"

BACKEND_MODEL_TYPE_HINTS = {
    "qwen2_vl": {"qwen2_vl", "qwen2_5_vl"},
    "qwen2_5_vl": {"qwen2_5_vl"},
    "qwen3_vl": {"qwen3_vl"},
}

BACKEND_SUGGESTED_MODEL_IDS = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3_vl": "Qwen/Qwen3-VL-8B-Instruct",
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
    "qwen2_vl": {
        "processor_cls": Qwen2VLProcessor,
        "model_cls": Qwen2VLForConditionalGeneration,
    },
    "qwen2_5_vl": {
        "processor_cls": AutoProcessor,
        "model_cls": Qwen2_5_VLForConditionalGeneration,
    },
    "qwen3_vl": {
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
        prompt: PromptInput,
        **frame_selector_kwargs: Any,
    ) -> str:
        raise NotImplementedError


class BaseVLM(VLMInterface):
    def __init__(
        self,
        model_id: str,
        frame_selector: FrameSelector,
        patch_selector: PatchSelector | None = None,
        backend: str = "qwen2_5_vl",
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        dtype: str | torch.dtype | None = None,
        quantization: dict[str, Any] | None = None,
        local_model_dir: str | None = None,
        **frame_selector_kwargs: Any,
    ):
        if backend not in VLM_BACKENDS:
            available = ", ".join(sorted(VLM_BACKENDS))
            raise ValueError(f"Unsupported backend: {backend}. Available: {available}")

        backend_config = VLM_BACKENDS[backend]

        self.frame_selector = frame_selector
        self.patch_selector = patch_selector
        self.frame_selector_kwargs = frame_selector_kwargs
        self.backend = backend
        self.processor_cls = backend_config["processor_cls"]
        self.model_cls = backend_config["model_cls"]
        self.processor_kwargs = dict(processor_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
        self.dtype = DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
        self.quantization = dict(quantization or {})
        self.local_model_dir = (
            Path(local_model_dir).expanduser()
            if local_model_dir is not None
            else None
        )
        self.resolved_model_source = model_id
        self.last_patch_selection_info: dict[str, Any] = {
            "applied": False,
            "backend": backend,
            "reason": "not_run",
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
            **self._build_runtime_model_kwargs(use_cuda=use_cuda),
            **quantization_kwargs,
            **self.model_kwargs,
        }
        model = self.model_cls.from_pretrained(
            model_source,
            **model_loading_kwargs,
        )

        if not use_cuda:
            model.to("cpu")

        self._apply_backend_runtime_workarounds(model)
        model.eval()
        return processor, model

    def _build_runtime_model_kwargs(
        self,
        *,
        use_cuda: bool,
    ) -> dict[str, Any]:
        if not use_cuda:
            return {}
        if "attn_implementation" in self.model_kwargs:
            return {}

        # CUDA 13 environments can fail when Transformers/PyTorch pick NVRTC-backed
        # kernels for multimodal metadata reductions, so prefer the safer eager path.
        return {"attn_implementation": DEFAULT_CUDA_ATTN_IMPLEMENTATION}

    def _resolve_dtype(self, dtype_value: Any) -> Any:
        if isinstance(dtype_value, str):
            return DTYPE_MAP.get(dtype_value.lower(), dtype_value)
        return dtype_value

    def _apply_backend_runtime_workarounds(
        self,
        model: PreTrainedModel | GenerationMixin,
    ) -> None:
        if self.backend != "qwen3_vl":
            return

        model_core = getattr(model, "model", None)
        if model_core is None or not hasattr(model_core, "get_image_features"):
            return

        def patched_get_image_features(
            core_self: Any,
            pixel_values: torch.FloatTensor,
            image_grid_thw: torch.LongTensor | None = None,
            **kwargs: Any,
        ) -> Any:
            pixel_values = pixel_values.type(core_self.visual.dtype)
            vision_output = core_self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                return_dict=True,
                **kwargs,
            )
            image_embeds = vision_output.pooler_output
            if image_grid_thw is None:
                return vision_output

            # Avoid CUDA NVRTC JIT for Tensor.prod() on CUDA metadata tensors.
            split_sizes = [
                int(grid_t) * int(grid_h) * int(grid_w)
                // (core_self.visual.spatial_merge_size**2)
                for grid_t, grid_h, grid_w in image_grid_thw.detach().cpu().tolist()
            ]
            vision_output.pooler_output = torch.split(image_embeds, split_sizes)
            return vision_output

        model_core.get_image_features = MethodType(patched_get_image_features, model_core)

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

    def _normalize_frame_selection_output(
        self,
        selection_output: torch.Tensor | FrameSelectionResult | None,
        *,
        video_path: str,
    ) -> FrameSelectionResult:
        if selection_output is None:
            return FrameSelectionResult(
                frames=None,  # type: ignore[arg-type]
                metadata={"video_path": video_path, "num_frames": 0},
            )
        if isinstance(selection_output, FrameSelectionResult):
            metadata = dict(selection_output.metadata)
            metadata.setdefault("video_path", video_path)
            return FrameSelectionResult(
                frames=selection_output.frames,
                metadata=metadata,
            )
        if torch.is_tensor(selection_output):
            return FrameSelectionResult(
                frames=selection_output,
                metadata={
                    "video_path": video_path,
                    "num_frames": int(selection_output.shape[0]),
                },
            )
        raise TypeError(
            "Frame selector output must be a torch.Tensor, FrameSelectionResult, or None."
        )

    def _normalize_prompt_input(
        self,
        prompt: PromptInput,
    ) -> tuple[str, str]:
        if isinstance(prompt, Mapping):
            system_prompt = str(prompt.get("system", "") or "").strip()
            user_prompt = str(prompt.get("user", "") or "").strip()
        else:
            system_prompt = ""
            user_prompt = str(prompt).strip()

        if not user_prompt:
            raise ValueError("Prompt must include a non-empty user prompt.")
        return system_prompt, user_prompt

    def _build_chat_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        has_video: bool,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content: list[dict[str, Any]] = []
        if has_video:
            user_content.append({"type": "video"})
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _prepare_legacy_prompt(self, prompt: str, has_video: bool) -> str:
        prompt = prompt.strip()
        if not has_video:
            return prompt

        video_token = getattr(self.processor, "video_token", None)
        if isinstance(video_token, str) and video_token and video_token not in prompt:
            prompt = f"{video_token}\n{prompt}"
        return prompt

    def _prepare_text_input(
        self,
        prompt: PromptInput,
        *,
        has_video: bool,
    ) -> str:
        system_prompt, user_prompt = self._normalize_prompt_input(prompt)
        if hasattr(self.processor, "apply_chat_template"):
            messages = self._build_chat_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                has_video=has_video,
            )
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_text = user_prompt
        if system_prompt:
            prompt_text = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
        return self._prepare_legacy_prompt(prompt_text, has_video=has_video)

    def _build_video_metadata(
        self,
        frame_selection: FrameSelectionResult,
    ) -> dict[str, Any] | None:
        video_tensor = frame_selection.frames
        if video_tensor is None:
            return None

        metadata = dict(frame_selection.metadata)
        total_num_frames = metadata.get("total_frames")
        if total_num_frames is None:
            total_num_frames = int(video_tensor.shape[0])

        fps = metadata.get("fps")
        sampled_indices = metadata.get("sampled_indices")
        if sampled_indices is None:
            sampled_indices = list(range(int(video_tensor.shape[0])))

        height = metadata.get("height")
        width = metadata.get("width")
        if (height is None or width is None) and video_tensor.ndim == 4:
            height = int(video_tensor.shape[1])
            width = int(video_tensor.shape[2])

        duration = metadata.get("duration")
        if duration is None and fps:
            duration = float(total_num_frames) / float(fps)

        return {
            "total_num_frames": int(total_num_frames),
            "fps": float(fps) if fps is not None else None,
            "frames_indices": list(sampled_indices),
            "height": int(height) if height is not None else None,
            "width": int(width) if width is not None else None,
            "duration": float(duration) if duration is not None else None,
        }

    def _build_model_inputs(
        self,
        prompt_text: str,
        video_tensor: torch.Tensor | None,
        frame_selection: FrameSelectionResult | None = None,
    ) -> dict[str, Any]:
        processor_inputs: dict[str, Any] = {
            "text": [prompt_text],
            "return_tensors": "pt",
        }
        if video_tensor is not None:
            processor_inputs["videos"] = [video_tensor]
            # The frame selector already sampled frames, so Qwen processors
            # should not re-sample the tensor again with their internal fps logic.
            processor_inputs["do_sample_frames"] = False
            if frame_selection is not None:
                video_metadata = self._build_video_metadata(frame_selection)
                if video_metadata is not None:
                    processor_inputs["video_metadata"] = [video_metadata]

        processor_attempts = [
            {
                **processor_inputs,
                "return_mm_token_type_ids": True,
            },
            dict(processor_inputs),
        ]
        if "do_sample_frames" in processor_inputs:
            processor_attempts.extend(
                [
                    {
                        key: value
                        for key, value in processor_inputs.items()
                        if key != "do_sample_frames"
                    }
                    | {"return_mm_token_type_ids": True},
                    {
                        key: value
                        for key, value in processor_inputs.items()
                        if key != "do_sample_frames"
                    },
                ]
            )

        last_error: TypeError | None = None
        inputs = None
        for attempt_kwargs in processor_attempts:
            try:
                inputs = self.processor(**attempt_kwargs)
                break
            except TypeError as exc:
                last_error = exc

        if inputs is None:
            raise last_error or RuntimeError("Failed to build processor inputs.")

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
        return value.to(device)

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

    def _prepare_generation_model_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        if self.backend != "qwen3_vl":
            return model_inputs

        video_grid_thw = model_inputs.get("video_grid_thw")
        mm_token_type_ids = model_inputs.get("mm_token_type_ids")
        if video_grid_thw is None or mm_token_type_ids is None:
            return model_inputs

        expanded_rows: list[list[int]] = []
        for row in video_grid_thw.detach().cpu().tolist():
            grid_t, grid_h, grid_w = (int(row[0]), int(row[1]), int(row[2]))
            expanded_rows.extend([[1, grid_h, grid_w]] * grid_t)

        if not expanded_rows:
            return model_inputs

        expanded_video_grid_thw = torch.tensor(
            expanded_rows,
            device=video_grid_thw.device,
            dtype=video_grid_thw.dtype,
        )
        prepared_inputs = dict(model_inputs)
        prepared_inputs["video_grid_thw"] = expanded_video_grid_thw
        return prepared_inputs

    def _run_standard_generation(
        self,
        model_inputs: dict[str, Any],
    ) -> str:
        generation_model_inputs = self._prepare_generation_model_inputs(model_inputs)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **generation_model_inputs,
                **self.generation_kwargs,
            )

        input_ids = generation_model_inputs.get("input_ids")
        prompt_length = input_ids.shape[1] if input_ids is not None else 0
        return self._decode_generation_output(output_ids, prompt_length=prompt_length)

    def _reset_multimodal_state(self) -> None:
        model_core = getattr(self.model, "model", None)
        if model_core is not None and hasattr(model_core, "rope_deltas"):
            model_core.rope_deltas = None

    def _extract_video_features(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        pixel_values_videos = model_inputs.get("pixel_values_videos")
        video_grid_thw = model_inputs.get("video_grid_thw")
        if pixel_values_videos is None or video_grid_thw is None:
            raise ValueError("Video inputs are required for patch selection.")

        if not hasattr(self.model, "get_video_features"):
            raise RuntimeError(
                f"Backend `{self.backend}` does not expose `get_video_features`."
            )

        outputs = self.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_dict=True,
        )
        video_features = outputs.pooler_output
        if isinstance(video_features, (list, tuple)):
            if len(video_features) != 1:
                raise ValueError(
                    "Patch selection currently expects a single video per prompt."
                )
            video_features = video_features[0]

        if video_features.ndim != 2:
            raise ValueError(
                "Expected merged video features with shape (N, D), "
                f"but got {tuple(video_features.shape)}."
            )

        metadata = {
            "video_grid_thw": video_grid_thw[0].detach().cpu().tolist(),
            "video_token_count": int(video_features.shape[0]),
        }
        return video_features, metadata

    def _call_patch_selector(
        self,
        video_features: torch.Tensor,
        *,
        prompt: PromptInput,
        frame_selection: FrameSelectionResult,
        model_inputs: dict[str, Any],
        extraction_metadata: dict[str, Any],
    ) -> Any:
        if self.patch_selector is None:
            return None

        selector_kwargs = {
            "video_features": video_features,
            "prompt": prompt,
            "frame_selection": frame_selection,
            "model_inputs": model_inputs,
            "extraction_metadata": extraction_metadata,
            "processor": self.processor,
            "model": self.model,
            "backend": self.backend,
        }

        signature = inspect.signature(self.patch_selector)
        accepts_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_keyword:
            return self.patch_selector(**selector_kwargs)

        filtered_kwargs = {
            name: value
            for name, value in selector_kwargs.items()
            if name in signature.parameters
        }
        return self.patch_selector(**filtered_kwargs)

    def _coerce_patch_indices(
        self,
        indices: torch.Tensor | list[int] | tuple[int, ...],
        *,
        device: torch.device,
        upper_bound: int,
    ) -> torch.Tensor:
        tensor = (
            indices
            if torch.is_tensor(indices)
            else torch.tensor(indices, device=device, dtype=torch.long)
        )
        tensor = tensor.to(device=device, dtype=torch.long).flatten()
        if tensor.numel() == 0:
            raise ValueError("Patch selector returned an empty selection.")
        if torch.any(tensor < 0) or torch.any(tensor >= upper_bound):
            raise ValueError(
                f"Patch selector indices must be within [0, {upper_bound}), got {tensor.tolist()}."
            )
        return tensor

    def _normalize_patch_selection_output(
        self,
        selection_output: Any,
        full_video_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        selected_indices: torch.Tensor | None = None
        selected_features: torch.Tensor | None = None
        metadata: dict[str, Any] = {}

        if selection_output is None:
            selected_indices = torch.arange(
                full_video_features.shape[0],
                device=full_video_features.device,
                dtype=torch.long,
            )
            selected_features = full_video_features
            return selected_indices, selected_features, metadata

        if isinstance(selection_output, PatchSelectionResult):
            selected_indices = selection_output.selected_indices
            selected_features = selection_output.selected_features
            metadata = dict(selection_output.metadata)
        elif isinstance(selection_output, dict):
            selected_indices = selection_output.get("selected_indices")
            selected_features = selection_output.get("selected_features")
            metadata = {
                key: value
                for key, value in selection_output.items()
                if key not in {"selected_indices", "selected_features"}
            }
        elif torch.is_tensor(selection_output):
            if selection_output.ndim == 1:
                selected_indices = selection_output
            elif selection_output.ndim == 2:
                selected_features = selection_output
            else:
                raise ValueError(
                    "Tensor patch selector output must be 1D indices or 2D features."
                )
        elif isinstance(selection_output, (list, tuple)) and selection_output and all(
            isinstance(item, int) for item in selection_output
        ):
            selected_indices = torch.tensor(
                selection_output,
                device=full_video_features.device,
                dtype=torch.long,
            )
        else:
            raise TypeError(
                "Unsupported patch selector output type. Expected None, PatchSelectionResult, dict, Tensor, or list[int]."
            )

        if selected_indices is None and selected_features is None:
            raise ValueError(
                "Patch selector must return `selected_indices`, `selected_features`, or both."
            )

        if selected_indices is not None:
            selected_indices = self._coerce_patch_indices(
                selected_indices,
                device=full_video_features.device,
                upper_bound=full_video_features.shape[0],
            )

        if selected_features is None:
            selected_features = full_video_features[selected_indices]
        else:
            selected_features = selected_features.to(
                device=full_video_features.device,
                dtype=full_video_features.dtype,
            )

        if selected_indices is None:
            selected_indices = torch.arange(
                selected_features.shape[0],
                device=full_video_features.device,
                dtype=torch.long,
            )

        if selected_features.ndim != 2:
            raise ValueError(
                "Selected video features must have shape (N, D), "
                f"but got {tuple(selected_features.shape)}."
            )
        if selected_indices.numel() != selected_features.shape[0]:
            raise ValueError(
                "Patch selector returned mismatched indices/features: "
                f"indices={selected_indices.numel()}, features={selected_features.shape[0]}."
            )

        return selected_indices, selected_features, metadata

    def _derive_mm_token_type_ids(
        self,
        input_ids: torch.Tensor,
        existing_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if existing_ids is not None:
            return existing_ids

        mm_token_type_ids = torch.zeros_like(input_ids)
        image_token_id = getattr(self.model.config, "image_token_id", None)
        video_token_id = getattr(self.model.config, "video_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == int(image_token_id)] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == int(video_token_id)] = 2
        return mm_token_type_ids

    def _compute_full_position_ids(
        self,
        model_inputs: dict[str, Any],
    ) -> torch.Tensor:
        model_core = getattr(self.model, "model", None)
        if model_core is None or not hasattr(model_core, "compute_3d_position_ids"):
            raise RuntimeError(
                f"Backend `{self.backend}` does not expose multimodal position-id computation."
            )

        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise ValueError(
                "`input_ids` and `attention_mask` are required for patch selection."
            )

        mm_token_type_ids = self._derive_mm_token_type_ids(
            input_ids=input_ids,
            existing_ids=model_inputs.get("mm_token_type_ids"),
        )
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        model_core.rope_deltas = None
        position_ids = model_core.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
            mm_token_type_ids=mm_token_type_ids,
        )
        if position_ids is None:
            raise RuntimeError("Failed to compute multimodal position ids for patch selection.")
        return position_ids

    def _build_generation_inputs_from_patch_selection(
        self,
        model_inputs: dict[str, Any],
        full_video_features: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_features: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids is None:
            raise ValueError("`input_ids` is required for patch selection.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        mm_token_type_ids = self._derive_mm_token_type_ids(
            input_ids=input_ids,
            existing_ids=model_inputs.get("mm_token_type_ids"),
        )
        position_ids = self._compute_full_position_ids(
            {**model_inputs, "mm_token_type_ids": mm_token_type_ids}
        )

        video_token_id = getattr(self.model.config, "video_token_id", None)
        if video_token_id is None:
            raise ValueError("Could not find the video placeholder token id from the model config.")

        video_positions = torch.nonzero(
            input_ids[0] == int(video_token_id),
            as_tuple=False,
        ).flatten()
        if int(video_positions.numel()) != int(full_video_features.shape[0]):
            raise ValueError(
                "Video placeholder count does not match merged video feature count: "
                f"tokens={int(video_positions.numel())}, features={int(full_video_features.shape[0])}."
            )

        kept_positions = video_positions[selected_indices]
        keep_mask = torch.ones(
            input_ids.shape[1],
            dtype=torch.bool,
            device=input_ids.device,
        )
        keep_mask[video_positions] = False
        keep_mask[kept_positions] = True

        pruned_input_ids = input_ids[:, keep_mask]
        pruned_attention_mask = attention_mask[:, keep_mask]
        pruned_mm_token_type_ids = mm_token_type_ids[:, keep_mask]
        pruned_position_ids = position_ids[:, :, keep_mask]
        pruned_inputs_embeds = self.model.get_input_embeddings()(pruned_input_ids)

        pruned_video_mask = pruned_input_ids[0] == int(video_token_id)
        pruned_video_features = selected_features.to(
            device=pruned_inputs_embeds.device,
            dtype=pruned_inputs_embeds.dtype,
        )
        if int(pruned_video_mask.sum().item()) != int(pruned_video_features.shape[0]):
            raise ValueError(
                "Pruned video placeholder count does not match selected feature count: "
                f"tokens={int(pruned_video_mask.sum().item())}, "
                f"features={int(pruned_video_features.shape[0])}."
            )

        pruned_inputs_embeds[0, pruned_video_mask] = pruned_video_features
        generation_inputs = {
            "input_ids": pruned_input_ids,
            "inputs_embeds": pruned_inputs_embeds,
            "attention_mask": pruned_attention_mask,
            "position_ids": pruned_position_ids,
            "mm_token_type_ids": pruned_mm_token_type_ids,
        }
        metadata = {
            "original_video_tokens": int(full_video_features.shape[0]),
            "selected_video_tokens": int(pruned_video_features.shape[0]),
            "input_length_before": int(input_ids.shape[1]),
            "input_length_after": int(pruned_input_ids.shape[1]),
        }
        return generation_inputs, metadata

    def _run_patch_selection_generation(
        self,
        *,
        prompt: PromptInput,
        frame_selection: FrameSelectionResult,
        model_inputs: dict[str, Any],
    ) -> str:
        full_video_features, extraction_metadata = self._extract_video_features(model_inputs)
        selection_output = self._call_patch_selector(
            video_features=full_video_features,
            prompt=prompt,
            frame_selection=frame_selection,
            model_inputs=model_inputs,
            extraction_metadata=extraction_metadata,
        )
        selected_indices, selected_features, selector_metadata = (
            self._normalize_patch_selection_output(
                selection_output=selection_output,
                full_video_features=full_video_features,
            )
        )
        generation_inputs, pruning_metadata = self._build_generation_inputs_from_patch_selection(
            model_inputs=model_inputs,
            full_video_features=full_video_features,
            selected_indices=selected_indices,
            selected_features=selected_features,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **generation_inputs,
                **self.generation_kwargs,
            )

        prompt_length = generation_inputs["input_ids"].shape[1]
        self.last_patch_selection_info = {
            "applied": True,
            "backend": self.backend,
            **extraction_metadata,
            **pruning_metadata,
            "selector_output_keys": sorted(selector_metadata.keys()),
        }
        return self._decode_generation_output(output_ids, prompt_length=prompt_length)

    def answer(
        self,
        video_path: str,
        prompt: PromptInput,
        **frame_selector_kwargs: Any,
    ) -> str:
        selector_kwargs = {
            **self.frame_selector_kwargs,
            **frame_selector_kwargs,
        }
        frame_selection = self._normalize_frame_selection_output(
            self.frame_selector(video_path=video_path, **selector_kwargs),
            video_path=video_path,
        )
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
