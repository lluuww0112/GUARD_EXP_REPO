from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationMixin,
    PreTrainedModel,
    ProcessorMixin,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)


FrameSelector = Callable[..., torch.Tensor]


DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


VLM_BACKENDS = {
    "video_llava": {
        "processor_cls": VideoLlavaProcessor,
        "model_cls": VideoLlavaForConditionalGeneration,
    },
    "qwen2_vl": {
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForVision2Seq,
    },
    "auto": {
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForVision2Seq,
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
        backend: str = "auto",
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        dtype: str | torch.dtype | None = None,
        **frame_selector_kwargs: Any,
    ):
        if backend not in VLM_BACKENDS:
            available = ", ".join(sorted(VLM_BACKENDS))
            raise ValueError(f"Unsupported backend: {backend}. Available: {available}")

        backend_config = VLM_BACKENDS[backend]

        self.frame_selector = frame_selector
        self.frame_selector_kwargs = frame_selector_kwargs
        self.processor_cls = backend_config["processor_cls"]
        self.model_cls = backend_config["model_cls"]
        self.processor_kwargs = processor_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.generation_kwargs = dict(generation_kwargs or {})
        self.dtype = DTYPE_MAP[dtype]

        self.processor: ProcessorMixin
        self.model: PreTrainedModel | GenerationMixin
        self.processor, self.model = self.build_vlm(model_id)

    def build_vlm(
        self,
        model_id: str,
    ) -> tuple[ProcessorMixin, PreTrainedModel | GenerationMixin]:
        use_cuda = torch.cuda.is_available()
        dtype = self.dtype or (torch.float16 if use_cuda else torch.float32)

        processor = self.processor_cls.from_pretrained(
            model_id,
            **self.processor_kwargs,
        )
        model = self.model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            **self.model_kwargs,
        )

        if not use_cuda:
            model.to("cpu")

        model.eval()
        return processor, model
    

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

        inputs = self.processor(
            text=prompt,
            videos=video_tensor,
            return_tensors="pt",
        )

        device = self.model.device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs,
            )

        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
