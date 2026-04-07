from __future__ import annotations

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from model.base import VLMInterface


def load_prompt(invoke_config: DictConfig) -> dict[str, str]:
    prompt_file = invoke_config.get("prompt_file")
    if not prompt_file:
        raise ValueError("`invoke.prompt_file` must be provided in the config.")

    prompt_path = Path(to_absolute_path(str(prompt_file)))
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    query_file = invoke_config.get("query_file")
    if not query_file:
        raise ValueError("`invoke.query_file` must be provided in the config.")

    query_path = Path(to_absolute_path(str(query_file)))
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    query = query_path.read_text(encoding="utf-8").strip()
    if not query:
        raise ValueError(f"Query file is empty: {query_path}")

    sections = {"system": [], "user": []}
    current_section: str | None = None
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[SYSTEM]":
            current_section = "system"
            continue
        if stripped == "[USER]":
            current_section = "user"
            continue
        if current_section is not None:
            sections[current_section].append(line)

    system_prompt = "\n".join(sections["system"]).strip()
    user_prompt = "\n".join(sections["user"]).strip()
    if not user_prompt:
        raise ValueError(f"`[USER]` section is missing in prompt file: {prompt_path}")

    user_prompt = user_prompt.format(query=str(query))
    return {
        "system": system_prompt,
        "user": user_prompt,
        "query": query,
    }


def build_vlm(config: DictConfig) -> VLMInterface:
    try:
        vlm = instantiate(config.vlm)
    except Exception as exc:
        message = str(exc)
        if "ignore_mismatched_sizes" in message:
            model_id = config.vlm.get("model_id")
            backend = config.vlm.get("backend")
            raise RuntimeError(
                "VLM checkpoint loading failed due to backend/checkpoint incompatibility. "
                f"backend=`{backend}`, model_id=`{model_id}`. "
                "Set `invoke.quiet_model_loading=false` to inspect the full Transformers "
                "loading report, and use a checkpoint architecture that matches the backend."
            ) from exc
        raise
    if not isinstance(vlm, VLMInterface):
        raise TypeError("Instantiated `vlm` does not implement VLMInterface.")
    return vlm


@contextmanager
def suppress_model_loading_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    logger_names = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "transformers",
    )
    previous_levels = {
        name: logging.getLogger(name).level
        for name in logger_names
    }
    progress_bars_were_disabled = are_progress_bars_disabled()

    disable_progress_bars()
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
        if not progress_bars_were_disabled:
            enable_progress_bars()


def summarize_config(config: DictConfig) -> dict[str, Any]:
    frame_selection = OmegaConf.to_container(config.frame_selection, resolve=True)
    patch_selection_cfg = config.get("patch_selection")
    patch_selection = (
        OmegaConf.to_container(patch_selection_cfg, resolve=True)
        if patch_selection_cfg is not None
        else None
    )
    generation_kwargs = OmegaConf.to_container(
        config.vlm.get("generation_kwargs"),
        resolve=True,
    )
    invoke_config = OmegaConf.to_container(config.invoke, resolve=True)

    return {
        "frame_selection": frame_selection,
        "patch_selection": patch_selection,
        "invoke": invoke_config,
        "model_id": config.vlm.get("model_id"),
        "backend": config.vlm.get("backend"),
        "dtype": config.vlm.get("dtype"),
        "local_model_dir": config.vlm.get("local_model_dir"),
        "generation_kwargs": generation_kwargs,
    }


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig) -> None:
    invoke_config = config.get("invoke")
    if invoke_config is None:
        raise ValueError("`invoke` section must be provided in the config.")

    video_value = invoke_config.get("video_path")
    if not video_value:
        raise ValueError("`invoke.video_path` must be provided in the config.")

    video_path = Path(to_absolute_path(str(video_value)))
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    prompt = load_prompt(invoke_config)
    with suppress_model_loading_output(
        enabled=invoke_config.get("quiet_model_loading", True),
    ):
        vlm = build_vlm(config)

    if invoke_config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(config, resolve=True).strip())
        print()

    summary = summarize_config(config)
    print("=== Inference Setup ===")
    print(f"Video       : {video_path}")
    print(f"Model       : {summary['model_id']}")
    print(f"Backend     : {summary['backend']}")
    print(f"DType       : {summary['dtype']}")
    if summary.get("local_model_dir") is not None:
        print(f"Local Store : {summary['local_model_dir']}")
    resolved_model_source = getattr(vlm, "resolved_model_source", None)
    if resolved_model_source and str(resolved_model_source) != str(summary["model_id"]):
        print(f"Load Source : {resolved_model_source}")
    frame_target = summary["frame_selection"]["_target_"]
    print(f"Frame Layer : {frame_target}")
    patch_selection = summary.get("patch_selection")
    if patch_selection is not None:
        print(f"Patch Layer : {patch_selection['_target_']}")
    print(f"Prompt File : {summary['invoke']['prompt_file']}")
    print(f"Query File  : {summary['invoke']['query_file']}")
    print()

    start_time = time.perf_counter()
    response = vlm.answer(
        video_path=str(video_path),
        prompt=prompt,
    )
    elapsed = time.perf_counter() - start_time

    print("=== Response ===")
    print(response.strip())
    print()
    print("=== Timing ===")
    print(f"Elapsed     : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
