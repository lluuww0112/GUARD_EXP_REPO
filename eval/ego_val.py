from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import requests
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


DEFAULT_URL = "https://validation-server.onrender.com/api/upload/"
DEFAULT_TIMEOUT_SECONDS = 120


def _load_payload(input_path: Path) -> Any:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number} in {input_path}: {exc}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Each JSONL row must be an object, got {type(item).__name__} "
                        f"at line {line_number}."
                    )
                items.append(item)
        return items

    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_prediction(value: Any, *, q_uid: str) -> int:
    if value is None:
        raise ValueError(f"Missing prediction for q_uid={q_uid}.")
    if isinstance(value, bool):
        raise ValueError(f"Boolean prediction is not valid for q_uid={q_uid}.")
    if isinstance(value, int):
        return int(value)

    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty prediction for q_uid={q_uid}.")
    if text.isdigit():
        return int(text)
    raise ValueError(
        f"Prediction must be an integer option index for q_uid={q_uid}, got {value!r}."
    )


def _normalize_submission_payload(raw_payload: Any) -> dict[str, int]:
    if isinstance(raw_payload, dict):
        normalized: dict[str, int] = {}
        for key, value in raw_payload.items():
            q_uid = str(key)
            if isinstance(value, dict):
                prediction = value.get("prediction", value.get("answer"))
            else:
                prediction = value
            normalized[q_uid] = _coerce_prediction(prediction, q_uid=q_uid)
        return normalized

    if isinstance(raw_payload, list):
        normalized: dict[str, int] = {}
        for index, item in enumerate(raw_payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"List payload entries must be objects, got {type(item).__name__} "
                    f"at index {index}."
                )
            q_uid = item.get("q_uid") or item.get("quid") or item.get("uid")
            if q_uid is None:
                raise ValueError(f"Missing q_uid in list payload at index {index}.")
            q_uid = str(q_uid)
            prediction = item.get("prediction", item.get("answer"))
            normalized[q_uid] = _coerce_prediction(prediction, q_uid=q_uid)
        return normalized

    raise ValueError(
        "Unsupported prediction file format. Expected JSON object, JSON list, or JSONL."
    )


def load_submission_payload(input_path: str | Path) -> dict[str, int]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    raw_payload = _load_payload(path)
    normalized_payload = _normalize_submission_payload(raw_payload)
    if not normalized_payload:
        raise ValueError(f"No predictions found in: {path}")
    return normalized_payload


def send_post_request(
    payload: dict[str, int],
    *,
    url: str = DEFAULT_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> requests.Response:
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    return response


def _to_abs_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(to_absolute_path(str(path_value))).expanduser().resolve()


@hydra.main(version_base=None, config_path="../config", config_name="val")
def main(config: DictConfig) -> None:
    val_config = config.get("egoschema_val")
    if val_config is None:
        raise ValueError("`egoschema_val` section must be provided in config/val.yaml.")

    input_file = _to_abs_path(str(val_config.get("input_file")) if val_config.get("input_file") else None)
    if input_file is None:
        raise ValueError("`egoschema_val.input_file` must be provided.")

    url = str(val_config.get("url") or DEFAULT_URL)
    timeout = int(val_config.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    dry_run = bool(val_config.get("dry_run", False))

    payload = load_submission_payload(input_file)
    print(f"Loaded predictions : {len(payload)}")

    if config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(config, resolve=True).strip())
        print()

    if dry_run:
        print("Dry run enabled; upload skipped.")
        return

    response = send_post_request(
        payload,
        url=url,
        timeout=timeout,
    )
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content:\n{response.text}")


if __name__ == "__main__":
    main()
