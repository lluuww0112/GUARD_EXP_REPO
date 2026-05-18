from __future__ import annotations

import argparse
import base64
import html
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from PIL import Image

from test import show_selected_patch as viz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one selected-patch frame as a standalone SVG.",
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "DDPS.yaml")
    parser.add_argument("--patch-selection-key", type=str, default=viz.DEFAULT_PATCH_SELECTION_KEY)
    parser.add_argument("--video-path", type=Path, default=viz.DEFAULT_DIRECT_VIDEO_PATH)
    parser.add_argument("--query-text", type=str, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument(
        "--view",
        choices=("clip", "qwen"),
        default="clip",
        help="clip exports raw CLIP patch grid; qwen exports merged final-token grid.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--clip-batch-size", type=int, default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def _diagnostic_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        patch_selection_key=args.patch_selection_key,
        dataset="direct",
        sample_index=0,
        sample_id=None,
        task_name=None,
        video_path=args.video_path,
        query_file=None,
        query_text=args.query_text,
        prompt_file=None,
        dataset_root=None,
        questions_file=None,
        annotation_file=None,
        annotations_dir=None,
        videos_dir=None,
        data_dir=None,
        uid_map_file=None,
        video_map_file=None,
        output=args.output,
        max_frame_panels=24,
        panel_cols=4,
        clip_batch_size=args.clip_batch_size,
        threshold=args.threshold,
        temperature=args.temperature,
        keep_ratio=args.keep_ratio,
        skip_query_diagnostics=True,
        quiet_progress=args.quiet_progress,
        show=False,
    )


def _png_data_uri(frame) -> tuple[str, int, int]:
    array = viz._frame_to_uint8(frame)
    height, width = array.shape[:2]
    image = Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}", width, height


def _rect_svg(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    selected: bool,
) -> str:
    if selected:
        fill = "rgba(37,99,235,0.28)"
        stroke = "rgba(29,78,216,0.96)"
        stroke_width = "3.2"
    else:
        fill = "rgba(255,255,255,0)"
        stroke = "rgba(255,255,255,0.55)"
        stroke_width = "1.15"
    return (
        f'  <rect x="{x:.4f}" y="{y:.4f}" width="{width:.4f}" height="{height:.4f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def _write_svg(
    *,
    output: Path,
    frame,
    mask,
    frame_index: int,
    source_frame_index: int,
    query_text: str,
    view: str,
) -> None:
    image_uri, image_width, image_height = _png_data_uri(frame)
    grid_h = int(mask.shape[0])
    grid_w = int(mask.shape[1])
    patch_w = image_width / grid_w
    patch_h = image_height / grid_h
    selected_count = int(mask.sum().item())
    token_count = grid_h * grid_w
    escaped_query = html.escape(query_text, quote=True)
    aria = (
        f"MaskCLIP raw patch selection, frame {frame_index}/source frame {source_frame_index}, "
        f"query {query_text}, selected {selected_count}/{token_count}, high-resolution export"
    )
    escaped_aria = html.escape(aria, quote=True)

    rects: list[str] = []
    for row in range(grid_h):
        for col in range(grid_w):
            rects.append(
                _rect_svg(
                    x=col * patch_w,
                    y=row * patch_h,
                    width=patch_w,
                    height=patch_h,
                    selected=bool(mask[row, col].item()),
                )
            )

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{image_width}" height="{image_height}" '
            f'viewBox="0 0 {image_width} {image_height}" role="img" aria-label="{escaped_aria}">',
            f'<title>{html.escape(view)} patch selection, frame {frame_index}</title>',
            f'<desc>Query: {escaped_query}</desc>',
            f'<image href="{image_uri}" x="0" y="0" width="{image_width}" height="{image_height}" preserveAspectRatio="none" />',
            '<g shape-rendering="crispEdges">',
            *rects,
            "</g>",
            "</svg>",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    diagnostic_args = _diagnostic_args(args)
    config = OmegaConf.load(str(args.config))
    viz._apply_common_config_overrides(config, diagnostic_args)

    with tempfile.TemporaryDirectory(prefix="export_single_patch_svg_") as temp_name:
        temp_dir = Path(temp_name)
        target = viz._resolve_target(config=config, args=diagnostic_args, temp_dir=temp_dir)
        viz._configure_query_file(config, target.query_file)
        diagnostics = viz._run_clip_only_selection_diagnostics(
            config=config,
            args=diagnostic_args,
            target=target,
        )

    frames, sampled_indices = viz._frames_for_temporal_grid(
        diagnostics.frame_selection,
        diagnostics.selector_metadata,
    )
    if args.frame_index < 0 or args.frame_index >= int(frames.shape[0]):
        raise IndexError(
            f"frame-index {args.frame_index} is out of range for {int(frames.shape[0])} frame(s)."
        )

    if args.view == "clip":
        mask = diagnostics.raw_clip_selected_mask
    else:
        mask = diagnostics.selected_mask
    if mask is None:
        raise ValueError(f"{args.view} selected mask is unavailable.")
    source_frame_index = sampled_indices[args.frame_index] if args.frame_index < len(sampled_indices) else args.frame_index
    _write_svg(
        output=args.output,
        frame=frames[args.frame_index],
        mask=mask[args.frame_index],
        frame_index=args.frame_index,
        source_frame_index=int(source_frame_index),
        query_text=args.query_text,
        view=args.view,
    )
    print(args.output)


if __name__ == "__main__":
    main()
