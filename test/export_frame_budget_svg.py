from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "test"
    / "artifacts"
    / "black_dog_playing_frame2_patch_selection_temporal.html"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "test" / "artifacts" / "black_dog_playing_frame_budget_allocation.svg"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frame-wise allocated patch budgets from a Plotly temporal HTML to SVG.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", type=str, default="Frame-wise Budget Allocation")
    return parser.parse_args()


def _extract_first_json_array_after(text: str, marker: str) -> list[dict[str, Any]]:
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ValueError(f"Could not find marker: {marker}")

    start = text.find("[", marker_index)
    if start < 0:
        raise ValueError("Could not find Plotly trace array.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("Unterminated Plotly trace array.")


def _trace_values(traces: list[dict[str, Any]], name: str) -> tuple[list[int], list[float]]:
    for trace in traces:
        if trace.get("name") == name:
            xs = [int(value) for value in trace["x"]]
            ys = [float(value) for value in trace["y"]]
            return xs, ys
    raise ValueError(f"Could not find trace named {name!r}.")


def _plot_budget_bars(
    *,
    frames: list[int],
    budgets: list[float],
    output: Path,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(
        frames,
        budgets,
        width=0.68,
        color="#F59E0B",
        edgecolor="#92400E",
        linewidth=0.9,
    )

    ax.bar_label(bars, labels=[str(int(value)) for value in budgets], padding=3, fontsize=9)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Temporal frame index")
    ax.set_ylabel("Allocated visual tokens")
    ax.set_xticks(frames)
    ax.set_ylim(0, max(budgets) + 1.4)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    total_budget = int(sum(budgets))
    ax.text(
        0.995,
        0.96,
        f"Total budget: {total_budget}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#334155",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    html = args.input.read_text(encoding="utf-8")
    traces = _extract_first_json_array_after(html, "Plotly.newPlot(")
    frames, budgets = _trace_values(traces, "Allocated budget")
    _plot_budget_bars(
        frames=frames,
        budgets=budgets,
        output=args.output,
        title=args.title,
    )
    print(args.output)


if __name__ == "__main__":
    main()
