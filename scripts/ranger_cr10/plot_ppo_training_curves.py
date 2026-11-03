# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot the historical Ranger+CR10 PPO training curves from TensorBoard event files.

The script only reads training artefacts. It never writes below ``logs/`` and never touches checkpoints,
training configs or a running job. Outputs go to ``outputs/ppo_training_curves/``.

Run with the Isaac Lab environment (TensorBoard is installed there)::

    /home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python \\
        scripts/ranger_cr10/plot_ppo_training_curves.py

Use ``--log-root`` to point at another log tree and ``--out-dir`` to change the output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------------------------
# Design tokens (validated categorical palette, light surface).
# ---------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "ranger_cr10_teleop"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "ppo_training_curves"
MIN_AUTO_ITERATIONS = 50

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------------------------
# Run selection.  Each logical series is one row in the comparison table.  A series may be made of
# several resumed segments; ``own_offset`` places a segment on the concatenated own-iteration axis
# and ``cumulative_base`` places the series on the "PPO iterations since 4.5j start" axis.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSegment:
    """One training run directory and its position inside a resumed series.

    ``truncate_to`` keeps only the first N logged iterations.  The 4.7 A and B runs continued for a few
    iterations past the checkpoint that the next segment actually resumed from (2028 logged vs 2000
    carried forward, 1018 vs 1000); those discarded tail iterations are not part of the lineage.
    """

    suffix: str
    own_offset: int = 0
    truncate_to: int | None = None


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata for one logical training version."""

    key: str
    label: str
    stage: str
    segments: tuple[RunSegment, ...]
    cumulative_base: int
    color: str
    linestyle: object = "-"
    status: str = "completed"
    init_note: str = ""
    note: str = ""
    primary: bool = True
    legend_label: str = ""


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        key="4.5j",
        label="4.5j (main)",
        stage="4.5j",
        segments=(RunSegment("stage45j_ppo"),),
        cumulative_base=0,
        color=BLUE,
        init_note="IK-teacher actor (`_stage45j/model_0.pt`), critic re-initialised, arm std 0.079-0.157",
        note="First standard-PPO run on the 8-D action; 6000/6000 iterations.",
        legend_label="4.5j",
    ),
    SeriesSpec(
        key="4.5j-v2",
        label="4.5j-v2 (parallel control)",
        stage="4.5j",
        segments=(RunSegment("stage45j_v2"),),
        cumulative_base=0,
        color=ORANGE,
        linestyle="--",
        status="completed (control)",
        init_note="Same IK-teacher init as 4.5j; arm std 0.03, initial lr 1e-4, 1500 iterations",
        note="Single-factor follow-up, parallel to 4.5j (not a continuation).",
        primary=False,
        legend_label="4.5j-v2 (control)",
    ),
    SeriesSpec(
        key="4.7",
        label="4.7 (segments A+B+C)",
        stage="4.7",
        segments=(
            RunSegment("stage47_ppo", 0, truncate_to=2000),
            RunSegment("stage47b_ppo", 2000, truncate_to=1000),
            RunSegment("stage47c_ppo", 3000),
        ),
        cumulative_base=6000,
        color=AQUA,
        init_note="4.5j `model_5999`, iteration rebased (actor/critic/optimizer retained)",
        note=(
            "Three resumed segments: A 2000 + B 1000 + C 3000 (checkpoint lineage). A/B logged 2028/1018 "
            "iterations, but only the first 2000/1000 were carried into the next segment; the 28/18 tail "
            "iterations were discarded. Curriculum stuck at stage 1."
        ),
        legend_label="4.7 (A+B+C)",
    ),
    SeriesSpec(
        key="4.8",
        label="4.8 (workspace + fine reward)",
        stage="4.8",
        segments=(RunSegment("stage48a_ppo"),),
        cumulative_base=12000,
        color=YELLOW,
        init_note="4.7 `model_2999`, iteration rebased (actor bit-identical)",
        note="Curriculum reached stage 3 at iteration 500 (both promotions by success, never forced).",
        legend_label="4.8",
    ),
    SeriesSpec(
        key="4.9",
        label="4.9 (safety fine-tune)",
        stage="4.9",
        segments=(RunSegment("stage49a_ppo"),),
        cumulative_base=18000,
        color=MAGENTA,
        init_note="4.8 `model_5999`, iteration rebased (actor bit-identical)",
        note="Curriculum starts at stage 3. Deployed checkpoint `model_5000`.",
        legend_label="4.9",
    ),
    SeriesSpec(
        key="4.10a",
        label="4.10a (fine-tune attempt)",
        stage="4.10",
        segments=(RunSegment("stage410a_ppo"),),
        cumulative_base=23000,
        color=GREEN,
        linestyle=(0, (6, 3)),
        status="stopped (535/6000)",
        init_note="4.9 `model_5000`; actor weights verified bit-identical to the 4.9 checkpoint",
        note="Fine-tune on the new forward pose; it stalls (large EE error, success ~0).",
        primary=False,
        legend_label="4.10a fine-tune",
    ),
    SeriesSpec(
        key="4.10b",
        label="4.10b (output-reset probe)",
        stage="4.10",
        segments=(RunSegment("stage410b_probe"),),
        cumulative_base=23000,
        color=GREEN,
        linestyle=":",
        status="probe (111/150)",
        init_note="Actor output layer zeroed (deliberate basin break); 150-iteration probe",
        note="Probe that motivated the output-layer reset retrain.",
        primary=False,
        legend_label="4.10b probe",
    ),
    SeriesSpec(
        key="4.10c",
        label="4.10c (retrain, stopped)",
        stage="4.10",
        segments=(RunSegment("stage410c_ppo"),),
        cumulative_base=23000,
        color=GREEN,
        linestyle="-.",
        status="stopped (1247/6000)",
        init_note="Output-layer-reset init `_stage410c/model_0.pt`",
        note="Retrain from the output-reset init; stopped at iteration 1247.",
        primary=False,
        legend_label="4.10c stopped",
    ),
    SeriesSpec(
        key="4.10d",
        label="4.10d (retrain, running)",
        stage="4.10",
        segments=(RunSegment("stage410d_ppo"),),
        cumulative_base=23000,
        color=GREEN,
        linestyle="--",
        status="running at snapshot",
        init_note="Same output-reset init as 4.10c (`_stage410c/model_0.pt`); restart, not a resume",
        note="Restart of the 4.10 retrain; the curves are a live snapshot and will grow.",
        primary=False,
        legend_label="4.10d running",
    ),
)

KL_REFERENCE_SPEC = SeriesSpec(
    key="4.5i-ctrlC",
    label="4.5i ctrl_C (KL reference)",
    stage="4.5i",
    segments=(RunSegment("ctrl_C"),),
    cumulative_base=0,
    color=RED,
    linestyle="--",
    status="reference run",
    init_note="4.5i controlled run `2026-10-23_19-59-08_ctrl_C`",
    note="The only run in this log tree that records Loss/kl (200 logged points, steps 999-1198).",
    primary=False,
)


# (tag, slug, y-label, log-y, zero-line, step metric, show curriculum switches)
@dataclass(frozen=True)
class MetricSpec:
    """One scalar to plot."""

    tag: str
    slug: str
    label: str
    logy: bool = False
    symlog: bool = False
    linthresh: float = 0.01
    hline: float | None = None
    step: bool = False
    show_switches: bool = False
    rate: bool = False


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("Loss/surrogate", "surrogate", "Surrogate loss", symlog=True, linthresh=0.005, hline=0.0),
    MetricSpec("Loss/value", "value", "Value loss", logy=True),
    MetricSpec("Loss/entropy", "entropy", "Entropy"),
    MetricSpec("Loss/kl", "kl", "KL divergence", logy=True),
    MetricSpec("Loss/learning_rate", "learning_rate", "Adaptive learning rate", logy=True),
    MetricSpec("Policy/mean_std", "mean_std", "Mean action std"),
    MetricSpec("Train/mean_reward", "reward", "Mean reward", show_switches=True),
    MetricSpec("Train/mean_episode_length", "episode_length", "Mean episode length", show_switches=True),
    MetricSpec("Metrics/ee_pose/position_error", "ee_position", "EE position error [m]", show_switches=True),
    MetricSpec("Metrics/ee_pose/orientation_error", "ee_orientation", "EE orientation error [rad]", show_switches=True),
    MetricSpec("Metrics/ee_pose/success", "success", "Success rate", show_switches=True, rate=True),
    MetricSpec("Metrics/ee_pose/collision", "collision", "Collision rate", rate=True),
    MetricSpec("Metrics/ee_pose/min_clearance", "clearance", "Minimum clearance [m]", hline=0.0),
    MetricSpec("Curriculum/ee_pose_levels", "curriculum_stage", "EE-pose curriculum stage", step=True),
    MetricSpec("Curriculum/lin_vel_cmd_levels", "lin_vel_level", "Linear-velocity command level", step=True),
    MetricSpec("Curriculum/ang_vel_cmd_levels", "ang_vel_level", "Angular-velocity command level", step=True),
)

METRIC_BY_SLUG = {m.slug: m for m in METRICS}
CORE_OVERVIEW_SLUGS = ("surrogate", "value", "entropy", "reward", "episode_length", "learning_rate")
TASK_OVERVIEW_SLUGS = ("ee_position", "ee_orientation", "success", "collision", "clearance", "curriculum_stage")
OWN_ZOOM_SLUGS = ("surrogate", "value", "entropy", "reward", "ee_position", "ee_orientation", "success", "collision")
DASHBOARD_SLUGS = (
    "surrogate",
    "value",
    "entropy",
    "mean_std",
    "reward",
    "episode_length",
    "learning_rate",
    "success",
    "ee_position",
    "ee_orientation",
    "collision",
    "clearance",
    "curriculum_stage",
    "lin_vel_level",
    "ang_vel_level",
)

# Fixed-target evaluations recorded in PROGRESS.md.  These are separate from the training scalars and
# are reproduced here only as context; they are not read from the event files.
DOCUMENTED_EVALUATIONS = (
    (
        "4.5j",
        "Chassis yaw tracking 54% -> 97%; EE attitude 0.55 -> 1.20 deg, position 10 -> 24 mm; "
        "chassis drift 0.0392 -> 0.0015 m/s (PROGRESS.md, 2026-10-30).",
    ),
    (
        "4.7",
        "New-box fixed 60-target set: median position 142 -> 61 mm, orientation 10.1 -> 4.0 deg, "
        "80 mm/10 deg coverage 8.3 -> 66.7% (PROGRESS.md, 2026-11-01).",
    ),
    (
        "4.8",
        "New-box fixed 60-target set: median position 95 -> 24 mm, orientation 6.7 -> 2.7 deg, "
        "50 mm/5 deg coverage 18 -> 77%; honest regression: 14 fixed convergence commands went "
        "0.81 -> 1.28 deg orientation, concentrated in negative pitch (PROGRESS.md, 2026-11-02).",
    ),
    (
        "4.9",
        "Lidar contact 51/93/93/12% -> 0% in every scenario; 50 mm/5 deg success on collision-free "
        "reachable targets identical to 4.8 (77.2% vs 77.2%) (PROGRESS.md, 2026-11-02).",
    ),
    (
        "4.10",
        "The 4.9 policy does not survive the new pose (650 mm / 136 deg); the 4.10 fine-tune stalls, "
        "and the output-reset retrain is still running at this snapshot (PROGRESS.md, 2026-11-03).",
    ),
)


# ---------------------------------------------------------------------------------------------
# Loading helpers.
# ---------------------------------------------------------------------------------------------
def load_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load every scalar tag from one run directory.

    Args:
        run_dir: Directory containing ``events.out.tfevents*``.

    Returns:
        Mapping from tag to ``(steps, values)`` arrays, sorted by step with duplicate steps removed.
    """
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        by_step: dict[int, float] = {}
        for event in accumulator.Scalars(tag):
            by_step[int(event.step)] = float(event.value)
        steps = np.array(sorted(by_step), dtype=np.int64)
        values = np.array([by_step[int(step)] for step in steps], dtype=np.float64)
        if steps.size:
            result[tag] = (steps, values)
    return result


def find_run_dir(log_root: Path, suffix: str) -> Path | None:
    """Return the most recent run directory whose name ends with ``suffix`` and holds an event file."""
    matches = sorted(
        p
        for p in log_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name.endswith(suffix) and any(p.glob("events.out.tfevents*"))
    )
    return matches[-1] if matches else None


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding (the output keeps the input length)."""
    if window <= 1 or values.size < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    largest_odd = values.size if values.size % 2 == 1 else values.size - 1
    window = min(window, largest_odd)
    if window < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def auto_window(n_points: int) -> int:
    """Pick a smoothing window that scales with the number of iterations."""
    if n_points >= 2000:
        return 101
    if n_points >= 500:
        return 51
    if n_points >= 150:
        return 21
    return 11


@dataclass
class SeriesData:
    """Loaded data plus provenance for one logical series."""

    spec: SeriesSpec
    data: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    segments: list[tuple[str, int, int]] = field(default_factory=list)
    loaded_dirs: list[Path] = field(default_factory=list)
    missing_segments: list[str] = field(default_factory=list)

    def cumulative(self, own_steps: np.ndarray) -> np.ndarray:
        """Map own-iteration steps to the cumulative PPO-iteration axis."""
        return own_steps + self.spec.cumulative_base

    @property
    def total_iterations(self) -> int:
        """Number of logged iterations in the concatenated own axis."""
        if "Loss/surrogate" not in self.data:
            return 0
        return int(self.data["Loss/surrogate"][0][-1] + 1) if self.segments else 0


def load_series(spec: SeriesSpec, log_root: Path) -> SeriesData | None:
    """Load and concatenate all segments of one logical series."""
    per_tag: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    series = SeriesData(spec=spec)
    for segment in spec.segments:
        run_dir = find_run_dir(log_root, segment.suffix)
        if run_dir is None:
            series.missing_segments.append(segment.suffix)
            continue
        scalars = load_scalars(run_dir)
        if "Loss/surrogate" not in scalars:
            series.missing_segments.append(segment.suffix)
            continue
        n_iter = int(scalars["Loss/surrogate"][0][-1] + 1)
        keep = n_iter if segment.truncate_to is None else min(n_iter, segment.truncate_to)
        series.segments.append((segment.suffix, segment.own_offset, segment.own_offset + keep - 1))
        series.loaded_dirs.append(run_dir)
        for tag, (steps, values) in scalars.items():
            mask = steps < keep
            per_tag.setdefault(tag, []).append((steps[mask] + segment.own_offset, values[mask]))
    if not series.segments:
        return None
    for tag, chunks in per_tag.items():
        steps = np.concatenate([chunk[0] for chunk in chunks])
        values = np.concatenate([chunk[1] for chunk in chunks])
        order = np.argsort(steps, kind="stable")
        series.data[tag] = (steps[order], values[order])
    return series


def discover_extra_runs(log_root: Path, known_suffixes: set[str]) -> list[SeriesSpec]:
    """Create series specs for future PPO runs that are not in ``SERIES_SPECS``.

    Only directories whose name contains ``stage4`` and that log at least ``MIN_AUTO_ITERATIONS``
    iterations are considered, so smoke tests and non-PPO runs are ignored.
    """
    extras: list[SeriesSpec] = []
    styles = [":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]
    known_ends = [spec.cumulative_base for spec in SERIES_SPECS]
    for run_dir in sorted(log_root.iterdir()):
        if not run_dir.is_dir() or "stage4" not in run_dir.name:
            continue
        if any(run_dir.name.endswith(suffix) for suffix in known_suffixes):
            continue
        event_files = list(run_dir.glob("events.out.tfevents*"))
        if not event_files or event_files[0].stat().st_size < 20_000:
            continue
        scalars = load_scalars(run_dir)
        if "Loss/surrogate" not in scalars:
            continue
        n_iter = len(scalars["Loss/surrogate"][0])
        if n_iter < MIN_AUTO_ITERATIONS:
            continue
        match = re.search(r"stage4(\d+)([a-z]?)", run_dir.name)
        stage = f"4.{match.group(1)}{match.group(2)}" if match else "auto"
        key = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", run_dir.name) or run_dir.name
        idx = len(extras)
        extras.append(
            SeriesSpec(
                key=key,
                label=f"{key} (auto-discovered)",
                stage=stage,
                segments=(RunSegment(run_dir.name),),
                cumulative_base=max(known_ends),
                color=VIOLET,
                linestyle=styles[idx % len(styles)],
                status="auto-discovered",
                init_note="Auto-discovered run; cumulative base inherited from the latest known lineage",
                note="Appeared after this script was written; edit SERIES_SPECS to pin its lineage.",
                primary=False,
            )
        )
    return extras


# ---------------------------------------------------------------------------------------------
# Plot helpers.
# ---------------------------------------------------------------------------------------------
def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    """Save a figure as PNG and PDF without letting dots in the stem become a file suffix."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.parent / f"{stem.name}.png", dpi=dpi)
    fig.savefig(stem.parent / f"{stem.name}.pdf")
    plt.close(fig)


def row_major_legend_order(handles: list[Line2D], labels: list[str], ncol: int) -> tuple[list[Line2D], list[str]]:
    """Reorder legend entries so that matplotlib's column-major fill reads row-major."""
    nrow = int(np.ceil(len(handles) / ncol))
    ordered: list[int] = []
    for column in range(ncol):
        for row in range(nrow):
            index = row * ncol + column
            if index < len(handles):
                ordered.append(index)
    return [handles[i] for i in ordered], [labels[i] for i in ordered]


def style_axes(ax: plt.Axes) -> None:
    """Apply the shared light-surface chart style."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-", alpha=1.0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=8.5, width=0.6, length=3)


def format_metric_axis(ax: plt.Axes, metric: MetricSpec) -> None:
    """Set the y scale, limits and tick formatting for one metric."""
    if metric.symlog:
        ax.set_yscale("symlog", linthresh=metric.linthresh)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    elif metric.logy:
        positive = True
        for line in ax.get_lines():
            values = np.asarray(line.get_ydata(), dtype=float)
            if values.size and not np.all(values[values > 0] > 0):
                positive = False
                break
        if positive and ax.get_lines():
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    if metric.hline is not None:
        ax.axhline(metric.hline, color=AXIS, linewidth=0.6, zorder=0)
    if metric.rate:
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    if metric.step:
        ax.set_ylim(-0.2, 3.2)
        ax.set_yticks([0, 1, 2, 3])
    if metric.slug == "learning_rate":
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 3.0), numticks=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))


def curriculum_switches(series: SeriesData) -> list[tuple[int, float, float]]:
    """Return ``(own_step, old_level, new_level)`` for real EE-pose curriculum promotions.

    Small intermediate values written during a promotion (for example 0 -> 0.375 -> 1) are coalesced
    into a single switch at the last sample of the burst.
    """
    if "Curriculum/ee_pose_levels" not in series.data:
        return []
    steps, values = series.data["Curriculum/ee_pose_levels"]
    raw: list[tuple[int, float, float]] = []
    for i in range(1, steps.size):
        if abs(values[i] - values[i - 1]) >= 0.3:
            raw.append((int(steps[i]), float(values[i - 1]), float(values[i])))
    merged: list[tuple[int, float, float]] = []
    for step, old, new in raw:
        if merged and step - merged[-1][0] <= 10:
            merged[-1] = (step, merged[-1][1], new)
        else:
            merged.append((step, old, new))
    return merged


def plot_metric_series(ax: plt.Axes, series: SeriesData, metric: MetricSpec, x_mode: str) -> Line2D | None:
    """Draw raw + smoothed curves for one series; return its legend handle."""
    if metric.tag not in series.data:
        return None
    own_steps, values = series.data[metric.tag]
    x = series.cumulative(own_steps) if x_mode == "cumulative" else own_steps
    order = np.argsort(x, kind="stable")
    x = x[order]
    values = values[order]
    linewidth = 2.0 if series.spec.primary else 1.35
    alpha = 1.0 if series.spec.primary else 0.85
    if metric.step:
        ax.step(
            x,
            values,
            where="post",
            color=series.spec.color,
            linewidth=linewidth,
            linestyle=series.spec.linestyle,
            alpha=alpha,
            zorder=3,
        )
    else:
        ax.plot(x, values, color=series.spec.color, linewidth=0.55, alpha=0.11, zorder=1)
        smooth = moving_average(values, auto_window(values.size))
        ax.plot(
            x,
            smooth,
            color=series.spec.color,
            linewidth=linewidth,
            linestyle=series.spec.linestyle,
            alpha=alpha,
            zorder=3,
        )
    legend_label = series.spec.legend_label or series.spec.label
    return Line2D(
        [0],
        [0],
        color=series.spec.color,
        linewidth=linewidth,
        linestyle=series.spec.linestyle,
        alpha=alpha,
        label=legend_label,
    )


def add_structure(ax: plt.Axes, series_list: list[SeriesData], metric: MetricSpec, x_mode: str) -> bool:
    """Add resume boundaries and curriculum-switch markers. Return whether any switch was drawn."""
    drew_switch = False
    for series in series_list:
        # A run that never logged this metric must not stretch the x-axis through its segment lines.
        if metric.tag not in series.data:
            continue
        if x_mode == "cumulative":
            boundaries = [series.cumulative(np.array([start]))[0] for _, start, _ in series.segments[1:]]
        else:
            boundaries = [start for _, start, _ in series.segments[1:]]
        for boundary in boundaries:
            ax.axvline(boundary, color=MUTED, linewidth=0.7, linestyle=(0, (2, 3)), alpha=0.55, zorder=0)
        if metric.show_switches:
            for step, _, _ in curriculum_switches(series):
                x = series.cumulative(np.array([step]))[0] if x_mode == "cumulative" else step
                ax.axvline(x, color=series.spec.color, linewidth=0.8, linestyle=":", alpha=0.55, zorder=2)
                drew_switch = True
    return drew_switch


def add_stage_labels(ax: plt.Axes, x_mode: str, x_max: float) -> None:
    """Label the training-stage bands at the bottom of a cumulative overlay."""
    if x_mode != "cumulative":
        return
    bands = (
        (0, 6000, "4.5j"),
        (6000, 12000, "4.7"),
        (12000, 18000, "4.8"),
        (18000, 23000, "4.9"),
        (23000, x_max, "4.10 branch"),
    )
    for start, end, label in bands:
        if start > x_max:
            continue
        ax.axvline(start, color=MUTED, linewidth=0.8, linestyle="-", alpha=0.45, zorder=0)
        ax.text(
            0.5 * (start + min(end, x_max)),
            0.015,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=SECONDARY,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
            zorder=4,
        )


def overlay_figure(
    series_list: list[SeriesData],
    metric: MetricSpec,
    x_mode: str,
    out_stem: Path,
    zoom_end: int | None = None,
) -> None:
    """Render one multi-version metric figure (PNG + PDF)."""
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.215)
    handles: list[Line2D] = []
    labels: list[str] = []
    for series in series_list:
        handle = plot_metric_series(ax, series, metric, x_mode)
        if handle is not None:
            handles.append(handle)
            labels.append(handle.get_label())
    style_axes(ax)
    format_metric_axis(ax, metric)
    drew_switch = add_structure(ax, series_list, metric, x_mode)
    x_max = float(ax.get_xlim()[1])
    add_stage_labels(ax, x_mode, x_max)
    if zoom_end is not None and x_mode != "cumulative":
        ax.set_xlim(0, zoom_end)
    x_label = {
        "cumulative": "Cumulative PPO iteration (4.5j start = 0)",
        "own": "PPO iteration within the training stage (4.7 segments concatenated)",
        "own_zoom": f"PPO iteration within the training stage (first {zoom_end} iterations)",
    }[x_mode]
    if metric.slug == "kl":
        x_label = "PPO iteration (4.5i control run; KL is logged only for steps 999-1198)"
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.label)
    fig.text(0.075, 0.955, metric.label, fontsize=14.5, fontweight="semibold", color=INK, ha="left")
    subtitle = "Thin faint lines = raw per-iteration values; thick lines = centred moving average."
    if metric.step:
        subtitle = "Curriculum levels are plotted as step functions (no smoothing)."
    fig.text(0.075, 0.915, subtitle, fontsize=8.8, color=SECONDARY, ha="left")
    if drew_switch:
        handles.append(Line2D([0], [0], color=MUTED, linewidth=0.8, linestyle=":", label="EE-pose curriculum switch"))
        labels.append("EE-pose curriculum switch")
    if len(handles) > 1:
        ncol = min(5, len(handles))
        ordered_handles, ordered_labels = row_major_legend_order(handles, labels, ncol)
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=ncol,
            handlelength=2.6,
            columnspacing=1.4,
            labelspacing=0.5,
        )
    elif len(handles) == 1:
        fig.text(0.075, 0.045, labels[0], fontsize=9, color=SECONDARY)
    if metric.slug == "kl":
        fig.text(
            0.525,
            0.795,
            "Only 4.5i ctrl_C (2026-10-23) logs Loss/kl; rsl_rl 5.0.1 does not record KL in the 4.5j-4.10 configs.\n"
            "The adaptive learning-rate curve is the available proxy for the KL controller.",
            fontsize=8.8,
            color=SECONDARY,
            ha="left",
            va="top",
        )
    if metric.slug in ("collision", "clearance"):
        fig.text(
            0.075,
            0.515,
            "These tags exist only from 4.9 onwards; earlier runs are absent, not zero.",
            fontsize=8.8,
            color=SECONDARY,
            ha="left",
            va="top",
        )
    save_figure(fig, out_stem)


def run_dashboard(series: SeriesData, out_stem: Path) -> None:
    """Render one run's raw + smoothed dashboard (PNG + PDF)."""
    fig, axes = plt.subplots(4, 4, figsize=(16.2, 12.2), constrained_layout=False)
    axes_flat = axes.ravel()
    metric_list = [METRIC_BY_SLUG[slug] for slug in DASHBOARD_SLUGS]
    x_all = None
    for ax, metric in zip(axes_flat, metric_list):
        style_axes(ax)
        if metric.tag not in series.data:
            ax.text(
                0.5,
                0.5,
                "not logged in this run",
                ha="center",
                va="center",
                fontsize=9,
                color=MUTED,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(metric.label, fontsize=10, color=SECONDARY)
            continue
        own_steps, values = series.data[metric.tag]
        x_all = own_steps
        if metric.step:
            ax.step(own_steps, values, where="post", color=series.spec.color, linewidth=1.8)
        else:
            ax.plot(own_steps, values, color=series.spec.color, linewidth=0.55, alpha=0.16)
            ax.plot(own_steps, moving_average(values, auto_window(values.size)), color=series.spec.color, linewidth=1.8)
        format_metric_axis(ax, metric)
        add_structure(ax, [series], metric, "own")
        ax.set_title(metric.label, fontsize=10, color=INK)
    for ax in axes_flat[12:15]:
        if ax.get_visible():
            ax.set_xlabel("iteration")
    for ax in axes_flat[len(metric_list) :]:
        ax.set_visible(False)
    own_range = "n/a"
    if x_all is not None:
        own_range = f"{int(x_all[0])}-{int(x_all[-1])}"
    cum_start = series.spec.cumulative_base
    cum_end = series.spec.cumulative_base + (int(x_all[-1]) if x_all is not None else 0)
    fig.suptitle(
        f"{series.spec.label}  —  own iterations {own_range}  |  cumulative {cum_start}-{cum_end}  |  "
        f"{series.spec.status}",
        fontsize=14,
        fontweight="semibold",
        color=INK,
        y=0.992,
    )
    fig.text(
        0.5,
        0.962,
        "Thin faint line = raw per-iteration value; thick line = moving average.",
        ha="center",
        fontsize=8.8,
        color=SECONDARY,
    )
    footnote = f"Init: {series.spec.init_note}."
    if series.spec.note:
        footnote += f"  {series.spec.note}"
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.93))
    fig.text(0.01, 0.008, footnote, fontsize=8.2, color=SECONDARY, ha="left")
    save_figure(fig, out_stem)


def overview_dashboard(series_list: list[SeriesData], slugs: tuple[str, ...], x_mode: str, out_stem: Path) -> None:
    """Render a compact multi-metric overview with all versions overlaid."""
    metrics = [METRIC_BY_SLUG[slug] for slug in slugs]
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17.0, 4.4 * nrows), squeeze=False, constrained_layout=True)
    handles: list[Line2D] = []
    labels: list[str] = []
    for ax, metric in zip(axes.ravel(), metrics):
        style_axes(ax)
        for series in series_list:
            handle = plot_metric_series(ax, series, metric, x_mode)
            if handle is not None and handle.get_label() not in labels:
                handles.append(handle)
                labels.append(handle.get_label())
        format_metric_axis(ax, metric)
        add_structure(ax, series_list, metric, x_mode)
        ax.set_title(metric.label, fontsize=10.5, color=INK)
        ax.set_xlabel("cumulative iteration" if x_mode == "cumulative" else "iteration within stage")
    for ax in axes.ravel()[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"PPO training overview — {'cumulative history' if x_mode == 'cumulative' else 'own iteration'}",
        fontsize=15,
        fontweight="semibold",
        color=INK,
    )
    ncol = min(5, len(handles))
    ordered_handles, ordered_labels = row_major_legend_order(handles, labels, ncol)
    fig.legend(ordered_handles, ordered_labels, loc="outside lower center", ncol=ncol, handlelength=2.6)
    save_figure(fig, out_stem, dpi=260)


def timeline_figure(series_list: list[SeriesData], out_stem: Path) -> None:
    """Render a Gantt-style timeline of own vs cumulative iteration ranges."""
    # label, cumulative start, cumulative end, own start, own end, colour
    rows: list[tuple[str, int, int, int, int, str]] = []
    for series in series_list:
        for index, (_, start, end) in enumerate(series.segments):
            if len(series.segments) == 1:
                label = series.spec.legend_label or series.spec.label
            else:
                label = f"{series.spec.stage} {chr(65 + index)}"
            rows.append(
                (
                    label,
                    series.spec.cumulative_base + start,
                    series.spec.cumulative_base + end,
                    start,
                    end,
                    series.spec.color,
                )
            )
    fig, ax = plt.subplots(figsize=(14.0, 0.62 * len(rows) + 2.2))
    fig.subplots_adjust(left=0.22, right=0.985, top=0.88, bottom=0.16)
    for y, (label, cum_start, cum_end, own_start, own_end, color) in enumerate(rows):
        ax.barh(
            y,
            cum_end - cum_start + 1,
            left=cum_start,
            color=color,
            alpha=0.22,
            height=0.62,
            edgecolor=color,
            linewidth=1.2,
        )
        ax.text(
            cum_end + 250,
            y,
            f"own {own_start}-{own_end}  ({own_end - own_start + 1} it)",
            va="center",
            ha="left",
            fontsize=8.2,
            color=SECONDARY,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -1.6)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cumulative PPO iteration (4.5j start = 0)")
    for boundary, label in ((6000, "4.7 start"), (12000, "4.8 start"), (18000, "4.9 start"), (23000, "4.10 branch")):
        ax.axvline(boundary, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.7)
        ax.text(boundary, -1.1, label, rotation=90, va="bottom", ha="right", fontsize=7.8, color=SECONDARY)
    ax.set_xlim(0, max(row[2] for row in rows) + 3000)
    fig.suptitle(
        "PPO lineage: resume points, segment boundaries and cumulative iterations",
        fontsize=14.5,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.01,
        0.02,
        "Bars start at each run's cumulative iteration. 4.7 is one logical version made of three resumed "
        "segments; 4.5j-v2 is a parallel control; 4.10a-d are alternative attempts from the same branch point.",
        fontsize=8.2,
        color=SECONDARY,
    )
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------------------------
# Tables and validation.
# ---------------------------------------------------------------------------------------------
def last_mean(values: np.ndarray, n: int = 100) -> float:
    """Mean of the last ``n`` samples (NaN when empty)."""
    return float(np.mean(values[-n:])) if values.size else float("nan")


def final_metrics_row(series: SeriesData) -> dict[str, object]:
    """Build one row of the final-metrics table."""
    row: dict[str, object] = {
        "series": series.spec.key,
        "label": series.spec.label,
        "stage": series.spec.stage,
        "status": series.spec.status,
        "run_dirs": ";".join(str(path.relative_to(PROJECT_ROOT)) for path in series.loaded_dirs),
        "total_iterations": series.total_iterations,
        "cumulative_start": series.spec.cumulative_base,
        "cumulative_end": series.spec.cumulative_base + max(0, series.total_iterations - 1),
        "init": series.spec.init_note,
        "note": series.spec.note,
    }
    for metric in METRICS:
        value = float("nan")
        if metric.tag in series.data:
            _, values = series.data[metric.tag]
            value = last_mean(values)
        row[f"{metric.slug}_last100"] = value
    return row


def write_final_metrics(rows: list[dict[str, object]], path: Path) -> None:
    """Write the machine-readable final-metrics table."""
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: ("" if isinstance(value, float) and np.isnan(value) else value) for key, value in row.items()}
            )


def write_inventory(series_list: list[SeriesData], path: Path) -> None:
    """Write one row per run segment."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "stage",
                "run_dir",
                "own_start",
                "own_end",
                "cumulative_start",
                "cumulative_end",
                "init_note",
                "status",
            ]
        )
        for series in series_list:
            for (name, start, end), run_dir in zip(series.segments, series.loaded_dirs):
                writer.writerow(
                    [
                        series.spec.key,
                        series.spec.stage,
                        str(run_dir.relative_to(PROJECT_ROOT)),
                        start,
                        end,
                        series.spec.cumulative_base + start,
                        series.spec.cumulative_base + end,
                        series.spec.init_note,
                        series.spec.status,
                    ]
                )


def fmt(value: object, digits: int = 4) -> str:
    """Format a table cell, keeping missing values visible as an em dash."""
    if isinstance(value, float) and np.isnan(value):
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def write_comparison_table(rows: list[dict[str, object]], path: Path) -> None:
    """Write the human-readable comparison table."""
    lines = [
        "# PPO training versions — final training metrics",
        "",
        "Values are the mean of the last 100 logged iterations. `—` means the scalar is not present in that run's",
        "TensorBoard log; it is not zero and was not synthesised.",
        "",
        "*Training-time EE and success metrics are measured against the curriculum box active at that iteration",
        "(and, for 4.5j-v2, against the same 4.5j box). They are not comparable across the 4.7/4.8/4.9/4.10 box*",
        "*changes; use the fixed-target evaluations in README.md for cross-version quality.*",
        "",
        "| Version | PPO iters | Cumulative | Status | Reward | Value loss | Surrogate | Entropy | EE pos (m)* | "
        "EE ori (rad)* | Success* | Collision* | Curriculum |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {iters} | {cum0}–{cum1} | {status} | {reward} | {value} | {surrogate} | {entropy} | "
            "{ee_pos} | {ee_ori} | {success} | {collision} | {stage} |".format(
                label=row["label"],
                iters=row["total_iterations"],
                cum0=row["cumulative_start"],
                cum1=row["cumulative_end"],
                status=row["status"],
                reward=fmt(row["reward_last100"], 2),
                value=fmt(row["value_last100"], 3),
                surrogate=fmt(row["surrogate_last100"], 4),
                entropy=fmt(row["entropy_last100"], 3),
                ee_pos=fmt(row["ee_position_last100"], 4),
                ee_ori=fmt(row["ee_orientation_last100"], 4),
                success=fmt(row["success_last100"], 3),
                collision=fmt(row["collision_last100"], 3),
                stage=fmt(row["curriculum_stage_last100"], 1),
            )
        )
    lines += [
        "",
        "## Documented fixed-target evaluations (external to the training logs)",
        "",
        "These are copied from `PROGRESS.md` and are **not** derived from the curves above; they are the only",
        "cross-version quality comparison that uses a fixed target set.",
        "",
    ]
    for stage, text in DOCUMENTED_EVALUATIONS:
        lines.append(f"- **{stage}**: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


CSV_MAP = {
    "surrogate": "Loss/surrogate",
    "value": "Loss/value",
    "entropy": "Loss/entropy",
    "reward": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
    "ee_pos": "Metrics/ee_pose/position_error",
    "ee_ori": "Metrics/ee_pose/orientation_error",
    "success": "Metrics/ee_pose/success",
}

SUPPLEMENTAL_CSVS = {
    "4.8": PROJECT_ROOT / "outputs" / "stage48" / "metrics_a.csv",
    "4.9": PROJECT_ROOT / "outputs" / "stage49" / "metrics_a.csv",
}


def csv_max_difference(csv_path: Path, data: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[int, float]:
    """Return ``(checked_rows, max_abs_diff)`` between a metrics CSV and scalar arrays."""
    max_diff = 0.0
    rows = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            iteration = int(float(row["iter"]))
            for column, tag in CSV_MAP.items():
                if column not in row or tag not in data:
                    continue
                steps, values = data[tag]
                index = np.searchsorted(steps, iteration)
                if index >= steps.size or int(steps[index]) != iteration:
                    continue
                max_diff = max(max_diff, abs(float(row[column]) - float(values[index])))
                rows += 1
    return rows, max_diff


def comparison_table_figure(rows: list[dict[str, object]], out_stem: Path) -> None:
    """Render the final-metrics comparison table as a PNG/PDF figure."""
    columns = ["Version", "PPO iters", "Reward", "Value loss", "EE pos (m)*", "EE ori (rad)*", "Success*", "Collision*"]
    cell_rows = []
    for row in rows:
        cell_rows.append(
            [
                str(row["label"]),
                str(row["total_iterations"]),
                fmt(row["reward_last100"], 2),
                fmt(row["value_last100"], 3),
                fmt(row["ee_position_last100"], 4),
                fmt(row["ee_orientation_last100"], 4),
                fmt(row["success_last100"], 3),
                fmt(row["collision_last100"], 3),
            ]
        )
    fig, ax = plt.subplots(figsize=(15.5, 0.62 * len(cell_rows) + 2.4))
    ax.axis("off")
    table = ax.table(cellText=cell_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.65)
    table.auto_set_column_width(list(range(len(columns))))
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(weight="semibold", color=INK)
        elif row_index % 2 == 0:
            cell.set_facecolor("#f9f9f7")
        else:
            cell.set_facecolor(SURFACE)
    fig.suptitle("PPO training versions — final training metrics", fontsize=14.5, fontweight="semibold", color=INK)
    fig.text(
        0.5,
        0.045,
        "Last-100-iteration mean.  * Training-time metrics against the curriculum box active at that iteration;\n"
        "they are not comparable across the 4.7/4.8/4.9/4.10 box changes.  — means the scalar was not logged.",
        ha="center",
        fontsize=8.6,
        color=SECONDARY,
    )
    save_figure(fig, out_stem)


def cross_check_csv(series_list: list[SeriesData], out_path: Path) -> None:
    """Compare recorded metrics CSVs against the event-file scalars at the same iterations.

    Each run's own ``metrics.csv`` is checked against that run's own event file (not against the merged
    series), so resumed segments with rebased iterations are compared correctly. The two supplemental
    ``outputs/stageNN/metrics_a.csv`` files are compared only with their matching series.
    """
    lines = ["# CSV ↔ TensorBoard validation", ""]
    lines.append("Each CSV row is looked up by its exact `iter` in the event-file scalar. The table reports")
    lines.append("the maximum absolute difference across all checked rows and metrics. 0.000e+00 means the")
    lines.append("recorded CSV and the plotted event file are identical at every recorded row.")
    lines.append("")
    lines.append("| Series | CSV | Rows | Max abs diff |")
    lines.append("|---|---|---:|---:|")
    for series in series_list:
        checked_any = False
        for run_dir in series.loaded_dirs:
            csv_path = run_dir / "metrics.csv"
            if not csv_path.exists():
                continue
            rows, max_diff = csv_max_difference(csv_path, load_scalars(run_dir))
            if rows:
                csv_rel = csv_path.relative_to(PROJECT_ROOT)
                lines.append(f"| {series.spec.key} | `{csv_rel}` | {rows} | {max_diff:.3e} |")
                checked_any = True
        supplemental = SUPPLEMENTAL_CSVS.get(series.spec.key)
        if supplemental is not None and supplemental.exists():
            rows, max_diff = csv_max_difference(supplemental, series.data)
            if rows:
                lines.append(
                    f"| {series.spec.key} | `{supplemental.relative_to(PROJECT_ROOT)}` | {rows} | {max_diff:.3e} |"
                )
                checked_any = True
        if not checked_any:
            lines.append(f"| {series.spec.key} | — (no metrics.csv recorded) | 0 | — |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    series_list: list[SeriesData],
    rows: list[dict[str, object]],
    out_dir: Path,
    log_root: Path,
    snapshot: dt.datetime,
) -> None:
    """Write the directory README with inventory, conventions and caveats."""
    lines = [
        "# Ranger+CR10 PPO training curves",
        "",
        f"Generated by `scripts/ranger_cr10/plot_ppo_training_curves.py` at {snapshot.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "All curves are read directly from TensorBoard `events.out.tfevents*` files under",
        f"`{log_root.relative_to(PROJECT_ROOT)}/`. Nothing below `logs/` was modified. The script also",
        "cross-checks the recorded `metrics.csv` files against the same scalars (see `data_validation.md`).",
        "",
        "TensorBoard events are the source of truth. The JSON files under `logs/` are IK-teacher or supervised",
        "teacher statistics (`logs/ik_teacher*/stats.json`, `logs/rsl_rl/ranger_cr10_teleop/_stage45g/*_stats.json`),",
        "and `_stage45i/loss_mix11.csv` records a supervised IK loss; none of them are PPO scalars, so none are",
        "plotted. Smoke directories below 50 iterations and the earlier controlled 4.5e runs are also excluded.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python scripts/ranger_cr10/plot_ppo_training_curves.py",
        "```",
        "",
        "New run directories matching `*stage4*` and logging at least "
        f"{MIN_AUTO_ITERATIONS} iterations are discovered automatically (violet, dashed). Pin their",
        "lineage in `SERIES_SPECS` when the exact cumulative offset matters.",
        "",
        "## Conventions",
        "",
        "- **Own iteration** is the concatenated iteration inside one logical version. The three 4.7 segments",
        "  (A logged 2028 truncated to 2000, B logged 1018 truncated to 1000, C 3000) are placed at",
        "  0–1999, 2000–2999, 3000–5999.",
        "- **Cumulative iteration** counts PPO updates from the 4.5j start. 4.5j = 0–5999, 4.7 = 6000–11999,",
        "  4.8 = 12000–17999, 4.9 = 18000–23999, 4.10 branches at 23000 (from 4.9 `model_5000`).",
        "- Raw per-iteration values are the thin faint lines; the thick lines are centred moving averages",
        "  (window 101 for ≥2000 points, 51 for ≥500, 21 for ≥150, otherwise 11).",
        "- Dotted vertical lines mark real `Curriculum/ee_pose_levels` promotions. Dashed grey lines mark",
        "  resume/segment boundaries.",
        "- 4.10a–d share the 4.10 hue and are distinguished by line style: they are alternative attempts",
        "  from the same branch point, not one continuous run.",
        "",
        "## Caveats",
        "",
        "- **EE position/orientation/success are training-time metrics against the active curriculum box.**",
        "  The box changed in 4.7, 4.8 and 4.9, so these curves cannot rank versions across stages. Use the",
        "  fixed-target evaluations in `comparison_table.md` for quality comparisons.",
        "- **`Metrics/ee_pose/success` is a running average since the last command resample**, not an",
        "  independent per-iteration estimate; the short-period swings (visible in 4.5j) are expected.",
        "- **The reward function changed in 4.8 (narrow fine-position kernel) and 4.9 (clearance penalty).**",
        "  Raw `Train/mean_reward` is therefore not a cross-stage ranking either; use it to judge within-run",
        "  convergence, not to declare one version better than another.",
        "- **KL is not logged by these configs.** `rsl_rl 5.0.1` records `Loss/kl` only in the 4.5i",
        "  control run `2026-10-23_19-59-08_ctrl_C` (200 points, steps 999–1198). The 4.5j–4.10 runs log",
        "  `Loss/learning_rate`, which is the adaptive controller's response and the available proxy.",
        "- **Collision and clearance exist only from 4.9.** Earlier series are absent, not zero.",
        "- **Each resumed segment's first iteration logs episode length 19.8** (the pre-rollout value) before",
        "  saturating at 200; the vertical dip at every resume boundary is a startup transient, not a collapse.",
        "- **4.5j-v2's adaptive learning rate sits at the 1e-5 floor for the whole 1500-iteration run**; its value",
        "  loss stays around 5.7 and it is not a like-for-like replacement for 4.5j.",
        "- **4.10d is a live snapshot** and will grow after this file was generated.",
        "- **4.7 A and B logged 2028/1018 iterations, but the next segment resumed from `model_2000`/`model_1000`.**",
        "  The 28/18 tail iterations were not carried forward and are excluded from the concatenated 4.7 series,",
        "  which is 2000 + 1000 + 3000 = 6000 iterations and matches the checkpoint lineage.",
        "- 4.5j-v2 is a parallel single-factor control from the same IK init, not a continuation of 4.5j.",
        "- The 4.10 fine-tune (4.10a) is a documented failure: it starts from the 4.9 actor on a pose the",
        "  actor was never trained for. Do not read its early collapse as a training-instability regression.",
        "",
        "## Versions",
        "",
        "| Version | Runs | Own iters | Cumulative | Status | Init |",
        "|---|---|---:|---|---|---|",
    ]
    for series in series_list:
        run_names = ", ".join(f"`{path.name}`" for path in series.loaded_dirs)
        lines.append(
            f"| {series.spec.label} | {run_names} | {series.total_iterations} | "
            f"{series.spec.cumulative_base}–{series.spec.cumulative_base + max(0, series.total_iterations - 1)} | "
            f"{series.spec.status} | {series.spec.init_note} |"
        )
    lines += [
        "",
        "## Metric availability",
        "",
        "| Version | surrogate | value | entropy | KL | reward | ep_len | EE pos | EE ori | success | collision | "
        "clearance | curriculum |",
        "|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for series in series_list:
        cells = []
        for slug in (
            "surrogate",
            "value",
            "entropy",
            "kl",
            "reward",
            "episode_length",
            "ee_position",
            "ee_orientation",
            "success",
            "collision",
            "clearance",
            "curriculum_stage",
        ):
            cells.append("yes" if METRIC_BY_SLUG[slug].tag in series.data else "—")
        lines.append(f"| {series.spec.label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`curriculum` is `Curriculum/ee_pose_levels`. 4.5j and 4.5j-v2 only log the base velocity levels",
        "(`Curriculum/lin_vel_cmd_levels` / `ang_vel_cmd_levels`), which are plotted as separate overlays.",
        "",
        "## Files",
        "",
        "- `comparison_table.md` — version vs final training metrics, plus documented fixed-target results.",
        "- `comparison_table.png/pdf` — the same comparison table rendered as a figure.",
        "- `final_metrics.csv` — same numbers, machine-readable (last-100 mean).",
        "- `run_inventory.csv` — one row per run directory and segment.",
        "- `data_validation.md` — CSV ↔ TensorBoard agreement check.",
        "- `overlays/` — one figure per metric in `own`, `own_zoom` and `cumulative` x-axes.",
        "- `individual/` — one raw + smoothed dashboard per logical version.",
        "- `overview/` — six-panel core and task overviews.",
        "- `timeline.png/pdf` — Gantt view of resume points and cumulative ranges.",
        "- `kl_reference.png/pdf` — the only KL series available in this log tree.",
        "",
        "## Cited fixed-target evaluations",
        "",
    ]
    for stage, text in DOCUMENTED_EVALUATIONS:
        lines.append(f"- **{stage}**: {text}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-auto-discover", action="store_true", help="only plot the pinned SERIES_SPECS")
    return parser.parse_args()


def main() -> None:
    """Load the runs, render every figure and write the tables."""
    args = parse_args()
    log_root = args.log_root.resolve()
    out_dir = args.out_dir.resolve()
    snapshot = dt.datetime.now()
    if not log_root.is_dir():
        raise FileNotFoundError(f"log root not found: {log_root}")

    specs = list(SERIES_SPECS)
    if not args.no_auto_discover:
        known = {segment.suffix for spec in SERIES_SPECS for segment in spec.segments}
        specs.extend(discover_extra_runs(log_root, known))

    series_list: list[SeriesData] = []
    for spec in specs:
        loaded = load_series(spec, log_root)
        if loaded is None:
            print(f"[skip] {spec.key}: no usable run directory")
            continue
        if loaded.missing_segments:
            print(f"[warn] {spec.key}: missing segments {loaded.missing_segments}")
        series_list.append(loaded)

    # KL is only logged by the 4.5i control run; keep it as a reference series, never as a PPO version.
    kl_reference = load_series(KL_REFERENCE_SPEC, log_root)
    kl_series = [kl_reference] if kl_reference is not None and "Loss/kl" in kl_reference.data else []

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)
    (out_dir / "individual").mkdir(exist_ok=True)
    (out_dir / "overview").mkdir(exist_ok=True)

    rows = [final_metrics_row(series) for series in series_list]
    write_final_metrics(rows, out_dir / "final_metrics.csv")
    write_inventory(series_list, out_dir / "run_inventory.csv")
    write_comparison_table(rows, out_dir / "comparison_table.md")
    comparison_table_figure(rows, out_dir / "comparison_table")
    cross_check_csv(series_list, out_dir / "data_validation.md")
    write_readme(series_list, rows, out_dir, log_root, snapshot)

    for metric in METRICS:
        plot_series = kl_series if (metric.slug == "kl" and kl_series) else series_list
        overlay_figure(plot_series, metric, "own", out_dir / "overlays" / f"overlay_own_{metric.slug}")
        overlay_figure(plot_series, metric, "cumulative", out_dir / "overlays" / f"overlay_cumulative_{metric.slug}")
        if metric.slug in OWN_ZOOM_SLUGS:
            overlay_figure(
                plot_series, metric, "own_zoom", out_dir / "overlays" / f"overlay_own_zoom_{metric.slug}", zoom_end=1500
            )
        print(f"[plot] overlay {metric.slug}")

    for series in series_list:
        run_dashboard(series, out_dir / "individual" / f"{series.spec.key}_dashboard")
        print(f"[plot] dashboard {series.spec.key}")

    for mode in ("own", "cumulative"):
        overview_dashboard(series_list, CORE_OVERVIEW_SLUGS, mode, out_dir / "overview" / f"overview_core_{mode}")
        overview_dashboard(series_list, TASK_OVERVIEW_SLUGS, mode, out_dir / "overview" / f"overview_task_{mode}")
    timeline_figure(series_list, out_dir / "timeline")

    # A dedicated, easy-to-find copy of the only real KL series.
    if kl_series:
        overlay_figure(kl_series, METRIC_BY_SLUG["kl"], "own", out_dir / "kl_reference")
    print(f"[done] {len(series_list)} versions, {len(METRICS)} metrics -> {out_dir}")


if __name__ == "__main__":
    main()
