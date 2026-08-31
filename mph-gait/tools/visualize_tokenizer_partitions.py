#!/usr/bin/env python3
"""Visualize the exact MPH-Gait regular-window routing on a real point cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


RELEASE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RELEASE_ROOT.parent
sys.path.insert(0, str(RELEASE_ROOT / "shared"))
sys.path.insert(0, str(RELEASE_ROOT / "mph_gait" / "code"))

from gait_core.pointcloud_dataset import (  # noqa: E402
    AXIS_ALIGNED_METRIC_COORDINATES,
    adapt_point_coordinates,
)
from mph_gait.tokenizer import LocalWindowTokenizer, WindowPartition  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "dataset"
    / "PersonRecognitionWalking_29"
    / "person_003"
    / "P3_C4_V001"
    / "clear_data_060.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RELEASE_ROOT / "docs" / "figures" / "tokenizer_real_frame",
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def deterministic_sample(points: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the evaluation loader's deterministic point sampling exactly."""
    count = int(points.shape[0])
    if count >= num_points:
        indices = np.linspace(0, count - 1, num_points).round().astype(np.int64)
    else:
        repeats = int(np.ceil(num_points / count))
        indices = np.tile(np.arange(count, dtype=np.int64), repeats)[:num_points]
    return points[indices].copy(), indices


def route_points(
    centered: np.ndarray,
    grid: tuple[int, int, int],
    level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Call the production tokenizer routing functions, rather than duplicating them."""
    coordinates = torch.from_numpy(centered).float().unsqueeze(0)
    partition = WindowPartition(
        grid=grid,
        shifted=False,
        shift_axes=(False, True, True),
        level=level,
    )
    routing = LocalWindowTokenizer._routing_coordinates(coordinates)
    window_ids = LocalWindowTokenizer._flat_window_indices(routing, partition)
    return routing.squeeze(0).cpu().numpy(), window_ids.squeeze(0).cpu().numpy()


def window_label(index: int, grid: tuple[int, int, int]) -> str:
    forward, lateral, height = np.unravel_index(index, grid)
    return f"F{forward}-L{lateral}-H{height}"


def add_partition_planes(
    axis,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Draw routing partition planes using display axes x=L, y=F, z=H."""
    forward, lateral, height = bounds
    plane_style = {"facecolor": "#64748b", "edgecolor": "#475569", "alpha": 0.055}

    for value in lateral[1:-1]:
        vertices = [[
            (value, forward[0], height[0]),
            (value, forward[-1], height[0]),
            (value, forward[-1], height[-1]),
            (value, forward[0], height[-1]),
        ]]
        axis.add_collection3d(Poly3DCollection(vertices, **plane_style))
    for value in forward[1:-1]:
        vertices = [[
            (lateral[0], value, height[0]),
            (lateral[-1], value, height[0]),
            (lateral[-1], value, height[-1]),
            (lateral[0], value, height[-1]),
        ]]
        axis.add_collection3d(Poly3DCollection(vertices, **plane_style))
    for value in height[1:-1]:
        vertices = [[
            (lateral[0], forward[0], value),
            (lateral[-1], forward[0], value),
            (lateral[-1], forward[-1], value),
            (lateral[0], forward[-1], value),
        ]]
        axis.add_collection3d(Poly3DCollection(vertices, **plane_style))


def draw_bbox(axis, bounds: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    forward, lateral, height = bounds
    x0, x1 = lateral[0], lateral[-1]
    y0, y1 = forward[0], forward[-1]
    z0, z1 = height[0], height[-1]
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1),
    ]
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7),
        (6, 7), (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start, end in edges:
        a, b = corners[start], corners[end]
        axis.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color="#334155", linewidth=0.85, alpha=0.7,
        )


def plot_partition(
    centered: np.ndarray,
    window_ids: np.ndarray,
    grid: tuple[int, int, int],
    level_name: str,
    source_label: str,
    output_path: Path,
    dpi: int,
) -> dict[str, object]:
    num_windows = int(np.prod(grid))
    palette = plt.get_cmap("tab20", num_windows)
    colors = palette(window_ids)
    minimum = centered.min(axis=0)
    maximum = centered.max(axis=0)
    bounds = tuple(
        np.linspace(minimum[axis], maximum[axis], grid[axis] + 1)
        for axis in range(3)
    )
    counts = np.bincount(window_ids, minlength=num_windows)

    figure = plt.figure(figsize=(16, 9.6))
    spec = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.22, 1.0),
        height_ratios=(1.0, 0.72),
        left=0.045,
        right=0.985,
        bottom=0.17,
        top=0.83,
        wspace=0.24,
        hspace=0.42,
    )
    axis_3d = figure.add_subplot(spec[:, 0], projection="3d")
    axis_front = figure.add_subplot(spec[0, 1])
    axis_counts = figure.add_subplot(spec[1, 1])

    # Display coordinates: x=lateral, y=forward, z=height.
    axis_3d.scatter(
        centered[:, 1], centered[:, 0], centered[:, 2],
        c=colors, s=8, alpha=0.9, linewidths=0, depthshade=False,
    )
    add_partition_planes(axis_3d, bounds)
    draw_bbox(axis_3d, bounds)
    axis_3d.set_xlabel("Lateral (m)", labelpad=9)
    axis_3d.set_ylabel("Forward (m)", labelpad=9)
    axis_3d.set_zlabel("Height (m)", labelpad=9)
    axis_3d.set_title("3D routing view", fontsize=13, fontweight="bold", pad=7)
    axis_3d.view_init(elev=13, azim=-62)
    # Preserve the tall human shape without making the horizontal axes unreadably thin.
    axis_3d.set_box_aspect((0.72, 0.62, 1.9))
    axis_3d.grid(False)

    axis_front.scatter(
        centered[:, 1], centered[:, 2], c=colors, s=10, alpha=0.9, linewidths=0,
    )
    for value in bounds[1][1:-1]:
        axis_front.axvline(value, color="#334155", linewidth=1.0, alpha=0.65)
    for value in bounds[2][1:-1]:
        axis_front.axhline(value, color="#334155", linewidth=1.0, alpha=0.65)
    axis_front.set_aspect("equal", adjustable="box")
    axis_front.set_xlabel("Lateral (m)")
    axis_front.set_ylabel("Height (m)")
    axis_front.set_title(
        "Front view (forward axis collapsed)", fontsize=12.5, fontweight="bold", pad=8
    )
    axis_front.grid(False)

    labels = [window_label(index, grid) for index in range(num_windows)]
    bars = axis_counts.bar(
        np.arange(num_windows), counts,
        color=[palette(index) for index in range(num_windows)],
        edgecolor="#334155", linewidth=0.55,
    )
    axis_counts.bar_label(bars, labels=[str(value) for value in counts], padding=2, fontsize=8)
    axis_counts.set_xticks(
        np.arange(num_windows), labels, rotation=48, ha="right", fontsize=8
    )
    axis_counts.set_ylabel("Sampled points")
    axis_counts.set_title("Points routed to each window", fontsize=13, fontweight="bold")
    axis_counts.spines[["top", "right"]].set_visible(False)

    grid_text = " x ".join(str(value) for value in grid)
    figure.suptitle(
        f"MPH-Gait Tokenizer: {level_name} regular partition "
        f"({grid_text} = {num_windows} windows)",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        f"Real sample: {source_label} | {centered.shape[0]:,} deterministic points | "
        "[forward, lateral, height] metric coordinates",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    figure.text(
        0.5,
        0.065,
        "Exact path: camera XYZ mm -> [forward, lateral, height] m -> deterministic "
        "sampling -> per-frame centering -> min-max routing -> regular-window index",
        ha="center",
        fontsize=9.3,
        color="#334155",
    )
    figure.text(
        0.5,
        0.035,
        "Point colors correspond to the window IDs and colors shown in the occupancy chart.",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "grid": list(grid),
        "num_windows": num_windows,
        "window_labels": labels,
        "point_counts": counts.astype(int).tolist(),
        "valid_windows_min_2_points": (counts >= 2).tolist(),
        "output_png": str(output_path.resolve()),
        "output_svg": str(output_path.with_suffix(".svg").resolve()),
    }


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    raw = np.load(source)
    raw = np.asarray(raw[:, :3], dtype=np.float32)
    finite = np.isfinite(raw).all(axis=1)
    raw = raw[finite]
    axis_aligned = adapt_point_coordinates(raw, AXIS_ALIGNED_METRIC_COORDINATES)
    sampled, sampled_indices = deterministic_sample(axis_aligned, args.num_points)
    center = sampled.mean(axis=0, keepdims=True)
    centered = sampled - center

    fine_routing, fine_ids = route_points(centered, (2, 2, 4), 0.0)
    coarse_routing, coarse_ids = route_points(centered, (1, 2, 2), 1.0)
    if not np.allclose(fine_routing, coarse_routing):
        raise RuntimeError("Fine and coarse routing coordinates unexpectedly differ")

    source_label = "P003 / C4 / V001 / frame 060"
    fine = plot_partition(
        centered, fine_ids, (2, 2, 4), "Fine", source_label,
        output_dir / "tokenizer_fine16_real_frame.png", args.dpi,
    )
    coarse = plot_partition(
        centered, coarse_ids, (1, 2, 2), "Coarse", source_label,
        output_dir / "tokenizer_coarse4_real_frame.png", args.dpi,
    )

    metadata = {
        "description": "Exact production routing visualization before learned token pooling",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_valid_points": int(raw.shape[0]),
        "coordinate_adapter": AXIS_ALIGNED_METRIC_COORDINATES,
        "coordinate_order": ["forward", "lateral", "height_up"],
        "sampling": "evaluation deterministic linspace/round",
        "sampled_points": int(sampled.shape[0]),
        "sampled_index_first_last": [int(sampled_indices[0]), int(sampled_indices[-1])],
        "sample_center_m": center.squeeze(0).tolist(),
        "centered_min_m": centered.min(axis=0).tolist(),
        "centered_max_m": centered.max(axis=0).tolist(),
        "routing_min": fine_routing.min(axis=0).tolist(),
        "routing_max": fine_routing.max(axis=0).tolist(),
        "fine": fine,
        "coarse": coarse,
        "important_note": (
            "The figures visualize exact window routing. Learned content scores, "
            "position penalties, point-feature pooling, and Transformer mixing occur after routing."
        ),
    }
    metadata_path = output_dir / "tokenizer_partition_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"fine_png={fine['output_png']}")
    print(f"coarse_png={coarse['output_png']}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
