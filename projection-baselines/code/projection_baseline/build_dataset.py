#!/usr/bin/env python3
"""Build projected image datasets from foreground point-cloud gait frames.

Default mode follows the LidarGait / SUSTech1K preprocessing style:
range-view point-to-depth projection, 64x64 aligned images, and RGB depth
images produced by applying a Jet-like colormap to normalized depth.

The source dataset is expected to look like:

    dataset/PersonRecognitionWalking_29/person_001/P1_C1_V001/clear_data_001.npy

The output is grouped by projection mode, then mirrors the source folder
structure under each mode directory, for example
``dataset_proj/PersonRecognitionWalking_29/lidargait_rgb/person_001/...``.
It writes one PNG per point-cloud frame and does not drop the first 30 frames;
projection datasets usually keep every projected frame and let the training
loader decide clip sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Pillow is required to write PNG files. Install it in the pytorch env: "
        "conda install -n pytorch pillow"
    ) from exc


VIDEO_RE = re.compile(r"P(\d+)_C(\d+)(?:_(.+))?_V(\d+)")
FRAME_RE = re.compile(r"clear_data_(\d+)\.npy")
LIDARGAIT_MODES = ("lidargait_rgb", "lidargait_gray", "lidargait_silhouette")
ALL_MODES = (*LIDARGAIT_MODES, "pseudo_rgb")


@dataclass(frozen=True)
class VideoInfo:
    person_id: int
    clothes_id: int
    condition_tag: str
    video_id: int


@dataclass(frozen=True)
class ProjectionBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                [self.x_min, self.x_max],
                [self.y_min, self.y_max],
                [self.z_min, self.z_max],
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class BuildResult:
    source_video: str
    target_video: str
    written_frames: int
    skipped_frames: int
    modes: tuple[str, ...]
    bounds: ProjectionBounds | None


def parse_video_name(name: str) -> VideoInfo | None:
    match = VIDEO_RE.fullmatch(name)
    if match is None:
        return None
    return VideoInfo(
        person_id=int(match.group(1)),
        clothes_id=int(match.group(2)),
        condition_tag=match.group(3) or "normal",
        video_id=int(match.group(4)),
    )


def frame_sort_key(path: Path) -> int:
    match = FRAME_RE.fullmatch(path.name)
    if match is None:
        return math.inf
    return int(match.group(1))


def iter_video_dirs(input_root: Path) -> Iterable[Path]:
    for person_dir in sorted(input_root.glob("person_*")):
        if not person_dir.is_dir():
            continue
        for video_dir in sorted(person_dir.glob("P*_C*_V*")):
            if video_dir.is_dir() and parse_video_name(video_dir.name) is not None:
                yield video_dir


def load_xyz(path: Path) -> np.ndarray:
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] point cloud, got {points.shape} from {path}")
    points = np.asarray(points[:, :3], dtype=np.float32)
    return points[np.isfinite(points).all(axis=1)]


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices]


def robust_bounds(
    points: np.ndarray,
    percentile_low: float,
    percentile_high: float,
    padding_ratio: float,
) -> ProjectionBounds:
    if len(points) == 0:
        raise ValueError("Cannot compute projection bounds from an empty point cloud")

    low = np.percentile(points, percentile_low, axis=0)
    high = np.percentile(points, percentile_high, axis=0)
    xy_span = np.maximum(high[:2] - low[:2], 1e-6)
    low[:2] -= xy_span * padding_ratio
    high[:2] += xy_span * padding_ratio
    span = np.maximum(high - low, 1e-6)
    high = low + span

    return ProjectionBounds(
        x_min=float(low[0]),
        x_max=float(high[0]),
        y_min=float(low[1]),
        y_max=float(high[1]),
        z_min=float(low[2]),
        z_max=float(high[2]),
    )


def compute_video_bounds(
    frame_paths: list[Path],
    percentile_low: float,
    percentile_high: float,
    padding_ratio: float,
    bounds_stride: int,
    bounds_max_points_per_frame: int,
) -> ProjectionBounds:
    sampled = []
    stride = max(1, bounds_stride)
    for frame_path in frame_paths[::stride]:
        points = load_xyz(frame_path)
        points = deterministic_subsample(points, bounds_max_points_per_frame)
        if len(points) > 0:
            sampled.append(points)

    if not sampled:
        raise ValueError("No valid points found while computing video bounds")

    points = np.concatenate(sampled, axis=0)
    return robust_bounds(points, percentile_low, percentile_high, padding_ratio)


def resize_image(image: np.ndarray, size: tuple[int, int], interpolation: int = Image.BICUBIC) -> np.ndarray:
    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L")
    else:
        pil_image = Image.fromarray(image, mode="RGB")
    return np.asarray(pil_image.resize(size, interpolation))


def align_image(image: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Center-align gait projection using the OpenGait/GaitSet-style crop.

    The height is normalized to ``image_size`` and the subject center is estimated
    from the horizontal cumulative foreground mass, then center-cropped to a
    square image.
    """
    if image.ndim == 2:
        mass = image.astype(np.float32)
    else:
        mass = image.astype(np.float32).sum(axis=2)

    if mass.sum() > 10000:
        y_nonzero = np.flatnonzero(mass.sum(axis=1) != 0)
        if len(y_nonzero) > 0:
            image = image[y_nonzero[0] : y_nonzero[-1] + 1]
            mass = mass[y_nonzero[0] : y_nonzero[-1] + 1]

    if image.shape[0] == 0 or image.shape[1] == 0:
        shape = (image_size, image_size) if image.ndim == 2 else (image_size, image_size, 3)
        return np.zeros(shape, dtype=np.uint8)

    ratio = image.shape[1] / max(image.shape[0], 1)
    resized_width = max(1, int(round(image_size * ratio)))
    image = resize_image(image.astype(np.uint8), (resized_width, image_size))

    if image.ndim == 2:
        mass = image.astype(np.float32)
    else:
        mass = image.astype(np.float32).sum(axis=2)

    x_center = image.shape[1] // 2
    total = mass.sum()
    if total > 0:
        x_csum = mass.sum(axis=0).cumsum()
        x_center = int(np.searchsorted(x_csum, total / 2.0))

    half_width = image_size // 2
    left = x_center - half_width
    right = left + image_size

    pad_left = max(0, -left)
    pad_right = max(0, right - image.shape[1])
    if pad_left or pad_right:
        if image.ndim == 2:
            image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode="constant")
        else:
            image = np.pad(image, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
        left += pad_left
        right += pad_left

    return image[:, left:right].astype(np.uint8)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.round(np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def normalize_occupied_depth(depth_map: np.ndarray, inverse: bool = True) -> np.ndarray:
    normalized = np.zeros(depth_map.shape, dtype=np.float32)
    mask = depth_map > 0
    if not mask.any():
        return normalized

    occupied_depth = depth_map[mask]
    d_min = float(occupied_depth.min())
    d_max = float(occupied_depth.max())
    if d_max - d_min < 1e-6:
        normalized[mask] = 1.0
    else:
        normalized[mask] = (occupied_depth - d_min) / (d_max - d_min)
        if inverse:
            normalized[mask] = 1.0 - normalized[mask]
    return normalized


def kinect_to_lidar_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map Azure/Kinect camera XYZ to LidarGait-style axes.

    The local dataset appears to store Kinect-like coordinates: X horizontal,
    Y vertical/down, Z forward/depth. LidarGait's range-view formula expects
    forward and horizontal axes in the ground plane plus a vertical axis.
    """
    forward = points[:, 2]
    horizontal = points[:, 0]
    vertical = -points[:, 1]
    return forward, horizontal, vertical


def raw_lidar_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = -points[:, 0]
    horizontal = -points[:, 1]
    vertical = points[:, 2]
    return forward, horizontal, vertical


def lidar_range_depth_map(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """LidarGait-style cylindrical/range-view projection.

    Horizontal coordinate: atan2(horizontal, forward) / h_res.
    Vertical coordinate: atan2(vertical, ground_distance) / v_res.
    Pixel value: ground-plane distance; collisions keep the closest point.
    """
    if len(points) == 0:
        return np.zeros((args.range_height, args.range_width), dtype=np.float32)

    if args.coord_system == "lidar":
        forward, horizontal, vertical = raw_lidar_axes(points)
    else:
        forward, horizontal, vertical = kinect_to_lidar_axes(points)

    ground_distance = np.sqrt(forward ** 2 + horizontal ** 2)
    valid = np.isfinite(ground_distance) & (ground_distance > 1e-6)
    if not valid.any():
        return np.zeros((args.range_height, args.range_width), dtype=np.float32)

    forward = forward[valid]
    horizontal = horizontal[valid]
    vertical = vertical[valid]
    ground_distance = ground_distance[valid]

    azimuth = np.degrees(np.arctan2(horizontal, forward))
    elevation = np.degrees(np.arctan2(vertical, ground_distance))

    col = np.floor((azimuth + args.h_fov / 2.0) / args.h_res).astype(np.int32)
    # Image coordinates grow downward. Higher elevation should appear near the
    # top of the image, so invert the vertical angle when mapping to rows.
    row = np.floor((args.v_fov_max - elevation) / args.v_res).astype(np.int32)

    valid = (
        (col >= 0)
        & (col < args.range_width)
        & (elevation >= args.v_fov_min)
        & (elevation <= args.v_fov_max)
    )
    if not valid.any():
        return np.zeros((args.range_height, args.range_width), dtype=np.float32)

    row = np.clip(row[valid], 0, args.range_height - 1)
    col = col[valid]
    ground_distance = ground_distance[valid]

    depth_map = np.zeros((args.range_height, args.range_width), dtype=np.float32)
    order = np.argsort(ground_distance)[::-1]
    depth_map[row[order], col[order]] = ground_distance[order]
    return depth_map


def lidargait_projection_image(points: np.ndarray, mode: str, args: argparse.Namespace) -> np.ndarray:
    depth_map = lidar_range_depth_map(points, args)

    if mode == "lidargait_silhouette":
        image = (depth_map > 0).astype(np.uint8) * 255
        return align_image(image, args.image_size)

    normalized = normalize_occupied_depth(depth_map, inverse=True)
    if mode == "lidargait_gray":
        image = np.round(normalized * 255).astype(np.uint8)
        return align_image(image, args.image_size)

    image = jet_colormap(normalized)
    image[depth_map <= 0] = 0
    return align_image(image, args.image_size)


def pseudo_rgb_projection_image(
    points: np.ndarray,
    bounds: ProjectionBounds,
    image_width: int,
    image_height: int,
    flip_y: bool,
) -> np.ndarray:
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    if len(points) == 0:
        return image

    b = bounds.as_array()
    x_min, x_max = b[0]
    y_min, y_max = b[1]
    z_min, z_max = b[2]

    in_bounds = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    points = points[in_bounds]
    if len(points) == 0:
        return image

    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    z_span = max(float(z_max - z_min), 1e-6)

    u = np.floor((points[:, 0] - x_min) / x_span * (image_width - 1)).astype(np.int32)
    if flip_y:
        v_float = (y_max - points[:, 1]) / y_span * (image_height - 1)
    else:
        v_float = (points[:, 1] - y_min) / y_span * (image_height - 1)
    v = np.floor(v_float).astype(np.int32)

    u = np.clip(u, 0, image_width - 1)
    v = np.clip(v, 0, image_height - 1)

    counts = np.zeros((image_height, image_width), dtype=np.int32)
    np.add.at(counts, (v, u), 1)

    order = np.argsort(points[:, 2])[::-1]
    visible_points = points[order]
    visible_u = u[order]
    visible_v = v[order]

    inverse_depth = 1.0 - (visible_points[:, 2] - z_min) / z_span
    inverse_depth = np.clip(inverse_depth, 0.0, 1.0)

    if flip_y:
        height_channel = (y_max - visible_points[:, 1]) / y_span
    else:
        height_channel = (visible_points[:, 1] - y_min) / y_span
    height_channel = np.clip(height_channel, 0.0, 1.0)

    image[visible_v, visible_u, 0] = np.round(inverse_depth * 255).astype(np.uint8)
    image[visible_v, visible_u, 1] = np.round(height_channel * 255).astype(np.uint8)

    occupied = counts > 0
    if occupied.any():
        density = np.zeros_like(counts, dtype=np.float32)
        density[occupied] = np.log1p(counts[occupied]) / np.log1p(counts.max())
        image[:, :, 2] = np.round(density * 255).astype(np.uint8)

    return image


def modes_from_arg(mode: str) -> tuple[str, ...]:
    if mode == "all":
        return LIDARGAIT_MODES
    return (mode,)


def output_frame_name(frame_path: Path, prefix: str) -> str:
    match = FRAME_RE.fullmatch(frame_path.name)
    if match is None:
        return f"{prefix}_{frame_path.stem}.png"
    return f"{prefix}_{int(match.group(1)):03d}.png"


def write_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image, mode="L").save(output_path)
    else:
        Image.fromarray(image, mode="RGB").save(output_path)


def build_video_projection(args_tuple: tuple[Path, Path, argparse.Namespace]) -> BuildResult:
    video_dir, target_video_dir, args = args_tuple
    frame_paths = sorted(video_dir.glob("clear_data_*.npy"), key=frame_sort_key)
    if args.max_frames_per_video is not None:
        frame_paths = frame_paths[: args.max_frames_per_video]

    modes = modes_from_arg(args.mode)
    needs_pseudo_bounds = "pseudo_rgb" in modes
    if needs_pseudo_bounds and args.bounds == "video":
        bounds = compute_video_bounds(
            frame_paths=frame_paths,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            padding_ratio=args.padding_ratio,
            bounds_stride=args.bounds_stride,
            bounds_max_points_per_frame=args.bounds_max_points_per_frame,
        )
    else:
        bounds = None

    written = 0
    skipped = 0
    relative_video_dir = video_dir.relative_to(args.input_root)

    for frame_path in frame_paths:
        points = load_xyz(frame_path)
        frame_bounds = None
        for mode in modes:
            mode_target_video_dir = args.output_root / mode / relative_video_dir
            prefix = args.output_prefix or mode
            if len(modes) > 1 and args.output_prefix:
                prefix = f"{args.output_prefix}_{mode}"
            output_path = mode_target_video_dir / output_frame_name(frame_path, prefix)
            if output_path.exists() and args.skip_existing:
                skipped += 1
                continue

            if mode == "pseudo_rgb":
                if bounds is not None:
                    frame_bounds = bounds
                elif frame_bounds is None:
                    frame_bounds = robust_bounds(
                        points,
                        percentile_low=args.percentile_low,
                        percentile_high=args.percentile_high,
                        padding_ratio=args.padding_ratio,
                    )
                image = pseudo_rgb_projection_image(
                    points=points,
                    bounds=frame_bounds,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    flip_y=args.flip_y,
                )
            else:
                image = lidargait_projection_image(points, mode, args)

            write_png(image, output_path)
            written += 1

    if args.write_video_meta:
        info = parse_video_name(video_dir.name)
        for mode in modes:
            mode_target_video_dir = args.output_root / mode / relative_video_dir
            meta = {
                "source_video": str(video_dir),
                "target_video": str(mode_target_video_dir),
                "video_info": asdict(info) if info is not None else None,
                "frame_count": len(frame_paths),
                "mode": mode,
                "lidargait_image_size": args.image_size,
                "pseudo_rgb_image_width": args.image_width,
                "pseudo_rgb_image_height": args.image_height,
                "coord_system": args.coord_system,
                "h_res": args.h_res,
                "v_res": args.v_res,
                "v_fov": [args.v_fov_min, args.v_fov_max],
                "vertical_row_mapping": "top row = larger elevation angle",
                "bounds_mode": args.bounds,
                "bounds": asdict(bounds) if bounds is not None else None,
                "channels": {
                    "lidargait_rgb": "Jet RGB colormap over inverse normalized range depth",
                    "lidargait_gray": "1-channel inverse normalized range depth",
                    "lidargait_silhouette": "1-channel binary projected occupancy",
                    "pseudo_rgb": "R inverse frontal depth, G normalized vertical position, B log density",
                },
            }
            mode_target_video_dir.mkdir(parents=True, exist_ok=True)
            (mode_target_video_dir / "projection_meta.json").write_text(json.dumps(meta, indent=2))

    return BuildResult(
        source_video=str(video_dir),
        target_video=str(args.output_root / "<mode>" / relative_video_dir),
        written_frames=written,
        skipped_frames=skipped,
        modes=modes,
        bounds=bounds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset_proj projection images from clear_data_*.npy point clouds."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("dataset/PersonRecognitionWalking_29"),
        help="Source point-cloud dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_proj/PersonRecognitionWalking_29"),
        help="Output projected-image dataset root.",
    )
    parser.add_argument(
        "--mode",
        choices=(*ALL_MODES, "all"),
        default="lidargait_rgb",
        help="Projection output mode. 'all' writes the three LidarGait-style modes.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Aligned square size for LidarGait-style projections. Default follows SUSTech1K.",
    )
    parser.add_argument("--image-width", type=int, default=64, help="Width for pseudo_rgb mode only.")
    parser.add_argument("--image-height", type=int, default=128, help="Height for pseudo_rgb mode only.")
    parser.add_argument(
        "--coord-system",
        choices=["kinect", "lidar"],
        default="kinect",
        help="Use kinect for local XYZ=[horizontal, vertical/down, depth]; lidar for OpenGait-like raw LiDAR axes.",
    )
    parser.add_argument("--h-res", type=float, default=0.19188, help="Horizontal angular resolution in degrees.")
    parser.add_argument("--v-res", type=float, default=0.2, help="Vertical angular resolution in degrees.")
    parser.add_argument("--h-fov", type=float, default=360.0, help="Horizontal field of view in degrees.")
    parser.add_argument("--v-fov-min", type=float, default=-25.0, help="Minimum vertical FOV in degrees.")
    parser.add_argument("--v-fov-max", type=float, default=15.0, help="Maximum vertical FOV in degrees.")
    parser.add_argument(
        "--bounds",
        choices=["video", "frame"],
        default="video",
        help="Use one robust crop per video or recompute bounds per frame for pseudo_rgb mode.",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument(
        "--bounds-stride",
        type=int,
        default=2,
        help="When using video bounds, estimate bounds from every Nth frame for pseudo_rgb mode.",
    )
    parser.add_argument(
        "--bounds-max-points-per-frame",
        type=int,
        default=4096,
        help="When estimating video bounds, cap sampled points per frame for pseudo_rgb mode.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip the projected vertical axis for pseudo_rgb mode.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional custom output prefix. Default uses the mode name.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-video-meta", action="store_true", default=True)
    parser.add_argument("--no-video-meta", dest="write_video_meta", action="store_false")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.range_width = int(math.ceil(args.h_fov / args.h_res))
    args.range_height = int(math.ceil((args.v_fov_max - args.v_fov_min) / args.v_res))
    return args


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root

    video_dirs = list(iter_video_dirs(input_root))
    if args.max_videos is not None:
        video_dirs = video_dirs[: args.max_videos]

    tasks = []
    total_frames = 0
    for video_dir in video_dirs:
        relative = video_dir.relative_to(input_root)
        target_video_dir = output_root / "<mode>" / relative
        frame_count = len(list(video_dir.glob("clear_data_*.npy")))
        if args.max_frames_per_video is not None:
            frame_count = min(frame_count, args.max_frames_per_video)
        total_frames += frame_count
        tasks.append((video_dir, target_video_dir, args))

    modes = modes_from_arg(args.mode)
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Videos: {len(tasks)}")
    print(f"Frames to project: {total_frames}")
    print(f"Modes: {', '.join(modes)}")
    print(f"LidarGait image size: {args.image_size}x{args.image_size}")
    print(f"Range map before alignment: {args.range_width}x{args.range_height}")
    print(f"Coord system: {args.coord_system}")
    print(f"Output layout: {output_root}/<mode>/person_xxx/video_xxx/*.png")

    if args.dry_run:
        return

    if args.workers <= 1:
        results = [build_video_projection(task) for task in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(build_video_projection, tasks))

    written = sum(result.written_frames for result in results)
    skipped = sum(result.skipped_frames for result in results)
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "video_count": len(results),
        "written_frames": written,
        "skipped_frames": skipped,
        "modes": list(modes),
        "lidargait_image_size": args.image_size,
        "pseudo_rgb_image_width": args.image_width,
        "pseudo_rgb_image_height": args.image_height,
        "coord_system": args.coord_system,
        "h_res": args.h_res,
        "v_res": args.v_res,
        "h_fov": args.h_fov,
        "v_fov": [args.v_fov_min, args.v_fov_max],
        "range_width": args.range_width,
        "range_height": args.range_height,
        "output_layout": "<output_root>/<mode>/<person>/<video>/*.png",
        "vertical_row_mapping": "top row = larger elevation angle",
        "channels": {
            "lidargait_rgb": "Jet RGB colormap over inverse normalized range depth",
            "lidargait_gray": "1-channel inverse normalized range depth",
            "lidargait_silhouette": "1-channel binary projected occupancy",
            "pseudo_rgb": "R inverse frontal depth, G normalized vertical position, B log density",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "projection_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Done. Written frames: {written}, skipped frames: {skipped}")
    print(f"Manifest: {output_root / 'projection_manifest.json'}")


if __name__ == "__main__":
    main()
