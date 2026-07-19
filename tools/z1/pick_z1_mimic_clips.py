#!/usr/bin/env python3
"""Select representative windows from Z1 mimic NPZ motions.

This follows the same shape as ``magicbot-XS/pick_xs_clips.py``:
load each source clip, slide short windows, score candidate windows, then save
the picked windows as standalone RobotState NPZ files for UFO.

Z1 mimic motions are in-place (root xy is fixed), so the score uses joint and
body activity instead of root-speed/yaw categories.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SRC_FOLDER = Path(
    r"D:\Desktop_Files\humanoid-terrain-dynamics\magicbot-mimic\source\whole_body_tracking\whole_body_tracking\datasets\mimic"
)
DEFAULT_OUT_FOLDER = REPO_ROOT / "humanoidverse" / "data" / "z1_mimic_robot_state_npz_selected"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "z1_23dof.yaml"

WINDOW_OPTIONS_SEC = (2.5, 3.0, 3.5, 4.0, 5.0)
MAX_OVERLAP_FRAC = 0.35


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    score: float
    mean_activity: float
    dof_span: float
    body_span: float

    @property
    def frames(self) -> int:
        return self.end - self.start


def load_joint_names(robot_config: Path) -> np.ndarray:
    config = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
    return np.asarray(config["control_joints"]["names"], dtype="<U32")


def normalize_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5.0, 95.0])
    scale = max(float(hi - lo), 1e-6)
    return np.clip((values - lo) / scale, 0.0, 1.0)


def per_frame_activity(data: dict[str, np.ndarray], fps: float) -> np.ndarray:
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)

    if "joint_vel" in data:
        joint_speed = np.linalg.norm(np.asarray(data["joint_vel"], dtype=np.float64)[:-1], axis=1) / np.sqrt(joint_pos.shape[1])
    else:
        joint_speed = np.linalg.norm(np.diff(joint_pos, axis=0), axis=1) * fps / np.sqrt(joint_pos.shape[1])

    body_speed = np.zeros_like(joint_speed)
    if "body_pos_w" in data:
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        if body_pos.ndim == 3 and body_pos.shape[0] == joint_pos.shape[0]:
            body_delta = np.diff(body_pos[:, 1:, :], axis=0)
            body_speed = np.linalg.norm(body_delta, axis=2).mean(axis=1) * fps

    root_ang_speed = np.zeros_like(joint_speed)
    if "root_quat" in data:
        quat = np.asarray(data["root_quat"], dtype=np.float64)
        if quat.shape[0] == joint_pos.shape[0]:
            dots = np.abs(np.sum(quat[:-1] * quat[1:], axis=1))
            dots = np.clip(dots, 0.0, 1.0)
            root_ang_speed = 2.0 * np.arccos(dots) * fps

    return (
        0.55 * normalize_signal(joint_speed)
        + 0.35 * normalize_signal(body_speed)
        + 0.10 * normalize_signal(root_ang_speed)
    )


def score_windows(data: dict[str, np.ndarray], fps: float) -> list[Candidate]:
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    total = joint_pos.shape[0]
    if total < 4:
        return []

    activity = per_frame_activity(data, fps)
    candidates: list[Candidate] = []
    max_window = min(max(WINDOW_OPTIONS_SEC), total / fps)
    window_options = [sec for sec in WINDOW_OPTIONS_SEC if sec <= max_window + 1e-6]
    if not window_options:
        window_options = [total / fps]

    body_pos = None
    if "body_pos_w" in data:
        raw_body = np.asarray(data["body_pos_w"], dtype=np.float64)
        if raw_body.ndim == 3 and raw_body.shape[0] == total:
            body_pos = raw_body[:, 1:, :]

    for window_sec in window_options:
        window_frames = min(total, max(4, int(round(window_sec * fps))))
        step = max(1, window_frames // 4)
        starts = list(range(0, max(1, total - window_frames + 1), step))
        final_start = max(0, total - window_frames)
        if final_start not in starts:
            starts.append(final_start)

        for start in starts:
            end = min(total, start + window_frames)
            if end - start < 4:
                continue
            act = activity[start : max(start, end - 1)]
            mean_activity = float(act.mean()) if act.size else 0.0
            dof_span = float(np.mean(np.ptp(joint_pos[start:end], axis=0)))
            body_span = 0.0
            if body_pos is not None:
                body_span = float(np.mean(np.ptp(body_pos[start:end], axis=0)))
            duration_bonus = min((end - start) / max(1.0, 4.0 * fps), 1.0) * 0.05
            score = mean_activity + 0.20 * dof_span + 0.15 * body_span + duration_bonus
            candidates.append(
                Candidate(
                    start=start,
                    end=end,
                    score=float(score),
                    mean_activity=mean_activity,
                    dof_span=dof_span,
                    body_span=body_span,
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def overlap_fraction(a: Candidate, b: Candidate) -> float:
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
    return overlap / max(1, min(a.frames, b.frames))


def pick_windows(candidates: list[Candidate], picks_per_source: int) -> list[Candidate]:
    picked: list[Candidate] = []
    for candidate in candidates:
        if all(overlap_fraction(candidate, chosen) <= MAX_OVERLAP_FRAC for chosen in picked):
            picked.append(candidate)
            if len(picked) == picks_per_source:
                return picked

    for candidate in candidates:
        if candidate not in picked:
            picked.append(candidate)
            if len(picked) == picks_per_source:
                return picked
    return picked


def write_robot_state_npz(
    src_path: Path,
    out_path: Path,
    data: dict[str, np.ndarray],
    start: int,
    end: int,
    fps: float,
    joint_names: np.ndarray,
    score: Candidate,
) -> None:
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)[start:end]
    root_quat = np.asarray(data["root_quat"], dtype=np.float32)[start:end]
    dof_pos = np.asarray(data["joint_pos"], dtype=np.float32)[start:end]
    motion_key = out_path.stem
    np.savez_compressed(
        out_path,
        root_pos=root_pos,
        root_quat=root_quat,
        dof_pos=dof_pos,
        fps=np.asarray([fps], dtype=np.float32),
        joint_names=joint_names,
        motion_key=np.asarray(motion_key),
        source_path=np.asarray(str(src_path)),
        source_motion_key=np.asarray(src_path.stem),
        clip_start_frame=np.asarray(start, dtype=np.int32),
        clip_end_frame=np.asarray(end, dtype=np.int32),
        selector_score=np.asarray(score.score, dtype=np.float32),
        selector_mean_activity=np.asarray(score.mean_activity, dtype=np.float32),
        selector_dof_span=np.asarray(score.dof_span, dtype=np.float32),
        selector_body_span=np.asarray(score.body_span, dtype=np.float32),
        selector_method=np.asarray("joint_body_activity_greedy_v1"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_folder", default=str(DEFAULT_SRC_FOLDER))
    parser.add_argument("--out_folder", default=str(DEFAULT_OUT_FOLDER))
    parser.add_argument("--robot_config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--picks_per_source", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Remove the output folder before writing.")
    parser.add_argument("--dry_run", action="store_true", help="Print the selection without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_folder = Path(args.src_folder)
    out_folder = Path(args.out_folder)
    robot_config = Path(args.robot_config)
    joint_names = load_joint_names(robot_config)

    src_files = sorted(src_folder.glob("*.npz"))
    print(f"Source folder: {src_folder}")
    print(f"Source motions: {len(src_files)}")
    print(f"Picks per source: {args.picks_per_source}")
    print(f"Output folder: {out_folder}")

    if len(src_files) != 31:
        raise SystemExit(f"Expected 31 source motions, got {len(src_files)}")
    if not args.dry_run and args.force and out_folder.exists():
        shutil.rmtree(out_folder)
    if not args.dry_run:
        out_folder.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for src_path in src_files:
        with np.load(src_path, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files}
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        candidates = score_windows(data, fps)
        picked = pick_windows(candidates, args.picks_per_source)
        if len(picked) != args.picks_per_source:
            raise RuntimeError(f"{src_path.name}: selected {len(picked)} windows, expected {args.picks_per_source}")

        for pick_idx, candidate in enumerate(picked):
            start_ms = int(round(candidate.start / fps * 1000.0))
            end_ms = int(round(candidate.end / fps * 1000.0))
            out_name = f"{src_path.stem}__pick{pick_idx:03d}_{start_ms:06d}_{end_ms:06d}.npz"
            out_path = out_folder / out_name
            if not args.dry_run:
                write_robot_state_npz(src_path, out_path, data, candidate.start, candidate.end, fps, joint_names, candidate)
            manifest_rows.append(
                {
                    "clip": out_name,
                    "source": src_path.name,
                    "pick_index": pick_idx,
                    "start_frame": candidate.start,
                    "end_frame": candidate.end,
                    "start_s": candidate.start / fps,
                    "end_s": candidate.end / fps,
                    "duration_s": candidate.frames / fps,
                    "score": candidate.score,
                    "mean_activity": candidate.mean_activity,
                    "dof_span": candidate.dof_span,
                    "body_span": candidate.body_span,
                }
            )
            print(
                f"  {src_path.stem:<18} pick{pick_idx} "
                f"[{candidate.start / fps:6.2f}-{candidate.end / fps:6.2f}s] "
                f"score={candidate.score:.3f} activity={candidate.mean_activity:.3f}"
            )

    if not args.dry_run:
        manifest_path = out_folder / "_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Manifest written: {manifest_path}")
    print(f"Selected windows: {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
