#!/usr/bin/env python3
"""Convert raw Z1 fall-and-get-up clips into UFO RobotState NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "humanoidverse" / "data" / "robots" / "z1_mimic_dataset" / "z1_mimic_recovery_npz" / "train_npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "humanoidverse" / "data" / "robots" / "z1_mimic_dataset" / "z1_mimic_recovery_npz" / "robot_state_npz"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "z1_23dof.yaml"


def _as_string_list(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for value in np.asarray(values).reshape(-1):
        result.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return result


def _require_array(data: np.lib.npyio.NpzFile, key: str, ndim: int, trailing_shape: tuple[int, ...]) -> np.ndarray:
    if key not in data:
        raise ValueError(f"missing required field '{key}'")
    value = np.asarray(data[key], dtype=np.float32)
    expected_shape = (value.shape[0], *trailing_shape)
    if value.ndim != ndim or value.shape != expected_shape:
        raise ValueError(f"{key} must have shape [T, {', '.join(map(str, trailing_shape))}], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains non-finite values")
    return value


def convert_file(input_path: Path, output_path: Path, control_joint_names: list[str]) -> None:
    with np.load(input_path, allow_pickle=True) as data:
        joint_pos = _require_array(data, "joint_pos", 2, (len(control_joint_names),))
        body_pos_w = _require_array(data, "body_pos_w", 3, (np.asarray(data["body_pos_w"]).shape[1], 3))
        body_quat_w = _require_array(data, "body_quat_w", 3, (np.asarray(data["body_quat_w"]).shape[1], 4))
        if body_pos_w.shape[:2] != body_quat_w.shape[:2] or body_pos_w.shape[0] != joint_pos.shape[0]:
            raise ValueError("joint_pos, body_pos_w, and body_quat_w must share frame count and body count")
        if "body_names" not in data or "joint_names" not in data or "fps" not in data:
            raise ValueError("missing one of required fields: body_names, joint_names, fps")

        body_names = _as_string_list(data["body_names"])
        source_joint_names = _as_string_list(data["joint_names"])
        if "pelvis" not in body_names:
            raise ValueError("body_names has no 'pelvis' entry")
        if len(body_names) != body_pos_w.shape[1]:
            raise ValueError("body_names length does not match body_pos_w body dimension")
        if len(source_joint_names) != joint_pos.shape[1] or set(source_joint_names) != set(control_joint_names):
            raise ValueError("joint_names must be an exact permutation of the Z1 controlled joint names")

        pelvis_idx = body_names.index("pelvis")
        root_pos = body_pos_w[:, pelvis_idx, :].copy()
        root_quat = body_quat_w[:, pelvis_idx, :].copy()
        quat_norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
        if np.any(quat_norm <= 0.0) or not np.all(np.isfinite(quat_norm)):
            raise ValueError("pelvis quaternion contains a zero or non-finite value")
        root_quat /= quat_norm

        # Keep the whole clip in one grounded world frame, matching the existing Z1 NPZ viewer.
        ground_offset = float(np.min(body_pos_w[:, :, 2]))
        root_pos[:, 2] -= ground_offset
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"invalid fps={fps}")

    motion_key = f"recovery_{input_path.stem}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        root_pos=root_pos.astype(np.float32),
        root_quat=root_quat.astype(np.float32),
        dof_pos=joint_pos.astype(np.float32),
        fps=np.asarray([fps], dtype=np.float32),
        joint_names=np.asarray(source_joint_names),
        motion_key=np.asarray(motion_key),
        source_path=np.asarray(str(input_path.resolve())),
        source_format=np.asarray("z1_recovery_schema_v2"),
        root_body_name=np.asarray("pelvis"),
        quaternion_order=np.asarray("xyzw"),
        ground_offset=np.asarray(ground_offset, dtype=np.float32),
        converter_version=np.asarray("z1_recovery_robot_state_v1"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")
    control_joint_names = list(load_robot_spec(args.robot_config).control_joint_names)

    converted = 0
    skipped = 0
    for input_path in files:
        output_path = output_dir / input_path.name
        if output_path.exists() and not args.force:
            print(f"[skip] {output_path.name}")
            skipped += 1
            continue
        convert_file(input_path, output_path, control_joint_names)
        print(f"[converted] {input_path.name} -> {output_path}")
        converted += 1
    print(f"Completed: converted={converted}, skipped={skipped}, total={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
