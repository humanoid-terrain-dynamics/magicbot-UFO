#!/usr/bin/env python3
"""Shared helpers for Z1 AMP NPZ viewers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import mujoco as mj
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
Z1_MJCF = str(
    REPO_ROOT
    / "humanoidverse"
    / "data"
    / "robots"
    / "magicbot_z1_description"
    / "mjcf"
    / "MAGICBOTZ1.xml"
)

DEFAULT_CAMERA_DISTANCE = 3.0
DEFAULT_CAMERA_ELEVATION = -20.0
DEFAULT_CAMERA_AZIMUTH = 90.0

Z1_POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
]


@dataclass
class MotionClip:
    path: Path
    root_pos: np.ndarray
    root_quat: np.ndarray
    dof_pos: np.ndarray
    fps: float
    num_frames: int
    ground_offset: float | None = None

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


def load_full_length_npz(npz_path: str | Path) -> MotionClip:
    data = np.load(npz_path)
    joint_pos = data["joint_pos"]
    fps = float(data["fps"][0])
    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    body_names = data["body_names"]

    pelvis_idx = np.where(body_names == "pelvis")[0]
    if not len(pelvis_idx):
        pelvis_idx = np.where(body_names == b"pelvis")[0]
    pelvis_idx = int(pelvis_idx[0]) if len(pelvis_idx) else 1

    root_pos = body_pos_w[:, pelvis_idx, :].copy()
    root_quat = body_quat_w[:, pelvis_idx, :]
    ground_z = float(np.min(body_pos_w[:, :, 2]))
    root_pos[:, 2] -= ground_z
    dof_pos = joint_pos

    return MotionClip(
        path=Path(npz_path),
        root_pos=root_pos.astype(np.float64),
        root_quat=root_quat.astype(np.float64),
        dof_pos=dof_pos.astype(np.float64),
        fps=fps,
        num_frames=len(joint_pos),
        ground_offset=ground_z,
    )


def load_trimmed_npz(npz_path: str | Path) -> MotionClip:
    data = np.load(npz_path)
    qpos = data["qpos"]
    fps = float(data["fps"][0])
    return MotionClip(
        path=Path(npz_path),
        root_pos=qpos[:, :3].astype(np.float64),
        root_quat=qpos[:, 3:7].astype(np.float64),
        dof_pos=qpos[:, 7:].astype(np.float64),
        fps=fps,
        num_frames=len(qpos),
    )


def load_npz_motion(npz_path: str | Path) -> MotionClip:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    keys = set(data.keys())
    if "qpos" in keys:
        return load_trimmed_npz(npz_path)
    if {"root_pos", "root_quat", "dof_pos", "fps"}.issubset(keys):
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        root_quat = np.asarray(data["root_quat"], dtype=np.float64)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
        return MotionClip(
            path=npz_path,
            root_pos=root_pos,
            root_quat=root_quat,
            dof_pos=dof_pos,
            fps=fps,
            num_frames=len(dof_pos),
        )
    if "joint_pos" in keys:
        return load_full_length_npz(npz_path)
    raise ValueError(f"Unknown NPZ format with keys: {keys}")


def iter_npz_files(folder: str | Path, pattern: str = "*.npz") -> Generator[Path, None, None]:
    folder = Path(folder)
    if folder.is_file():
        yield folder
        return
    for npz_file in sorted(folder.rglob(pattern)):
        yield npz_file


def create_model_and_data() -> tuple[mj.MjModel, mj.MjData]:
    model = mj.MjModel.from_xml_path(Z1_MJCF)
    data = mj.MjData(model)
    return model, data


def create_free_camera() -> mj.MjvCamera:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = DEFAULT_CAMERA_DISTANCE
    camera.elevation = DEFAULT_CAMERA_ELEVATION
    camera.azimuth = DEFAULT_CAMERA_AZIMUTH
    return camera


def apply_motion_frame(model: mj.MjModel, data: mj.MjData, motion: MotionClip, frame_idx: int) -> np.ndarray:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = motion.root_pos[frame_idx]
    data.qpos[3:7] = motion.root_quat[frame_idx]
    dof_pos = motion.dof_pos[frame_idx]
    if data.qpos[7:].shape[0] == dof_pos.shape[0]:
        data.qpos[7:] = dof_pos
    else:
        if dof_pos.shape[0] != len(Z1_POLICY_JOINT_NAMES):
            raise ValueError(
                f"Cannot map {dof_pos.shape[0]} DOFs into model qpos width {data.qpos[7:].shape[0]}"
            )
        for joint_name, value in zip(Z1_POLICY_JOINT_NAMES, dof_pos):
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Model is missing joint {joint_name!r}")
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    return motion.root_pos[frame_idx]


def clip_speed(motion: MotionClip) -> float:
    if motion.num_frames < 2:
        return 0.0
    delta_xy = np.diff(motion.root_pos[:, :2], axis=0)
    return float(np.linalg.norm(delta_xy, axis=1).mean() * motion.fps)


def clip_kind(name: str) -> str:
    stem = name.lower()
    if "__" not in stem:
        return ""
    tag = stem.split("__", 1)[1].split("_", 1)[0]
    return {
        "pure": "pure-run/walk",
        "stand": "stand",
        "run2walk": "run->walk",
        "run2stop": "run->stop",
    }.get(tag, tag)
