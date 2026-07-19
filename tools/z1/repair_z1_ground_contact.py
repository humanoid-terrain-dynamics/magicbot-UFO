#!/usr/bin/env python3
"""Repair Z1 mimic RobotState NPZ root height using foot mesh grounding.

The mimic source clips are in-place and were converted without true root
translation. Some exported windows therefore show both soles floating above
the floor. This tool keeps root xy/quaternion and joint angles unchanged, runs
MuJoCo FK with the official MagicBot Z1 MJCF, and shifts only root z so the
lowest foot mesh vertex sits at a small clearance above z=0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from _npz_viewer_core import Z1_MJCF, Z1_POLICY_JOINT_NAMES


DEFAULT_FOLDERS = [
    "humanoidverse/data/z1_mimic_robot_state_npz",
    "humanoidverse/data/z1_mimic_robot_state_npz_selected",
]
FOOT_BODIES = {"left_ankle_roll_link", "right_ankle_roll_link"}


def mesh_vertices_for_geom(model: mj.MjModel, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return np.empty((0, 3), dtype=np.float64)
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)


def foot_geom_ids(model: mj.MjModel) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if body_name in FOOT_BODIES and int(model.geom_dataid[geom_id]) >= 0:
            ids.append(geom_id)
    if not ids:
        raise RuntimeError(f"No mesh foot geoms found for bodies: {sorted(FOOT_BODIES)}")
    return ids


def set_frame_qpos(model: mj.MjModel, data: mj.MjData, root_pos: np.ndarray, root_quat: np.ndarray, dof_pos: np.ndarray) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    if data.qpos[7:].shape[0] == dof_pos.shape[0]:
        data.qpos[7:] = dof_pos
    else:
        for joint_name, value in zip(Z1_POLICY_JOINT_NAMES, dof_pos):
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Model is missing joint {joint_name!r}")
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)


def min_foot_z(model: mj.MjModel, data: mj.MjData, foot_geoms: list[int]) -> float:
    z_values: list[float] = []
    for geom_id in foot_geoms:
        verts = mesh_vertices_for_geom(model, geom_id)
        if verts.size == 0:
            continue
        xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        world = verts @ xmat.T + xpos
        z_values.append(float(np.min(world[:, 2])))
    if not z_values:
        raise RuntimeError("Foot mesh geoms did not expose vertices")
    return min(z_values)


def repair_file(path: Path, model: mj.MjModel, foot_geoms: list[int], clearance: float, dry_run: bool) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as npz:
        payload = {key: npz[key] for key in npz.files}

    required = {"root_pos", "root_quat", "dof_pos", "fps"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")

    root_pos = np.asarray(payload["root_pos"], dtype=np.float32).copy()
    root_quat = np.asarray(payload["root_quat"], dtype=np.float32)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
    if root_pos.shape[0] != root_quat.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(f"{path}: root_pos/root_quat/dof_pos frame mismatch")

    data = mj.MjData(model)
    before: list[float] = []
    deltas: list[float] = []
    after: list[float] = []
    for frame in range(root_pos.shape[0]):
        set_frame_qpos(model, data, root_pos[frame], root_quat[frame], dof_pos[frame])
        foot_z = min_foot_z(model, data, foot_geoms)
        delta = clearance - foot_z
        before.append(foot_z)
        deltas.append(delta)
        root_pos[frame, 2] += np.float32(delta)
        after.append(clearance)

    if not dry_run:
        payload["root_pos"] = root_pos
        payload["ground_repair_method"] = np.asarray("official_magicbot_z1_foot_mesh_min_z_v1")
        payload["ground_repair_xml"] = np.asarray(Z1_MJCF)
        payload["ground_repair_clearance"] = np.asarray(clearance, dtype=np.float32)
        payload["ground_repair_delta_z_min"] = np.asarray(np.min(deltas), dtype=np.float32)
        payload["ground_repair_delta_z_max"] = np.asarray(np.max(deltas), dtype=np.float32)
        np.savez_compressed(path, **payload)

    return {
        "frames": float(root_pos.shape[0]),
        "before_min": float(np.min(before)),
        "before_mean": float(np.mean(before)),
        "before_max": float(np.max(before)),
        "delta_min": float(np.min(deltas)),
        "delta_mean": float(np.mean(deltas)),
        "delta_max": float(np.max(deltas)),
        "after_min": float(np.min(after)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS)
    parser.add_argument("--clearance", type=float, default=0.005, help="Target minimum foot z after repair, meters.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = mj.MjModel.from_xml_path(Z1_MJCF)
    foot_geoms = foot_geom_ids(model)
    print(f"XML: {Z1_MJCF}")
    print(f"Foot geoms: {foot_geoms}")
    print(f"Target clearance: {args.clearance:.4f} m")

    all_stats: list[dict[str, float]] = []
    for folder_raw in args.folders:
        folder = Path(folder_raw)
        files = sorted(folder.glob("*.npz"))
        print(f"\nFolder: {folder} ({len(files)} files)")
        for path in files:
            stats = repair_file(path, model, foot_geoms, args.clearance, args.dry_run)
            all_stats.append(stats)
            print(
                f"  {path.name:<42} foot_z before "
                f"[{stats['before_min']:+.3f}, {stats['before_mean']:+.3f}, {stats['before_max']:+.3f}] "
                f"delta_mean={stats['delta_mean']:+.3f}"
            )

    if all_stats:
        print("\nSummary:")
        print(f"  files: {len(all_stats)}")
        print(f"  before_min: {min(s['before_min'] for s in all_stats):+.4f}")
        print(f"  before_mean: {np.mean([s['before_mean'] for s in all_stats]):+.4f}")
        print(f"  delta_mean: {np.mean([s['delta_mean'] for s in all_stats]):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
