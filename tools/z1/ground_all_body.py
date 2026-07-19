#!/usr/bin/env python3
"""Per-frame full-body grounding for Z1 RobotState npz.

Unlike repair_z1_ground_contact.py (foot-mesh only), this shifts root z each
frame so the LOWEST mesh vertex over ALL bodies sits at `clearance`. Needed
because acrobatic motions (flips/handstands) dip the pelvis/body below the
floor even when the feet are planted. Preserves root xy/quat and all joint
angles; only root z is offset per-frame. For clean review rendering.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import mujoco as mj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _npz_viewer_core import Z1_MJCF, Z1_POLICY_JOINT_NAMES  # noqa: E402


def mesh_geom_ids(model: mj.MjModel) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_dataid[g]) >= 0]


def min_mesh_z(model: mj.MjModel, data: mj.MjData, geoms: list[int]) -> float:
    zs = []
    for g in geoms:
        mid = int(model.geom_dataid[g])
        s = int(model.mesh_vertadr[mid])
        c = int(model.mesh_vertnum[mid])
        if c <= 0:
            continue
        verts = np.asarray(model.mesh_vert[s : s + c])
        xpos = np.asarray(data.geom_xpos[g])
        xmat = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        world = verts @ xmat.T + xpos
        zs.append(float(world[:, 2].min()))
    return min(zs) if zs else 0.0


def set_frame(model, data, root_pos, root_quat, dof_pos) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    for jn, v in zip(Z1_POLICY_JOINT_NAMES, dof_pos):
        ji = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)
        if ji >= 0:
            data.qpos[int(model.jnt_qposadr[ji])] = v
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--clearance", type=float, default=0.01)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(Z1_MJCF)
    data = mj.MjData(model)
    geoms = mesh_geom_ids(model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for f in sorted(glob.glob(os.path.join(args.src_dir, "*.npz"))):
        z = dict(np.load(f, allow_pickle=True))
        rp = np.asarray(z["root_pos"], dtype=np.float64).copy()
        rq = np.asarray(z["root_quat"], dtype=np.float64)
        dof = np.asarray(z["dof_pos"], dtype=np.float64)
        before_min = np.inf
        for i in range(rp.shape[0]):
            set_frame(model, data, rp[i], rq[i], dof[i])
            mz = min_mesh_z(model, data, geoms)
            before_min = min(before_min, mz)
            rp[i, 2] += args.clearance - mz  # shift so lowest mesh vertex = clearance
        z["root_pos"] = rp.astype(np.float32)
        z["ground_all_body_method"] = np.asarray("min_mesh_z_all_bodies_per_frame_v1")
        z["ground_all_body_clearance"] = np.asarray(args.clearance, dtype=np.float32)
        z["ground_all_body_before_min_z"] = np.asarray(before_min, dtype=np.float32)
        np.savez_compressed(out / os.path.basename(f), **z)
        print(f"  {os.path.basename(f):<22} before_min_body_z={before_min:+.3f} -> grounded root_z[min]={rp[:, 2].min():+.3f}")
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
