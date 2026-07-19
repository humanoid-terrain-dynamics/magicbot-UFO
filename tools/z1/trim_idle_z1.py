#!/usr/bin/env python3
"""Trim idle lead-in/out from Z1 RobotState npz motions (keep only the active motion).

Activity metric reuses the idea from pick_z1_mimic_clips.py:per_frame_activity,
adapted to the root_pos-format npz (dof_pos + root_quat; no body_pos_w available):
joint-speed (from dof_pos diff) + root angular speed (from root_quat). Leading /
trailing frames whose smoothed activity falls below a threshold are dropped.

Only array fields whose first axis matches the frame count are sliced; metadata
scalars (fps, joint_names, motion_key, ...) are preserved.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Median-of-active (75th pct) relative normalization: typical-active frame ~1, idle <<1.

    Using the 75th percentile (not min/max) as the reference means uniformly-active
    motions normalize to ~1 everywhere (no spurious trimming), while motions with a
    real idle region map idle frames near 0."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    ref = float(np.percentile(v, 75.0))
    return v / max(ref, 1e-6)


def per_frame_activity(dof_pos: np.ndarray, root_quat: np.ndarray, fps: float) -> np.ndarray:
    """Length N-1 per-frame activity in ~[0,1]."""
    joint_speed = (
        np.linalg.norm(np.diff(dof_pos, axis=0), axis=1) * fps / np.sqrt(dof_pos.shape[1])
    )
    root_ang = np.zeros_like(joint_speed)
    q = np.asarray(root_quat, dtype=np.float64)
    if q.shape[0] == dof_pos.shape[0]:
        dots = np.clip(np.abs(np.sum(q[:-1] * q[1:], axis=1)), 0.0, 1.0)
        root_ang = 2.0 * np.arccos(dots) * fps
    return 0.7 * normalize(joint_speed) + 0.3 * normalize(root_ang)


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def active_window(activity: np.ndarray, thr: float, smooth_w: int) -> tuple[int, int]:
    """Return [start, end) frame indices (in original N-frame space) of the active region."""
    a = smooth(activity, smooth_w)
    mask = a > thr
    if not mask.any():
        return 0, len(activity) + 1  # keep everything
    edges = np.where(mask)[0]
    # activity has length N-1 (frame i is the edge between frame i and i+1)
    start = int(edges[0])
    end = int(edges[-1]) + 2
    return start, end


def trim_file(path: str, out_dir: Path, thr: float, smooth_w: int, min_keep_sec: float) -> dict:
    z = dict(np.load(path, allow_pickle=True))
    dof = np.asarray(z["dof_pos"], dtype=np.float64)
    quat = np.asarray(z["root_quat"], dtype=np.float64)
    fps = float(np.asarray(z["fps"]).reshape(-1)[0])
    n = dof.shape[0]

    act = per_frame_activity(dof, quat, fps)
    s, e = active_window(act, thr, smooth_w)
    s = max(0, s)
    e = min(n, e)

    min_keep = max(4, int(min_keep_sec * fps))
    if e - s < min_keep:
        s, e = 0, n  # too short after trim -> keep full
        kept_full = True
    else:
        kept_full = False

    for k in list(z.keys()):
        a = z[k]
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == n:
            z[k] = a[s:e]
    np.savez_compressed(out_dir / os.path.basename(path), **z)
    return {
        "name": os.path.basename(path),
        "n_before": n,
        "n_after": e - s,
        "fps": fps,
        "s": s,
        "e": e,
        "kept_full": kept_full,
        "peak_activity": float(np.max(act)) if act.size else 0.0,
        "mean_activity": float(np.mean(act)) if act.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--thr", type=float, default=0.30, help="activity threshold (fraction of 75th-pct typical; idle below this is trimmed)")
    ap.add_argument("--smooth", type=int, default=10, help="smoothing window in frames")
    ap.add_argument("--min-keep-sec", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src_dir, "*.npz")))
    print(f"src={args.src_dir}  out={out}  files={len(files)}  thr={args.thr} smooth={args.smooth}")
    for f in files:
        r = trim_file(f, out, args.thr, args.smooth, args.min_keep_sec)
        flag = "  (kept full - trim too short)" if r["kept_full"] else ""
        print(
            f"  {r['name']:<22} {r['n_before']/r['fps']:5.1f}s -> {r['n_after']/r['fps']:5.1f}s  "
            f"[{r['s']/r['fps']:.1f}-{r['e']/r['fps']:.1f}s]  peak_act={r['peak_activity']:.2f}{flag}"
        )
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
