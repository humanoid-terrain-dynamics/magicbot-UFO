"""Headless MuJoCo recording for random dragging across 44 Z1 UFO priors.

The script simulates random slider changes between the 44 curated priors for a
longer simulated duration and writes an accelerated MP4. Example:

  python tools/z1/record_ufo_44_random_drag.py --duration-s 120 --speedup 4

That produces a 30-second video from 120 seconds of simulated motion.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from deploy_onnx_mujoco import (
    History,
    _actor_obs,
    _actuator_ids,
    _artifact_dir,
    _joint_addresses,
    _load_yaml,
    _make_runtime_xml,
    _project_root,
    _raw_obs,
)
from ufo_44_slider_tk_mujoco import SliderFsm, _load_entries, _set_clip_pose


def _open_ffmpeg(path: Path, *, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "veryfast",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    artifact_dir = _artifact_dir(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=artifact_dir)
    parser.add_argument("--cache-root", type=Path, default=project_root / "cache" / "motion_data" / "z1_mimic_clean")
    parser.add_argument("--latent-dir", type=Path, default=artifact_dir / "z44_latents")
    parser.add_argument("--output", type=Path, default=artifact_dir / "mujoco_44_slider" / "ufo_44_random_drag_120s_x4.mp4")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--speedup", type=float, default=4.0)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--blend-ticks", type=int, default=15)
    parser.add_argument("--min-hold-s", type=float, default=1.0)
    parser.add_argument("--max-hold-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--start", default="stagestand")
    return parser.parse_args()


def _start_index(entries, requested: str) -> int:
    if requested.isdigit():
        return int(np.clip(int(requested), 0, len(entries) - 1))
    for entry in entries:
        if requested in entry.key:
            return entry.index
    return 0


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    project_root = _project_root()
    artifact_dir = args.artifact_dir.resolve()
    entries = _load_entries(project_root, artifact_dir, args.cache_root.resolve(), args.latent_dir.resolve())

    cfg = _load_yaml(project_root / "configs" / "robots" / "z1_23dof.yaml")
    meta = json.loads((artifact_dir / "FBcprAuxModel_41613312.meta.json").read_text(encoding="utf-8"))
    joint_names = list(meta["control_joint_names"])
    control = cfg["training"]["control"]
    default_q = np.asarray([cfg["default_dof_pos"][j] for j in joint_names], dtype=np.float32)
    kp = np.asarray([control["stiffness"][j] for j in joint_names], dtype=np.float32)
    kd = np.asarray([control["damping"][j] for j in joint_names], dtype=np.float32)
    effort = np.asarray(control["effort_limit"], dtype=np.float32)
    action_scale = float(control["action_scale"]) * effort / kp

    output_dir = artifact_dir / "mujoco_44_slider"
    runtime_xml = _make_runtime_xml(project_root / cfg["xml_path"], output_dir)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = _joint_addresses(model, joint_names)
    actuator_ids = _actuator_ids(model, joint_names)

    start_index = _start_index(entries, args.start)
    fsm = SliderFsm(entries, start_index, args.blend_ticks)
    _set_clip_pose(model, data, fsm.current.clip)
    mujoco.mj_forward(model, data)

    policy = ort.InferenceSession(str(artifact_dir / "FBcprAuxModel_41613312.onnx"), providers=[args.provider])
    history = History()
    action_scaled = np.zeros(23, dtype=np.float32)
    target_q = default_q.copy()

    decimation = max(1, int(round((1.0 / float(args.policy_hz)) / model.opt.timestep)))
    render_interval = max(1, int(round((1.0 / float(args.render_fps)) / model.opt.timestep)))
    total_steps = int(round(float(args.duration_s) / model.opt.timestep))
    video_fps = float(args.render_fps) * float(args.speedup)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 3.0
    cam.azimuth = 135.0
    cam.elevation = -18.0

    ffmpeg = _open_ffmpeg(args.output.resolve(), width=args.width, height=args.height, fps=video_fps)
    next_switch_step = 0
    switch_log = []
    frames = 0
    try:
        for step in range(total_steps):
            if step >= next_switch_step:
                old = fsm.current_index
                new = random.randrange(len(entries))
                if new == old:
                    new = (new + random.randrange(1, len(entries))) % len(entries)
                fsm.request(new)
                hold_s = random.uniform(float(args.min_hold_s), float(args.max_hold_s))
                next_switch_step = step + max(1, int(round(hold_s / model.opt.timestep)))
                switch_log.append(
                    {
                        "sim_time_s": step * model.opt.timestep,
                        "from": old,
                        "to": new,
                        "to_key": entries[new].key,
                    }
                )

            if step % decimation == 0:
                raw = _raw_obs(data, qpos_addr, qvel_addr, default_q, action_scaled)
                z = fsm.next_z()
                if fsm.changed:
                    history = History()
                obs = _actor_obs(raw, history, z)
                action = policy.run(["action"], {"actor_obs": obs})[0][0].astype(np.float32)
                action_scaled = np.clip(action * float(control["normalize_action_to"]), -5.0, 5.0)
                target_q = default_q + action_scaled * action_scale
                history.add(raw)

            q = data.qpos[qpos_addr].astype(np.float32)
            qd = data.qvel[qvel_addr].astype(np.float32)
            torque = np.clip(kp * (target_q - q) - kd * qd, -effort, effort)
            data.ctrl[actuator_ids] = torque
            mujoco.mj_step(model, data)

            if step % render_interval == 0:
                root = data.qpos[:3]
                cam.lookat[:] = [float(root[0]), float(root[1]), float(root[2] + 0.15)]
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
                frames += 1

            if step % max(1, int(round(10.0 / model.opt.timestep))) == 0:
                print(
                    f"[record] sim={step * model.opt.timestep:6.1f}/{args.duration_s:.1f}s "
                    f"motion={fsm.current_index:02d} frames={frames}",
                    flush=True,
                )
    finally:
        renderer.close()
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        ret = ffmpeg.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")

    log_path = args.output.with_suffix(".json")
    log_path.write_text(
        json.dumps(
            {
                "video": str(args.output.resolve()),
                "duration_s": args.duration_s,
                "speedup": args.speedup,
                "video_duration_s": args.duration_s / args.speedup,
                "render_fps": args.render_fps,
                "video_fps": video_fps,
                "frames": frames,
                "switches": switch_log,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[record] wrote {args.output.resolve()}", flush=True)
    print(f"[record] wrote {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
