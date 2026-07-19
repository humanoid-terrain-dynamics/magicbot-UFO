"""Interactive MuJoCo viewer for UFO latent finite-state switching.

Keys:
  0: IDLE/default-pose PD hold
  1: aini
  2: kick
  3: cekongfan
  4: mabu
  R: reset to current state's source start pose
  Q / Esc: quit
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

from deploy_onnx_mujoco import (
    CLIPS,
    History,
    _actor_obs,
    _actuator_ids,
    _artifact_dir,
    _initial_qvel,
    _joint_addresses,
    _load_yaml,
    _make_runtime_xml,
    _project_root,
    _raw_obs,
)


KEY_TO_STATE = {
    ord("1"): "aini",
    ord("2"): "kick",
    ord("3"): "cekongfan",
    ord("4"): "mabu",
}


def _project_z(z: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(z))
    if norm < 1.0e-8:
        return z.astype(np.float32)
    return (math.sqrt(z.shape[-1]) * z / norm).astype(np.float32)


def _load_clip_data(project_root: Path, artifact_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for name, info in CLIPS.items():
        npz = np.load(project_root / info["npz"], allow_pickle=True)
        out[name] = {
            "root_pos": np.asarray(npz["root_pos"], dtype=np.float32),
            "root_quat": np.asarray(npz["root_quat"], dtype=np.float32),
            "dof_pos": np.asarray(npz["dof_pos"], dtype=np.float32),
            "fps": np.asarray(npz["fps"], dtype=np.float32),
            "z": np.asarray(joblib.load(artifact_dir / info["z"]), dtype=np.float32),
        }
    return out


def _set_default_pose(data: mujoco.MjData, default_q: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qpos[:3] = [0.0, 0.0, 0.75]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = default_q
    data.qvel[:] = 0.0


def _set_clip_start(model: mujoco.MjModel, data: mujoco.MjData, clip: dict[str, np.ndarray]) -> None:
    data.qpos[:3] = clip["root_pos"][0]
    data.qpos[3:7] = clip["root_quat"][0]
    data.qpos[7:] = clip["dof_pos"][0]
    data.qvel[:] = _initial_qvel(model, data.qpos.copy(), clip, float(clip["fps"].reshape(-1)[0]))


class UfoFsm:
    def __init__(self, clips: dict[str, dict[str, np.ndarray]], *, blend_ticks: int) -> None:
        self.clips = clips
        self.current = "IDLE"
        self.requested = "IDLE"
        self.frame = 0
        self.blend_ticks = max(1, int(blend_ticks))
        self.blend_i = self.blend_ticks
        self.blend_from = np.zeros(256, dtype=np.float32)
        self.z = np.zeros(256, dtype=np.float32)

    def request(self, state: str) -> None:
        if state == self.current and state != "IDLE":
            return
        self.requested = state

    def apply_request(self) -> None:
        if self.requested == self.current:
            return
        self.blend_from = self.z.copy()
        self.blend_i = 0
        self.current = self.requested
        self.frame = 0
        print(f"[fsm] switched to {self.current}", flush=True)

    def next_z(self) -> np.ndarray | None:
        self.apply_request()
        if self.current == "IDLE":
            self.z[:] = 0.0
            return None
        seq = self.clips[self.current]["z"]
        target = seq[self.frame % len(seq)].astype(np.float32)
        self.frame += 1
        if self.blend_i < self.blend_ticks:
            alpha = float(self.blend_i + 1) / float(self.blend_ticks)
            self.z = _project_z((1.0 - alpha) * self.blend_from + alpha * target)
            self.blend_i += 1
        else:
            self.z = target
        return self.z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive UFO Z1 FSM MuJoCo viewer.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--blend-ticks", type=int, default=15)
    parser.add_argument("--start", choices=["IDLE", *sorted(CLIPS.keys())], default="IDLE")
    parser.add_argument(
        "--reset-to-clip-on-switch",
        action="store_true",
        help="Teleport to the target clip start pose when switching. Useful for debugging, not real-robot faithful.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = _project_root()
    args = parse_args()
    artifact_dir = (args.artifact_dir or _artifact_dir(project_root)).resolve()
    output_dir = artifact_dir / "mujoco_fsm_viewer"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(project_root / "configs" / "robots" / "z1_23dof.yaml")
    meta = json.loads((artifact_dir / "FBcprAuxModel_41613312.meta.json").read_text(encoding="utf-8"))
    joint_names = list(meta["control_joint_names"])
    control = cfg["training"]["control"]
    default_q = np.asarray([cfg["default_dof_pos"][j] for j in joint_names], dtype=np.float32)
    kp = np.asarray([control["stiffness"][j] for j in joint_names], dtype=np.float32)
    kd = np.asarray([control["damping"][j] for j in joint_names], dtype=np.float32)
    effort = np.asarray(control["effort_limit"], dtype=np.float32)
    action_scale = float(control["action_scale"]) * effort / kp

    runtime_xml = _make_runtime_xml(project_root / cfg["xml_path"], output_dir)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = _joint_addresses(model, joint_names)
    actuator_ids = _actuator_ids(model, joint_names)
    clips = _load_clip_data(project_root, artifact_dir)

    policy = ort.InferenceSession(str(artifact_dir / "FBcprAuxModel_41613312.onnx"), providers=["CPUExecutionProvider"])
    history = History()
    fsm = UfoFsm(clips, blend_ticks=args.blend_ticks)
    fsm.request(args.start)
    if args.start == "IDLE":
        _set_default_pose(data, default_q)
    else:
        _set_clip_start(model, data, clips[args.start])
    mujoco.mj_forward(model, data)

    action_scaled = np.zeros(23, dtype=np.float32)
    target_q = default_q.copy()
    decimation = 10
    sim_step = 0
    quit_requested = False
    reset_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested, reset_requested
        if keycode in KEY_TO_STATE:
            fsm.request(KEY_TO_STATE[keycode])
            return
        if keycode == ord("0"):
            fsm.request("IDLE")
            return
        if keycode in (ord("R"), ord("r")):
            reset_requested = True
            return
        if keycode in (ord("Q"), ord("q"), 256):
            quit_requested = True

    print("[fsm] controls: 0=IDLE 1=aini 2=kick 3=cekongfan 4=mabu R=reset Q/Esc=quit", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not quit_requested:
            step_start = time.time()
            if reset_requested:
                history = History()
                action_scaled[:] = 0.0
                if fsm.current == "IDLE":
                    _set_default_pose(data, default_q)
                else:
                    _set_clip_start(model, data, clips[fsm.current])
                mujoco.mj_forward(model, data)
                reset_requested = False
                print(f"[fsm] reset {fsm.current}", flush=True)

            if sim_step % decimation == 0:
                if args.reset_to_clip_on_switch and fsm.requested != fsm.current and fsm.requested != "IDLE":
                    _set_clip_start(model, data, clips[fsm.requested])
                    mujoco.mj_forward(model, data)
                    history = History()
                    action_scaled[:] = 0.0
                raw = _raw_obs(data, qpos_addr, qvel_addr, default_q, action_scaled)
                z = fsm.next_z()
                if z is None:
                    action_scaled[:] = 0.0
                    target_q = default_q.copy()
                else:
                    obs = _actor_obs(raw, history, z)
                    action = policy.run(["action"], {"actor_obs": obs})[0][0].astype(np.float32)
                    action_scaled = np.clip(action * float(control["normalize_action_to"]), -5.0, 5.0)
                    target_q = default_q + action_scaled * action_scale
                history.add(raw)

            q = data.qpos[qpos_addr].astype(np.float32)
            qd = data.qvel[qvel_addr].astype(np.float32)
            torque = kp * (target_q - q) - kd * qd
            data.ctrl[actuator_ids] = np.clip(torque, -effort, effort)
            mujoco.mj_step(model, data)
            viewer.sync()
            sim_step += 1

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
