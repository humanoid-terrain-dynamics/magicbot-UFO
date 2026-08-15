#!/usr/bin/env python3
"""Ordinary Z1 UFO ONNX sim2sim, analogous to the upstream G1 two-terminal flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float32)


class History:
    def __init__(self, dof: int) -> None:
        self.values = {key: np.zeros((4, size), dtype=np.float32) for key, size in {"actions": dof, "base_ang_vel": 3, "dof_pos": dof, "dof_vel": dof, "projected_gravity": 3}.items()}

    def add(self, raw: dict[str, np.ndarray]) -> None:
        for key, value in raw.items():
            self.values[key][1:] = self.values[key][:-1]
            self.values[key][0] = value

    def vector(self) -> np.ndarray:
        return np.concatenate([self.values[key].reshape(-1) for key in sorted(self.values)]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a staged Z1 UFO policy in MuJoCo.")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--policy-name",
        default="z1_policy",
        help="Directory name under bundle/model (for example z1_policy or z1_policy_recovery5).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Output JSON path; defaults to outputs/sim2sim_<policy-name>_metrics.json.",
    )
    parser.add_argument("--latent-dir", type=Path, help="Override the tracking latent directory.")
    parser.add_argument("--task-manifest", type=Path, help="Override the clean-task name manifest.")
    parser.add_argument("--latent", default="zs_0.pkl", help="Tracking latent filename.")
    parser.add_argument(
        "--task-name",
        help="Task name rendered in the interactive MuJoCo window; defaults to the selected latent name.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs the entire latent sequence.")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def key_name(key: object) -> str:
    """Normalize MuJoCo/GLFW callback values to a lowercase character."""
    if isinstance(key, str):
        return key.lower()
    try:
        return chr(int(key)).lower()
    except (TypeError, ValueError):
        return ""


def render_task_name(viewer: object, task_name: str) -> None:
    """Render the active task name as a world-space label above the robot."""
    scene = viewer.user_scn
    scene.ngeom = 1
    geom = scene.geoms[0]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3),
        np.array([0.0, 0.0, 1.5]),
        np.eye(3).flatten(),
        np.array([1.0, 1.0, 1.0, 1.0]),
    )
    geom.label = f"Task: {task_name}"


def load_task_names(manifest_path: Path, expected_count: int) -> list[str]:
    """Load the clean-20 task order from the copied remote MD5 manifest."""
    task_names = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1].endswith(".npz"):
            task_names.append(Path(parts[1]).stem)
    if len(task_names) != expected_count:
        raise ValueError(
            f"Expected {expected_count} task names in {manifest_path}, found {len(task_names)}."
        )
    return task_names


def main() -> None:
    args = parse_args()
    root = args.bundle_root.resolve()
    model_root = root / "model" / args.policy_name
    if not model_root.is_dir():
        raise FileNotFoundError(f"Policy bundle not found: {model_root}")
    manifest = json.loads((model_root / "release_manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        got = sha256(root / rel)
        if got != expected:
            raise RuntimeError(f"Artifact hash mismatch: {rel}")
    contract = json.loads((model_root / "z1_control_contract.json").read_text(encoding="utf-8"))
    meta = json.loads((model_root / "exported/policy.meta.json").read_text(encoding="utf-8"))
    if meta["actor_obs_dim"] != 631 or meta["output_action_dim"] != 23 or len(contract["control_joint_names"]) != 23:
        raise RuntimeError("Z1 policy/control interface must be actor_obs[*,631] -> action[*,23].")
    latent_root = (args.latent_dir.resolve() if args.latent_dir else model_root / "tracking_inference")
    latent_paths = sorted(latent_root.glob("zs_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    if not latent_paths:
        raise FileNotFoundError(f"No tracking latents found under {latent_root}")
    requested = latent_root / args.latent
    if requested not in latent_paths:
        raise ValueError(f"Unknown latent {args.latent}; choose one of {[p.name for p in latent_paths]}")
    latent_index = latent_paths.index(requested)
    task_manifest = args.task_manifest.resolve() if args.task_manifest else root / "docs/z1_clean20_md5_remote.txt"
    task_names = load_task_names(task_manifest, len(latent_paths))
    task_name = args.task_name or task_names[latent_index]
    latents = [np.asarray(joblib.load(path), dtype=np.float32) for path in latent_paths]
    for path, sequence in zip(latent_paths, latents):
        if sequence.ndim != 2 or sequence.shape[1] != 256 or not np.isfinite(sequence).all():
            raise ValueError(f"Invalid latent sequence {path.name}: {sequence.shape}")

    xml = root / "robot/mjcf/MAGICBOTZ1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    model.opt.timestep = float(contract["sim_dt"])
    names = contract["control_joint_names"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
    if min(joint_ids + actuator_ids) < 0 or model.nu != 23:
        raise RuntimeError("Staged MJCF does not expose the 23 expected named joints and actuators.")
    qpos_addr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    qvel_addr = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    default_q = np.asarray(contract["default_dof_pos"], dtype=np.float32)
    kp, kd, effort = (np.asarray(contract[key], dtype=np.float32) for key in ("stiffness", "damping", "effort_limit"))
    data.qpos[2] = 0.75
    data.qpos[3] = 1.0
    data.qpos[qpos_addr] = default_q
    mujoco.mj_forward(model, data)
    policy = ort.InferenceSession(str(model_root / "exported/policy.onnx"), providers=["CPUExecutionProvider"])
    if policy.get_inputs()[0].name != "actor_obs" or policy.get_inputs()[0].shape[-1] != 631:
        raise RuntimeError("Unexpected policy ONNX input contract.")
    history, action_scaled = History(23), np.zeros(23, dtype=np.float32)
    target_q = default_q.copy()
    mode = {"value": "tracking"}
    frame_index = {"value": 0}
    stand_start = {"value": default_q.copy()}
    stand_frame = {"value": 0}
    stand_frames = 50
    realtime = {"value": True}

    def print_status() -> None:
        sequence = latents[latent_index]
        print(
            f"[sim2sim] mode={mode['value']} latent={latent_paths[latent_index].stem} "
            f"task_name={task_name} "
            f"latent={latent_paths[latent_index].name} frame={frame_index['value']}/{len(sequence)} "
            f"z_dim={sequence.shape[1]} z_mean={sequence.mean():.5f} z_std={sequence.std():.5f} "
            f"realtime={'on' if realtime['value'] else 'off'}",
            flush=True,
        )

    def print_controls() -> None:
        print(
            "[sim2sim] controls:\n"
            "  i  interpolate to default standing pose\n"
            "  ]  enable policy action using the current latent\n"
            "  [  start tracking motion from frame zero\n"
            "  p  reset tracking motion to frame zero\n"
            "  o  stop policy action and hold current joints\n"
            "  n  switch to the next tracking/goal latent\n"
            "  r  toggle real-time pacing (off runs simulation as fast as possible)",
            flush=True,
        )

    def handle_key(key: object) -> None:
        nonlocal latent_index, task_name
        name = key_name(key)
        if name == "i":
            mode["value"], stand_frame["value"] = "stand", 0
            stand_start["value"] = data.qpos[qpos_addr].copy()
        elif name == "]":
            mode["value"] = "policy"
        elif name == "[":
            mode["value"], frame_index["value"] = "tracking", 0
        elif name == "p":
            mode["value"], frame_index["value"] = "tracking", 0
        elif name == "o":
            mode["value"] = "hold"
            target_q[:] = data.qpos[qpos_addr]
        elif name == "n":
            latent_index = (latent_index + 1) % len(latents)
            if args.task_name is None:
                task_name = task_names[latent_index]
            frame_index["value"] = 0
            mode["value"] = "tracking"
            print_status()
        elif name == "r":
            realtime["value"] = not realtime["value"]
            print_status()

    decimation = round((1.0 / float(contract["control_hz"])) / model.opt.timestep)
    steps = len(latents[latent_index]) if args.max_steps == 0 else min(len(latents[latent_index]), args.max_steps)
    min_root_z, max_tilt, max_torque = float("inf"), 0.0, 0.0

    def control_step(frame: int) -> None:
        nonlocal action_scaled, target_q, min_root_z, max_tilt, max_torque, latent_index
        quat = data.qpos[3:7].astype(np.float32)
        gravity = quat_to_rotmat_wxyz(quat).T @ np.array([0, 0, -1], dtype=np.float32)
        raw = {"actions": action_scaled, "base_ang_vel": data.qvel[3:6].astype(np.float32) * 0.25, "dof_pos": data.qpos[qpos_addr].astype(np.float32) - default_q, "dof_vel": data.qvel[qvel_addr].astype(np.float32), "projected_gravity": gravity}
        state = np.concatenate([raw["dof_pos"], raw["dof_vel"], raw["projected_gravity"], raw["base_ang_vel"]])
        if mode["value"] in ("policy", "tracking"):
            sequence = latents[latent_index]
            active_frame = frame_index["value"] % len(sequence)
            obs = np.concatenate([state, raw["actions"], history.vector(), sequence[active_frame]])[None].astype(np.float32)
            action = policy.run(["action"], {"actor_obs": obs})[0][0]
            action_scaled = np.clip(action * float(contract["normalize_action_to"]), -5, 5).astype(np.float32)
            target_q = default_q + action_scaled * (float(contract["action_scale"]) * effort / kp)
            frame_index["value"] += 1
        elif mode["value"] == "stand":
            alpha = min(1.0, stand_frame["value"] / stand_frames)
            target_q = (1.0 - alpha) * stand_start["value"] + alpha * default_q
            action_scaled.fill(0)
            stand_frame["value"] += 1
        elif mode["value"] == "hold":
            action_scaled.fill(0)
        history.add(raw)
        min_root_z = min(min_root_z, float(data.qpos[2]))
        max_tilt = max(max_tilt, float(np.arccos(np.clip(-gravity[2], -1, 1))))

    viewer = None if args.headless else mujoco.viewer.launch_passive(model, data, key_callback=handle_key)
    if viewer is not None:
        print_controls()
        print_status()
    try:
        for frame in range(steps):
            control_step(frame)
            for _ in range(decimation):
                torque = np.clip(kp * (target_q - data.qpos[qpos_addr]) - kd * data.qvel[qvel_addr], -effort, effort)
                data.ctrl[actuator_ids] = torque
                max_torque = max(max_torque, float(np.max(np.abs(torque))))
                mujoco.mj_step(model, data)
                if viewer is not None:
                    render_task_name(viewer, task_name)
                    viewer.sync()
                    if realtime["value"]:
                        time.sleep(model.opt.timestep)
    finally:
        if viewer is not None:
            viewer.close()
    result = {"latent": latent_paths[latent_index].name, "steps": steps, "min_root_z": min_root_z, "max_base_tilt_rad": max_tilt, "max_abs_torque": max_torque, "final_root_z": float(data.qpos[2])}
    metrics_path = args.metrics_path.resolve() if args.metrics_path else root / "outputs" / f"sim2sim_{args.policy_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
