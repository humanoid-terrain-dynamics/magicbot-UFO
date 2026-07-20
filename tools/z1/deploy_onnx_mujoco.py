"""Run exported Z1 UFO ONNX policy in plain MuJoCo.

This is a small sim-to-sim harness for the ONNX actor exported by
humanoidverse.tracking_inference. It intentionally mirrors the Z1 training
observation/action contract instead of using the training environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import joblib
import mujoco
import numpy as np
import onnxruntime as ort
import yaml

CLIPS = {
    "aini": {
        "npz": "humanoidverse/data/z1_mimic_robot_state_npz_selected/aini__pick000_028000_032000.npz",
        "z": "aini_zs.pkl",
    },
    "kick": {
        "npz": "humanoidverse/data/z1_mimic_robot_state_npz_selected/kick__pick000_000620_003120.npz",
        "z": "kick_zs.pkl",
    },
    "cekongfan": {
        "npz": "humanoidverse/data/z1_mimic_robot_state_npz_selected/cekongfan__pick001_002480_004980.npz",
        "z": "cekongfan_zs.pkl",
    },
    "mabu": {
        "npz": "humanoidverse/data/z1_mimic_robot_state_npz_selected/mabu__pick000_002000_006000.npz",
        "z": "mabu_zs.pkl",
    },
}

TERRAIN_CHOICES = ("plane", "gravel")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_dir(project_root: Path) -> Path:
    return project_root.parent / "checkpoints" / "Z1_UFO" / "compare_hand_vs_nohand" / "fakehand_32M"


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_wxyz_to_mat(q).T @ v.astype(np.float32)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _replace_meshdir(text: str, source_xml: Path) -> str:
    meshdir = (source_xml.parent / "../meshes").resolve().as_posix()
    return re.sub(r'meshdir=(["\'])(.*?)\1', f'meshdir="{meshdir}"', text, count=1)


def _strip_floor_planes(text: str) -> str:
    floor_plane = re.compile(
        r'\n?[ \t]*<geom\b(?=[^>]*\bname\s*=\s*["\']floor["\'])(?=[^>]*\btype\s*=\s*["\']plane["\'])[^>]*/>[ \t]*',
        flags=re.IGNORECASE,
    )
    return floor_plane.sub("", text)


def _gravel_hfield_asset(nrow: int = 33, ncol: int = 33) -> str:
    rng = np.random.default_rng(20260720)
    height = rng.uniform(0.0, 1.0, size=(nrow, ncol)).astype(np.float32)
    for _ in range(2):
        height = (
            height
            + np.roll(height, 1, axis=0)
            + np.roll(height, -1, axis=0)
            + np.roll(height, 1, axis=1)
            + np.roll(height, -1, axis=1)
        ) / 5.0
    height -= float(height.min())
    max_height = float(height.max())
    if max_height > 0.0:
        height /= max_height
    elevation = " ".join(f"{v:.6f}" for v in height.reshape(-1))
    return f'<hfield name="sim2sim_gravel_hfield" nrow="{nrow}" ncol="{ncol}" size="20 20 0.04 0.02" elevation="{elevation}"/>'


def _apply_runtime_terrain(text: str, terrain: str) -> str:
    if terrain == "plane":
        return text
    if terrain != "gravel":
        raise ValueError(f"Unsupported terrain {terrain!r}; expected one of {TERRAIN_CHOICES}")

    text = _strip_floor_planes(text)
    if 'name="sim2sim_gravel_hfield"' not in text and "name='sim2sim_gravel_hfield'" not in text:
        hfield = _gravel_hfield_asset()
        if "</asset>" in text:
            text = text.replace("</asset>", f"    {hfield}\n  </asset>", 1)
        else:
            text = re.sub(r"(<mujoco\b[^>]*>)", rf"\1\n  <asset>\n    {hfield}\n  </asset>", text, count=1)
    gravel_geom = (
        '<geom name="sim2sim_gravel" type="hfield" hfield="sim2sim_gravel_hfield" '
        'rgba="0.34 0.33 0.30 1" friction="1.2 0.02 0.001"/>'
    )
    text = text.replace("<worldbody>", "<worldbody>\n    " + gravel_geom, 1)
    return text


def _make_runtime_xml(source_xml: Path, output_dir: Path, *, terrain: str = "plane") -> Path:
    text = source_xml.read_text(encoding="utf-8")
    text = _replace_meshdir(text, source_xml)
    text = _apply_runtime_terrain(text, terrain)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"MAGICBOTZ1_sim2sim_{terrain}.xml"
    out.write_text(text, encoding="utf-8")
    return out


def _joint_addresses(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos = []
    qvel = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found in MuJoCo model: {name}")
        qpos.append(int(model.jnt_qposadr[jid]))
        qvel.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos, dtype=np.int64), np.asarray(qvel, dtype=np.int64)


def _actuator_ids(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    ids = []
    for name in joint_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Actuator not found in MuJoCo model: {name}")
        ids.append(aid)
    return np.asarray(ids, dtype=np.int64)


class History:
    def __init__(self) -> None:
        self.data = {
            "actions": np.zeros((4, 23), dtype=np.float32),
            "base_ang_vel": np.zeros((4, 3), dtype=np.float32),
            "dof_pos": np.zeros((4, 23), dtype=np.float32),
            "dof_vel": np.zeros((4, 23), dtype=np.float32),
            "projected_gravity": np.zeros((4, 3), dtype=np.float32),
        }

    def vector(self) -> np.ndarray:
        parts = []
        for key in sorted(self.data.keys()):
            parts.append(self.data[key].reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def add(self, raw: dict[str, np.ndarray]) -> None:
        for key, value in raw.items():
            old = self.data[key].copy()
            self.data[key][1:] = old[:-1]
            self.data[key][0] = value.astype(np.float32)


def _raw_obs(
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    default_q: np.ndarray,
    action_scaled: np.ndarray,
) -> dict[str, np.ndarray]:
    q = data.qpos[qpos_addr].copy().astype(np.float32)
    qd = data.qvel[qvel_addr].copy().astype(np.float32)
    quat_wxyz = data.qpos[3:7].copy().astype(np.float32)
    base_ang_vel = data.qvel[3:6].copy().astype(np.float32)
    projected_gravity = _rotate_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    return {
        "actions": action_scaled.astype(np.float32),
        "base_ang_vel": (base_ang_vel * 0.25).astype(np.float32),
        "dof_pos": (q - default_q).astype(np.float32),
        "dof_vel": qd,
        "projected_gravity": projected_gravity.astype(np.float32),
    }


def _actor_obs(raw: dict[str, np.ndarray], history: History, z: np.ndarray) -> np.ndarray:
    state = np.concatenate(
        [raw["dof_pos"], raw["dof_vel"], raw["projected_gravity"], raw["base_ang_vel"]],
        axis=0,
    )
    obs = np.concatenate([state, raw["actions"], history.vector(), z.astype(np.float32)], axis=0)
    if obs.shape != (631,):
        raise ValueError(f"Expected actor_obs shape (631,), got {obs.shape}")
    return obs[None, :].astype(np.float32)


def _initial_qvel(model: mujoco.MjModel, qpos0: np.ndarray, clip: dict[str, np.ndarray], fps: float) -> np.ndarray:
    qpos1 = qpos0.copy()
    if len(clip["root_pos"]) > 1:
        qpos1[:3] = clip["root_pos"][1]
        qpos1[3:7] = clip["root_quat"][1]
        qpos1[7:] = clip["dof_pos"][1]
    qvel = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel, 1.0 / fps, qpos0, qpos1)
    return qvel


def _render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    root = data.qpos[:3]
    cam.lookat[:] = [float(root[0]), float(root[1]), float(root[2] + 0.2)]
    cam.distance = 3.0
    cam.azimuth = 135.0
    cam.elevation = -18.0
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames to write for {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open OpenCV VideoWriter for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def run_clip(
    name: str,
    *,
    project_root: Path,
    artifact_dir: Path,
    output_dir: Path,
    policy: ort.InferenceSession,
    render: bool,
    terrain: str,
) -> dict:
    cfg = _load_yaml(project_root / "configs" / "robots" / "z1_23dof.yaml")
    meta = json.loads((artifact_dir / "FBcprAuxModel_41613312.meta.json").read_text(encoding="utf-8"))
    joint_names = list(meta["control_joint_names"])
    control = cfg["training"]["control"]
    default_map = cfg["default_dof_pos"]
    default_q = np.asarray([default_map[j] for j in joint_names], dtype=np.float32)
    kp = np.asarray([control["stiffness"][j] for j in joint_names], dtype=np.float32)
    kd = np.asarray([control["damping"][j] for j in joint_names], dtype=np.float32)
    effort = np.asarray(control["effort_limit"], dtype=np.float32)
    action_scale = float(control["action_scale"]) * effort / kp
    clip_info = CLIPS[name]
    clip_npz = np.load(project_root / clip_info["npz"], allow_pickle=True)
    clip = {k: clip_npz[k] for k in clip_npz.files}
    z_seq = np.asarray(joblib.load(artifact_dir / clip_info["z"]), dtype=np.float32)
    steps = min(len(z_seq), len(clip["dof_pos"]))
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    sim_dt = 0.002
    control_dt = 1.0 / fps
    decimation = max(1, int(round(control_dt / sim_dt)))

    runtime_xml = _make_runtime_xml(project_root / cfg["xml_path"], output_dir, terrain=terrain)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = _joint_addresses(model, joint_names)
    actuator_ids = _actuator_ids(model, joint_names)

    qpos0 = np.zeros(model.nq, dtype=np.float64)
    qpos0[:3] = clip["root_pos"][0]
    qpos0[3:7] = clip["root_quat"][0]
    qpos0[7:] = clip["dof_pos"][0]
    data.qpos[:] = qpos0
    data.qvel[:] = _initial_qvel(model, qpos0, clip, fps)
    mujoco.mj_forward(model, data)

    history = History()
    action_scaled = np.zeros(23, dtype=np.float32)
    frames = []
    root_z = []
    action_abs_max = []
    torque_abs_max = []
    dof_mae = []
    renderer = mujoco.Renderer(model, height=480, width=640) if render else None
    try:
        for step in range(steps):
            raw = _raw_obs(data, qpos_addr, qvel_addr, default_q, action_scaled)
            obs = _actor_obs(raw, history, z_seq[step])
            history.add(raw)
            action = policy.run(["action"], {"actor_obs": obs})[0][0].astype(np.float32)
            action_scaled = np.clip(action * float(control["normalize_action_to"]), -5.0, 5.0)
            target_q = default_q + action_scaled * action_scale

            for _ in range(decimation):
                q = data.qpos[qpos_addr].astype(np.float32)
                qd = data.qvel[qvel_addr].astype(np.float32)
                torque = kp * (target_q - q) - kd * qd
                torque = np.clip(torque, -effort, effort)
                data.ctrl[actuator_ids] = torque
                mujoco.mj_step(model, data)

            if renderer is not None:
                frames.append(_render_frame(renderer, model, data))
            root_z.append(float(data.qpos[2]))
            action_abs_max.append(float(np.max(np.abs(action_scaled))))
            torque_abs_max.append(float(np.max(np.abs(data.ctrl[actuator_ids]))))
            dof_mae.append(float(np.mean(np.abs(data.qpos[qpos_addr] - clip["dof_pos"][min(step, len(clip["dof_pos"]) - 1)]))))
    finally:
        if renderer is not None:
            renderer.close()

    video_path = output_dir / f"{name}_mujoco_onnx.mp4"
    if render:
        _write_mp4(video_path, frames, fps=fps)

    return {
        "clip": name,
        "steps": int(steps),
        "fps": fps,
        "decimation": int(decimation),
        "terrain": terrain,
        "video": str(video_path) if render else None,
        "min_root_z": float(np.min(root_z)) if root_z else None,
        "final_root_z": float(root_z[-1]) if root_z else None,
        "max_abs_scaled_action": float(np.max(action_abs_max)) if action_abs_max else None,
        "max_abs_torque": float(np.max(torque_abs_max)) if torque_abs_max else None,
        "mean_dof_mae_vs_reference": float(np.mean(dof_mae)) if dof_mae else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z1 UFO ONNX MuJoCo sim-to-sim runner.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clips", nargs="+", default=list(CLIPS.keys()), choices=sorted(CLIPS.keys()))
    parser.add_argument(
        "--terrain",
        choices=TERRAIN_CHOICES,
        default="plane",
        help="plane uses only the robot MJCF floor; gravel removes that floor and adds one hfield.",
    )
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    project_root = _project_root()
    args = parse_args()
    artifact_dir = (args.artifact_dir or _artifact_dir(project_root)).resolve()
    output_dir = (args.output_dir or artifact_dir / "mujoco_sim2sim").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = artifact_dir / "FBcprAuxModel_41613312.onnx"
    policy = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    results = []
    for clip in args.clips:
        print(f"[sim2sim] running {clip}", flush=True)
        result = run_clip(
            clip,
            project_root=project_root,
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            policy=policy,
            render=not args.no_render,
            terrain=args.terrain,
        )
        print(f"[sim2sim] done {clip}: {result}", flush=True)
        results.append(result)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[sim2sim] wrote {metrics_path}")


if __name__ == "__main__":
    main()
