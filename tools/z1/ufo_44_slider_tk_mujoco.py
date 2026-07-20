"""Tkinter MuJoCo viewer for the 44 curated Z1 UFO motion priors.

Use ``generate_z1_44_latents.py`` first. This viewer loads the 44 cached
training motions and their generated z latents, then uses a slider to switch
between motion priors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import joblib
import mujoco
import numpy as np
import onnxruntime as ort
from deploy_onnx_mujoco import (
    TERRAIN_CHOICES,
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
from PIL import Image, ImageDraw, ImageFont, ImageTk

GROUP_FILES = [
    "z1_cyclic_locomotion_train_near10s_ufo.pkl",
    "z1_atomic_skills_train_near10s_ufo.pkl",
    "z1_acrobatics_recovery_train_near10s_ufo.pkl",
    "z1_pose_low_motion_train_near10s_ufo.pkl",
]
PANEL_WIDTH = 720
RENDER_WIDTH = PANEL_WIDTH * 2
RENDER_HEIGHT = 700


@dataclass
class MotionEntry:
    index: int
    key: str
    group: str
    local_index: int
    clip: dict[str, np.ndarray]
    z: np.ndarray

    @property
    def short_name(self) -> str:
        return self.key.replace("__clip000", "")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _project_z(z: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(z))
    if norm < 1.0e-8:
        return z.astype(np.float32)
    return (math.sqrt(z.shape[-1]) * z / norm).astype(np.float32)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + suffix


def _motion_clip(raw: dict) -> dict[str, np.ndarray]:
    return {
        "root_pos": np.asarray(raw["root_trans_offset"], dtype=np.float32),
        "root_quat": np.asarray(raw["root_quat"], dtype=np.float32),
        "dof_pos": np.asarray(raw["dof_pos"], dtype=np.float32),
        "fps": np.asarray([float(raw["fps"])], dtype=np.float32),
    }


def _load_entries(project_root: Path, artifact_dir: Path, cache_root: Path, latent_dir: Path) -> list[MotionEntry]:
    entries: list[MotionEntry] = []
    index = 0
    for group_name in GROUP_FILES:
        group_path = cache_root / group_name
        motions = joblib.load(group_path)
        for local_index, (key, raw) in enumerate(motions.items()):
            z_path = latent_dir / f"{index:02d}_{_safe_name(key)}_zs.pkl"
            if not z_path.exists():
                raise FileNotFoundError(
                    f"Missing latent {z_path}. Run: "
                    f"{project_root / '.venv' / 'Scripts' / 'python.exe'} "
                    f"{project_root / 'tools' / 'z1' / 'generate_z1_44_latents.py'}"
                )
            entries.append(
                MotionEntry(
                    index=index,
                    key=key,
                    group=group_name,
                    local_index=local_index,
                    clip=_motion_clip(raw),
                    z=np.asarray(joblib.load(z_path), dtype=np.float32),
                )
            )
            index += 1
    if len(entries) != 44:
        raise RuntimeError(f"Expected 44 curated motions, got {len(entries)}")
    return entries


def _set_clip_pose(model: mujoco.MjModel, data: mujoco.MjData, clip: dict[str, np.ndarray], frame: int = 0) -> None:
    idx = int(np.clip(frame, 0, len(clip["dof_pos"]) - 1))
    data.qpos[:3] = clip["root_pos"][idx]
    data.qpos[3:7] = clip["root_quat"][idx]
    data.qpos[7:] = clip["dof_pos"][idx]
    if idx == 0:
        data.qvel[:] = _initial_qvel(model, data.qpos.copy(), clip, float(clip["fps"].reshape(-1)[0]))
    else:
        data.qvel[:] = 0.0


class SliderFsm:
    def __init__(self, entries: list[MotionEntry], start_index: int, blend_ticks: int, z_lead_frames: int = 0) -> None:
        self.entries = entries
        self.current_index = int(start_index)
        self.requested_index = int(start_index)
        self.frame = 0
        self.blend_ticks = max(1, int(blend_ticks))
        self.z_lead_frames = int(z_lead_frames)
        self.blend_i = self.blend_ticks
        self.blend_from = self.entries[self.current_index].z[0].astype(np.float32)
        self.z = self.blend_from.copy()
        self.changed = False

    @property
    def current(self) -> MotionEntry:
        return self.entries[self.current_index]

    def request(self, index: int) -> None:
        self.requested_index = int(np.clip(index, 0, len(self.entries) - 1))

    def _apply_request(self) -> None:
        self.changed = False
        if self.requested_index == self.current_index:
            return
        self.blend_from = self.z.copy()
        self.blend_i = 0
        self.frame = 0
        self.current_index = self.requested_index
        self.changed = True

    def next_z(self) -> np.ndarray:
        self._apply_request()
        seq = self.current.z
        target = seq[(self.frame + self.z_lead_frames) % len(seq)].astype(np.float32)
        self.frame += 1
        if self.blend_i < self.blend_ticks:
            alpha = float(self.blend_i + 1) / float(self.blend_ticks)
            self.z = _project_z((1.0 - alpha) * self.blend_from + alpha * target)
            self.blend_i += 1
        else:
            self.z = target
        return self.z


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = _project_root()
        self.artifact_dir = (args.artifact_dir or _artifact_dir(self.project_root)).resolve()
        self.cache_root = (args.cache_root or self.project_root / "cache" / "motion_data" / "z1_mimic_clean").resolve()
        self.latent_dir = (args.latent_dir or self.artifact_dir / "z44_latents").resolve()
        self.entries = _load_entries(self.project_root, self.artifact_dir, self.cache_root, self.latent_dir)
        self.start_index = self._default_start_index(args.start)

        cfg = _load_yaml(self.project_root / "configs" / "robots" / "z1_23dof.yaml")
        default_onnx = self.artifact_dir / "FBcprAuxModel_41613312.onnx"
        policy_path = (Path(args.onnx) if args.onnx else default_onnx).resolve()
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy ONNX not found: {policy_path}")
        # Meta carries control_joint_names (robot-fixed). Prefer a sibling of the
        # chosen ONNX (same stem + .meta.json); fall back to the default artifact meta.
        sibling_meta = policy_path.parent / (policy_path.stem + ".meta.json")
        meta_path = sibling_meta if sibling_meta.exists() else self.artifact_dir / "FBcprAuxModel_41613312.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[ufo-44-slider] policy ONNX: {policy_path}", flush=True)
        print(f"[ufo-44-slider] meta JSON:   {meta_path}", flush=True)
        self.policy_path = policy_path
        self.joint_names = list(meta["control_joint_names"])
        control = cfg["training"]["control"]
        self.control = control
        self.default_q = np.asarray([cfg["default_dof_pos"][j] for j in self.joint_names], dtype=np.float32)
        self.kp = np.asarray([control["stiffness"][j] for j in self.joint_names], dtype=np.float32)
        self.kd = np.asarray([control["damping"][j] for j in self.joint_names], dtype=np.float32)
        self.effort = np.asarray(control["effort_limit"], dtype=np.float32)
        self.action_scale = float(control["action_scale"]) * self.effort / self.kp

        out_dir = self.artifact_dir / "mujoco_44_slider"
        out_dir.mkdir(parents=True, exist_ok=True)
        runtime_xml = _make_runtime_xml(self.project_root / cfg["xml_path"], out_dir, terrain=args.terrain)
        self.model = mujoco.MjModel.from_xml_path(str(runtime_xml))
        self.model.opt.timestep = float(args.sim_dt)
        self.data = mujoco.MjData(self.model)
        self.expert_data = mujoco.MjData(self.model)
        self.qpos_addr, self.qvel_addr = _joint_addresses(self.model, self.joint_names)
        self.actuator_ids = _actuator_ids(self.model, self.joint_names)
        self.policy = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])

        self.history = History()
        self.fsm = SliderFsm(self.entries, self.start_index, args.blend_ticks, args.z_lead_frames)
        self.action_scaled = np.zeros(23, dtype=np.float32)
        self.target_q = self.default_q.copy()
        self.last_torque = np.zeros(23, dtype=np.float32)
        self.sim_step = 0
        self.decimation = max(1, int(round((1.0 / float(args.policy_hz)) / self.model.opt.timestep)))
        self.sim_steps_per_render = max(1, int(round((1.0 / float(args.render_hz)) / self.model.opt.timestep)))
        self.paused = False
        self.running = True
        self._photo: ImageTk.PhotoImage | None = None

        _set_clip_pose(self.model, self.data, self.fsm.current.clip)
        _set_clip_pose(self.model, self.expert_data, self.fsm.current.clip)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.expert_data)

        self.renderer = mujoco.Renderer(self.model, height=RENDER_HEIGHT, width=PANEL_WIDTH)
        self.expert_renderer = mujoco.Renderer(self.model, height=RENDER_HEIGHT, width=PANEL_WIDTH)
        self.cam = mujoco.MjvCamera()
        self.expert_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        mujoco.mjv_defaultFreeCamera(self.model, self.expert_cam)
        self.cam.distance = 3.0
        self.cam.azimuth = 135.0
        self.cam.elevation = -18.0
        self.expert_cam.distance = 3.0
        self.expert_cam.azimuth = 135.0
        self.expert_cam.elevation = -18.0
        self.scene_option = mujoco.MjvOption()
        self.header_font = _load_font(22)
        self.status_font = _load_font(16)

        self.root = tk.Tk()
        self.reset_on_switch = tk.BooleanVar(master=self.root, value=bool(args.reset_on_switch))
        self.slider_var = tk.IntVar(master=self.root, value=self.start_index)
        self.state_var = tk.StringVar(master=self.root)
        self.metrics_var = tk.StringVar(master=self.root)
        self.root.title("Z1 UFO 44-Motion Prior Slider: Expert vs UFO")
        self.root.geometry("1820x820")
        self.root.configure(bg="#15171d")
        self._build_ui()
        self._bind_keys()

    def _default_start_index(self, requested: str) -> int:
        if requested.isdigit():
            return int(np.clip(int(requested), 0, len(self.entries) - 1))
        for entry in self.entries:
            if requested in entry.key:
                return entry.index
        for preferred in ("stagestand", "mabu", "aini"):
            for entry in self.entries:
                if preferred in entry.key:
                    return entry.index
        return 0

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg="#15171d")
        shell.pack(fill="both", expand=True)
        sidebar = tk.Frame(shell, width=360, bg="#10131a")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        viewer = tk.Frame(shell, bg="#15171d")
        viewer.pack(side="right", fill="both", expand=True)

        tk.Label(sidebar, text="44 UFO Priors", bg="#10131a", fg="#f2f5ff", font=("Segoe UI", 18, "bold"), anchor="w", padx=16, pady=16).pack(fill="x")
        tk.Label(sidebar, textvariable=self.state_var, bg="#10131a", fg="#dfe6f7", font=("Consolas", 10), anchor="w", justify="left", padx=16).pack(fill="x")

        tk.Scale(
            sidebar,
            from_=0,
            to=len(self.entries) - 1,
            orient="horizontal",
            variable=self.slider_var,
            command=self._on_slider,
            resolution=1,
            length=310,
            bg="#10131a",
            fg="#dfe6f7",
            troughcolor="#202838",
            highlightthickness=0,
        ).pack(fill="x", padx=16, pady=16)

        controls = tk.Frame(sidebar, bg="#10131a")
        controls.pack(fill="x", padx=14)
        tk.Button(controls, text="Reset Current", command=self.reset_pose, height=2).pack(fill="x", pady=4)
        tk.Button(controls, text="Pause / Resume", command=self.toggle_pause, height=2).pack(fill="x", pady=4)
        tk.Checkbutton(
            controls,
            text="Reset to motion start on switch",
            variable=self.reset_on_switch,
            bg="#10131a",
            fg="#dfe6f7",
            selectcolor="#10131a",
            activebackground="#10131a",
            activeforeground="#ffffff",
            anchor="w",
        ).pack(fill="x", pady=8)
        tk.Button(controls, text="Quit", command=self.quit, height=2).pack(fill="x", pady=4)

        tk.Label(sidebar, textvariable=self.metrics_var, bg="#10131a", fg="#dfe6f7", font=("Consolas", 10), anchor="w", justify="left", padx=16, pady=16).pack(fill="x")
        tk.Label(
            sidebar,
            text="Slider selects one of 44 curated priors.\nLeft/Right nudge slider.\nR reset, Space pause, Q/Esc quit.",
            bg="#10131a",
            fg="#7f8ca8",
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            padx=16,
            pady=10,
        ).pack(fill="x", side="bottom")

        self.image_label = tk.Label(viewer, bg="#15171d")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=12)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _e: self._nudge(-1))
        self.root.bind("<Right>", lambda _e: self._nudge(1))
        self.root.bind("r", lambda _e: self.reset_pose())
        self.root.bind("R", lambda _e: self.reset_pose())
        self.root.bind("<space>", lambda _e: self.toggle_pause())
        self.root.bind("q", lambda _e: self.quit())
        self.root.bind("Q", lambda _e: self.quit())
        self.root.bind("<Escape>", lambda _e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _on_slider(self, value: str) -> None:
        self.fsm.request(int(float(value)))

    def _nudge(self, delta: int) -> None:
        value = int(np.clip(self.slider_var.get() + delta, 0, len(self.entries) - 1))
        self.slider_var.set(value)
        self.fsm.request(value)

    def reset_pose(self) -> None:
        self.history = History()
        self.action_scaled[:] = 0.0
        self.last_torque[:] = 0.0
        _set_clip_pose(self.model, self.data, self.fsm.current.clip)
        _set_clip_pose(self.model, self.expert_data, self.fsm.current.clip)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.expert_data)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def quit(self) -> None:
        self.running = False
        try:
            self.renderer.close()
        except Exception:
            pass
        try:
            self.expert_renderer.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _policy_tick(self) -> None:
        if self.reset_on_switch.get() and self.fsm.requested_index != self.fsm.current_index:
            _set_clip_pose(self.model, self.data, self.entries[self.fsm.requested_index].clip)
            mujoco.mj_forward(self.model, self.data)
            self.history = History()
            self.action_scaled[:] = 0.0

        raw = _raw_obs(self.data, self.qpos_addr, self.qvel_addr, self.default_q, self.action_scaled)
        z = self.fsm.next_z()
        if self.fsm.changed:
            self.history = History()
        obs = _actor_obs(raw, self.history, z)
        action = self.policy.run(["action"], {"actor_obs": obs})[0][0].astype(np.float32)
        self.action_scaled = np.clip(action * float(self.control["normalize_action_to"]), -5.0, 5.0)
        self.target_q = self.default_q + self.action_scaled * self.action_scale
        self.history.add(raw)

    def _sim_step_once(self) -> None:
        if self.sim_step % self.decimation == 0:
            self._policy_tick()
        q = self.data.qpos[self.qpos_addr].astype(np.float32)
        qd = self.data.qvel[self.qvel_addr].astype(np.float32)
        torque = self.kp * (self.target_q - q) - self.kd * qd
        self.last_torque = np.clip(torque, -self.effort, self.effort).astype(np.float32)
        self.data.ctrl[self.actuator_ids] = self.last_torque
        mujoco.mj_step(self.model, self.data)
        self.sim_step += 1

    def _render_data(self, renderer: mujoco.Renderer, data: mujoco.MjData, cam: mujoco.MjvCamera) -> Image.Image:
        root = data.qpos[:3]
        cam.lookat[:] = [float(root[0]), float(root[1]), float(root[2] + 0.15)]
        renderer.update_scene(data, camera=cam, scene_option=self.scene_option)
        return Image.fromarray(renderer.render())

    def _draw_banner(self, draw: ImageDraw.ImageDraw, x: int, title: str, status: str, title_fill: tuple[int, int, int]) -> None:
        for text, y, font, fill in (
            (title, 12, self.header_font, title_fill),
            (status, 48, self.status_font, (220, 255, 220)),
        ):
            text = _fit_text(draw, text, font, PANEL_WIDTH - 32)
            box = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([x + 10, y - 2, min(x + box[2] + 24, x + PANEL_WIDTH - 10), y + box[3] + 8], fill=(0, 0, 0))
            draw.text((x + 16, y), text, fill=fill, font=font)

    def _render(self) -> None:
        entry = self.fsm.current
        expert_frame = max(0, self.fsm.frame - 1) % len(entry.clip["dof_pos"])
        _set_clip_pose(self.model, self.expert_data, entry.clip, expert_frame)
        mujoco.mj_forward(self.model, self.expert_data)

        expert_img = self._render_data(self.expert_renderer, self.expert_data, self.expert_cam)
        ufo_img = self._render_data(self.renderer, self.data, self.cam)
        img = Image.new("RGB", (RENDER_WIDTH, RENDER_HEIGHT), (21, 23, 29))
        img.paste(expert_img, (0, 0))
        img.paste(ufo_img, (PANEL_WIDTH, 0))

        draw = ImageDraw.Draw(img)
        draw.rectangle([PANEL_WIDTH - 2, 0, PANEL_WIDTH + 2, RENDER_HEIGHT], fill=(255, 235, 80))
        run_state = "PAUSED" if self.paused else "RUN"
        self._draw_banner(
            draw,
            0,
            f"EXPERT  |  {entry.index:02d} / 43",
            f"{entry.short_name}  |  {run_state}  |  clip frame {expert_frame:04d}  root_z {self.expert_data.qpos[2]:.3f}",
            (120, 205, 255),
        )
        self._draw_banner(
            draw,
            PANEL_WIDTH,
            f"UFO POLICY  |  {entry.index:02d} / 43",
            f"{entry.short_name}  |  {run_state}  |  "
            f"frame {self.fsm.frame:04d}  root_z {self.data.qpos[2]:.3f}  "
            f"action_max {np.max(np.abs(self.action_scaled)):.2f}  torque_max {np.max(np.abs(self.last_torque)):.1f}",
            (255, 235, 80),
        )

        self._photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._photo)
        self.state_var.set(
            "\n".join(
                [
                    f"index : {entry.index:02d}/43",
                    f"motion: {entry.short_name}",
                    f"group : {entry.group.replace('_train_near10s_ufo.pkl', '')}",
                    f"z len : {len(entry.z)}",
                    f"z lead: {self.fsm.z_lead_frames} frames",
                ]
            )
        )
        self.metrics_var.set(
            "\n".join(
                [
                    f"root_z     {self.data.qpos[2]: .3f}",
                    f"frame      {self.fsm.frame}",
                    f"action_max {np.max(np.abs(self.action_scaled)): .3f}",
                    f"torque_max {np.max(np.abs(self.last_torque)): .2f}",
                    f"sim_step   {self.sim_step}",
                ]
            )
        )

    def _tick(self) -> None:
        if not self.running:
            return
        start = time.perf_counter()
        try:
            if not self.paused:
                for _ in range(self.sim_steps_per_render):
                    self._sim_step_once()
            self._render()
        except Exception as exc:
            print(f"[ufo-44-slider] error: {exc}", flush=True)
            self.quit()
            return
        elapsed_ms = int((time.perf_counter() - start) * 1000.0)
        target_ms = max(1, int(round(1000.0 / float(self.args.render_hz))))
        self.root.after(max(1, target_ms - elapsed_ms), self._tick)

    def run(self) -> None:
        self._tick()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter slider viewer for all 44 Z1 UFO priors.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Policy ONNX to load; overrides the default <artifact-dir>/FBcprAuxModel_41613312.onnx",
    )
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--latent-dir", type=Path, default=None)
    parser.add_argument("--start", default="stagestand")
    parser.add_argument("--blend-ticks", type=int, default=15)
    parser.add_argument(
        "--z-lead-frames",
        type=int,
        default=0,
        help="Feed z[t + N] to the policy while keeping the expert display at frame t. Use this to test phase-lead alignment.",
    )
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--render-hz", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument(
        "--terrain",
        choices=TERRAIN_CHOICES,
        default="plane",
        help="plane uses only the robot MJCF floor; gravel removes that floor and adds one hfield.",
    )
    parser.add_argument("--reset-on-switch", action="store_true")
    return parser.parse_args()


def main() -> None:
    print("[ufo-44-slider] slider selects 44 motions; Left/Right nudge, R reset, Space pause, Q quit", flush=True)
    App(parse_args()).run()


if __name__ == "__main__":
    main()
