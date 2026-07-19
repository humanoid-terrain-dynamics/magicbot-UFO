"""Tkinter-hosted MuJoCo viewer for UFO latent finite-state switching.

This runs the exported Z1 UFO ONNX actor in MuJoCo and exposes a simple
finite-state UI:

  IDLE/STAND, aini, kick, cekongfan, mabu

IDLE/STAND intentionally does not use a zero latent. By default it holds the
first pose of the ``mabu`` clip as a temporary stand target. For deployment, a
real idle/stand mocap latent should be trained or curated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import tkinter as tk
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import joblib
import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageTk

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


RENDER_WIDTH = 1120
RENDER_HEIGHT = 700
STATE_ORDER = ["IDLE", "aini", "kick", "cekongfan", "mabu"]
KEY_TO_STATE = {"0": "IDLE", "1": "aini", "2": "kick", "3": "cekongfan", "4": "mabu"}


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


def _load_clip_data(project_root: Path, artifact_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    clips: dict[str, dict[str, np.ndarray]] = {}
    for name, info in CLIPS.items():
        npz = np.load(project_root / info["npz"], allow_pickle=True)
        clips[name] = {
            "root_pos": np.asarray(npz["root_pos"], dtype=np.float32),
            "root_quat": np.asarray(npz["root_quat"], dtype=np.float32),
            "dof_pos": np.asarray(npz["dof_pos"], dtype=np.float32),
            "fps": np.asarray(npz["fps"], dtype=np.float32),
            "z": np.asarray(joblib.load(artifact_dir / info["z"]), dtype=np.float32),
        }
    return clips


def _set_clip_pose(model: mujoco.MjModel, data: mujoco.MjData, clip: dict[str, np.ndarray], frame: int = 0) -> None:
    idx = int(np.clip(frame, 0, len(clip["dof_pos"]) - 1))
    data.qpos[:3] = clip["root_pos"][idx]
    data.qpos[3:7] = clip["root_quat"][idx]
    data.qpos[7:] = clip["dof_pos"][idx]
    if idx == 0:
        data.qvel[:] = _initial_qvel(model, data.qpos.copy(), clip, float(clip["fps"].reshape(-1)[0]))
    else:
        data.qvel[:] = 0.0


class LatentFsm:
    def __init__(self, clips: dict[str, dict[str, np.ndarray]], blend_ticks: int) -> None:
        self.clips = clips
        self.current = "IDLE"
        self.requested = "IDLE"
        self.frame = 0
        self.blend_ticks = max(1, int(blend_ticks))
        self.blend_i = self.blend_ticks
        self.blend_from = np.zeros(256, dtype=np.float32)
        self.z = np.zeros(256, dtype=np.float32)
        self.changed = False

    def request(self, state: str) -> None:
        if state not in STATE_ORDER:
            raise ValueError(f"Unknown FSM state: {state}")
        self.requested = state

    def _apply_request(self) -> None:
        self.changed = False
        if self.requested == self.current:
            return
        self.blend_from = self.z.copy()
        self.blend_i = 0
        self.frame = 0
        self.current = self.requested
        self.changed = True

    def next_z(self) -> np.ndarray | None:
        self._apply_request()
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


class UfoTkMujocoApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = _project_root()
        self.artifact_dir = (args.artifact_dir or _artifact_dir(self.project_root)).resolve()
        self.output_dir = self.artifact_dir / "mujoco_fsm_tk"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg = _load_yaml(self.project_root / "configs" / "robots" / "z1_23dof.yaml")
        meta = json.loads((self.artifact_dir / "FBcprAuxModel_41613312.meta.json").read_text(encoding="utf-8"))
        self.joint_names = list(meta["control_joint_names"])
        control = cfg["training"]["control"]
        self.control = control

        self.default_q = np.asarray([cfg["default_dof_pos"][j] for j in self.joint_names], dtype=np.float32)
        self.kp = np.asarray([control["stiffness"][j] for j in self.joint_names], dtype=np.float32)
        self.kd = np.asarray([control["damping"][j] for j in self.joint_names], dtype=np.float32)
        self.effort = np.asarray(control["effort_limit"], dtype=np.float32)
        self.action_scale = float(control["action_scale"]) * self.effort / self.kp

        runtime_xml = _make_runtime_xml(self.project_root / cfg["xml_path"], self.output_dir)
        self.model = mujoco.MjModel.from_xml_path(str(runtime_xml))
        self.model.opt.timestep = float(args.sim_dt)
        self.data = mujoco.MjData(self.model)
        self.qpos_addr, self.qvel_addr = _joint_addresses(self.model, self.joint_names)
        self.actuator_ids = _actuator_ids(self.model, self.joint_names)
        self.clips = _load_clip_data(self.project_root, self.artifact_dir)
        self.idle_clip = self.clips[args.idle_source]
        self.idle_q = self.idle_clip["dof_pos"][0].astype(np.float32)

        self.policy = ort.InferenceSession(
            str(self.artifact_dir / "FBcprAuxModel_41613312.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.history = History()
        self.fsm = LatentFsm(self.clips, blend_ticks=args.blend_ticks)
        self.action_scaled = np.zeros(23, dtype=np.float32)
        self.target_q = self.idle_q.copy()
        self.last_torque = np.zeros(23, dtype=np.float32)
        self.control_counter = 0
        self.sim_step = 0
        self.paused = False
        self.running = True
        self._photo: ImageTk.PhotoImage | None = None

        self.decimation = max(1, int(round((1.0 / float(args.policy_hz)) / self.model.opt.timestep)))
        self.sim_steps_per_render = max(1, int(round((1.0 / float(args.render_hz)) / self.model.opt.timestep)))

        _set_clip_pose(self.model, self.data, self.idle_clip)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.renderer = mujoco.Renderer(self.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self.cam.distance = 3.0
        self.cam.azimuth = 135.0
        self.cam.elevation = -18.0
        self.scene_option = mujoco.MjvOption()
        self.header_font = _load_font(22)
        self.status_font = _load_font(16)

        self.root = tk.Tk()
        self.reset_on_switch = tk.BooleanVar(master=self.root, value=bool(args.reset_on_switch))
        self.root.title("Z1 UFO FSM MuJoCo - Tkinter")
        self.root.geometry("1480x820")
        self.root.configure(bg="#15171d")
        self._build_ui()
        self._bind_keys()
        self._set_active_button()

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg="#15171d")
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, width=320, bg="#10131a")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        viewer = tk.Frame(shell, bg="#15171d")
        viewer.pack(side="right", fill="both", expand=True)

        tk.Label(
            sidebar,
            text="UFO FSM",
            bg="#10131a",
            fg="#f2f5ff",
            font=("Segoe UI", 18, "bold"),
            anchor="w",
            padx=16,
            pady=16,
        ).pack(fill="x")

        self.state_var = tk.StringVar()
        tk.Label(
            sidebar,
            textvariable=self.state_var,
            bg="#10131a",
            fg="#aeb8cf",
            font=("Consolas", 10),
            anchor="w",
            justify="left",
            padx=16,
        ).pack(fill="x")

        button_panel = tk.Frame(sidebar, bg="#10131a")
        button_panel.pack(fill="x", padx=14, pady=14)
        self.state_buttons: dict[str, tk.Button] = {}
        labels = {
            "IDLE": "0  IDLE/STAND",
            "aini": "1  aini",
            "kick": "2  kick",
            "cekongfan": "3  cekongfan",
            "mabu": "4  mabu",
        }
        for state in STATE_ORDER:
            btn = tk.Button(
                button_panel,
                text=labels[state],
                command=lambda s=state: self.request_state(s),
                height=2,
                anchor="w",
                padx=12,
                bg="#202838",
                fg="#eef3ff",
                activebackground="#315fba",
                activeforeground="#ffffff",
                relief="flat",
                font=("Segoe UI", 11, "bold" if state == "IDLE" else "normal"),
            )
            btn.pack(fill="x", pady=4)
            self.state_buttons[state] = btn

        controls = tk.Frame(sidebar, bg="#10131a")
        controls.pack(fill="x", padx=14)
        tk.Button(controls, text="Reset Pose", command=self.reset_pose, height=2).pack(fill="x", pady=4)
        tk.Button(controls, text="Pause / Resume", command=self.toggle_pause, height=2).pack(fill="x", pady=4)
        tk.Checkbutton(
            controls,
            text="Reset to clip start on switch",
            variable=self.reset_on_switch,
            bg="#10131a",
            fg="#dfe6f7",
            selectcolor="#10131a",
            activebackground="#10131a",
            activeforeground="#ffffff",
            anchor="w",
        ).pack(fill="x", pady=8)
        tk.Button(controls, text="Quit", command=self.quit, height=2).pack(fill="x", pady=4)

        self.metrics_var = tk.StringVar()
        tk.Label(
            sidebar,
            textvariable=self.metrics_var,
            bg="#10131a",
            fg="#dfe6f7",
            font=("Consolas", 10),
            anchor="w",
            justify="left",
            padx=16,
            pady=16,
        ).pack(fill="x")

        tk.Label(
            sidebar,
            text=(
                "Keys: 0-4 switch states\n"
                "R reset   Space pause\n"
                "Q / Esc quit\n\n"
                f"Idle source: {self.args.idle_source}\n"
                "Zero latent is not used as stand."
            ),
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
        for key, state in KEY_TO_STATE.items():
            self.root.bind(key, lambda _event, s=state: self.request_state(s))
        self.root.bind("r", lambda _event: self.reset_pose())
        self.root.bind("R", lambda _event: self.reset_pose())
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self.root.bind("q", lambda _event: self.quit())
        self.root.bind("Q", lambda _event: self.quit())
        self.root.bind("<Escape>", lambda _event: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def request_state(self, state: str) -> None:
        self.fsm.request(state)

    def _set_active_button(self) -> None:
        for state, btn in self.state_buttons.items():
            if state == self.fsm.current:
                btn.configure(bg="#2e6be6", fg="#ffffff")
            else:
                btn.configure(bg="#202838", fg="#eef3ff")

    def reset_pose(self) -> None:
        self.history = History()
        self.action_scaled[:] = 0.0
        self.last_torque[:] = 0.0
        if self.fsm.current == "IDLE":
            _set_clip_pose(self.model, self.data, self.idle_clip)
            self.data.qvel[:] = 0.0
            self.target_q = self.idle_q.copy()
        else:
            _set_clip_pose(self.model, self.data, self.clips[self.fsm.current])
        mujoco.mj_forward(self.model, self.data)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def quit(self) -> None:
        self.running = False
        try:
            self.renderer.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _policy_tick(self) -> None:
        if self.reset_on_switch.get() and self.fsm.requested != self.fsm.current and self.fsm.requested != "IDLE":
            _set_clip_pose(self.model, self.data, self.clips[self.fsm.requested])
            mujoco.mj_forward(self.model, self.data)
            self.history = History()
            self.action_scaled[:] = 0.0

        raw = _raw_obs(self.data, self.qpos_addr, self.qvel_addr, self.default_q, self.action_scaled)
        z = self.fsm.next_z()
        if self.fsm.changed:
            self.history = History()
            self._set_active_button()

        if z is None:
            self.action_scaled[:] = 0.0
            self.target_q = self.idle_q.copy()
        else:
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

    def _render(self) -> None:
        root = self.data.qpos[:3]
        self.cam.lookat[:] = [float(root[0]), float(root[1]), float(root[2] + 0.15)]
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.scene_option)
        rgb = self.renderer.render()

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        title = f"Z1 UFO FSM  |  {self.fsm.current}  |  {'PAUSED' if self.paused else 'RUN'}"
        status = (
            f"frame {self.fsm.frame:04d}  root_z {self.data.qpos[2]:.3f}  "
            f"action_max {np.max(np.abs(self.action_scaled)):.2f}  "
            f"torque_max {np.max(np.abs(self.last_torque)):.1f}"
        )
        for text, y, font, fill in (
            (title, 12, self.header_font, (255, 235, 80)),
            (status, 48, self.status_font, (220, 255, 220)),
        ):
            box = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([10, y - 2, box[2] + 24, y + box[3] + 8], fill=(0, 0, 0))
            draw.text((16, y), text, fill=fill, font=font)

        self._photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._photo)
        self.state_var.set(
            "\n".join(
                [
                    f"artifact: {self.artifact_dir.name}",
                    f"state   : {self.fsm.current}",
                    f"request : {self.fsm.requested}",
                    f"policy  : {self.args.policy_hz:.1f} Hz",
                    f"render  : {self.args.render_hz:.1f} Hz",
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
            print(f"[ufo-fsm-tk] error: {exc}", flush=True)
            self.quit()
            return

        elapsed_ms = int((time.perf_counter() - start) * 1000.0)
        target_ms = max(1, int(round(1000.0 / float(self.args.render_hz))))
        self.root.after(max(1, target_ms - elapsed_ms), self._tick)

    def run(self) -> None:
        self._tick()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter UFO FSM MuJoCo viewer for Z1.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--idle-source", choices=sorted(CLIPS.keys()), default="mabu")
    parser.add_argument("--blend-ticks", type=int, default=15)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--render-hz", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--reset-on-switch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "[ufo-fsm-tk] controls: 0=IDLE 1=aini 2=kick 3=cekongfan 4=mabu "
        "R=reset Space=pause Q/Esc=quit",
        flush=True,
    )
    UfoTkMujocoApp(args).run()


if __name__ == "__main__":
    main()
