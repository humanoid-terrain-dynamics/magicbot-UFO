#!/usr/bin/env python3
"""Tkinter MuJoCo browser for local Z1 UFO motion clips (motion-only, no policy).

A UFO-native counterpart to magicbot_amp_mjlab's ``vis_pkl_dataset_labeled.py``.
It loads the UFO ``*_ufo.pkl`` motion dicts (each a dict of clips in
``root_trans_offset / root_quat / dof_pos / fps`` format) directly — no
``general_motion_retargeting`` / ``mink`` dependency — and renders each clip on
the UFO Z1 MJCF taken from ``configs/robots/z1_23dof.yaml``. Pure kinematic
replay: it sets ``qpos`` from the clip and calls ``mj_forward`` (no dynamics,
no policy, no latent z).

Controls (same orbit-camera UX as the magicbot browser):
  Space        pause / resume
  Left / Right previous / next clip
  Up / Down    step frame while paused
  Home / End   first / last clip
  R            restart current clip
  Q / Esc      quit

  Camera (orbit / rotate the view around the robot):
  Mouse drag   orbit azimuth + tilt elevation
  Mouse wheel  zoom in / out
  [ / ]        rotate azimuth  (left / right)
  , / .        tilt elevation  (down / up)
  - / =        zoom out / in
  O            toggle auto-orbit (slow 360 turntable)
  F            toggle follow-heading (camera trails the robot's travel)
  C            reset camera to default

Usage:
  python tools/z1/vis_z1_motions_ufo.py
  python tools/z1/vis_z1_motions_ufo.py --folder cache/motion_data/z1_mimic_clean
  python tools/z1/vis_z1_motions_ufo.py --pkl path/to/z1_cyclic_locomotion_full_ufo.pkl
  python tools/z1/vis_z1_motions_ufo.py --loop_each --max_frames 300
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Windows: the shell inherits MUJOCO_GL=egl which is invalid here -> glfw works locally.
os.environ.setdefault("MUJOCO_GL", "glfw")

import tkinter as tk

import joblib
import mujoco as mj
import numpy as np

# tools/z1/ is sys.path[0] when run as a script, so deploy_onnx_mujoco is importable
# the same way ufo_44_slider_tk_mujoco.py imports it.
from deploy_onnx_mujoco import _load_yaml, _make_runtime_xml, _project_root
from PIL import Image, ImageDraw, ImageFont, ImageTk

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
LIST_WIDTH = 40

DEFAULT_CAMERA_DISTANCE = 3.0
DEFAULT_CAMERA_ELEVATION = -20.0
DEFAULT_CAMERA_AZIMUTH = 90.0

# --- Interactive camera (orbit / rotation) controls -------------------------
CAMERA_AZIMUTH_STEP = 15.0          # degrees per [ ] nudge
CAMERA_ELEVATION_STEP = 5.0         # degrees per , . nudge
CAMERA_MIN_ELEVATION = -89.0
CAMERA_MAX_ELEVATION = 89.0
CAMERA_MIN_DISTANCE = 1.0
CAMERA_MAX_DISTANCE = 12.0
CAMERA_ZOOM_FACTOR = 0.12           # fractional distance change per zoom step
CAMERA_DRAG_AZIMUTH_GAIN = 0.30     # degrees per mouse-pixel while dragging
CAMERA_DRAG_ELEVATION_GAIN = 0.30
AUTO_ORBIT_DEG_PER_FRAME = 0.8      # turntable speed in auto-orbit mode
FOLLOW_HEADING_OFFSET_DEG = 180.0   # 180 = trail behind the robot; 0 = front chase
FOLLOW_HEADING_WINDOW = 3           # frames each side for heading smoothing


@dataclass
class MotionClip:
    source: Path        # pkl file the clip came from
    key: str            # clip key inside that dict
    root_pos: np.ndarray
    root_quat: np.ndarray
    dof_pos: np.ndarray
    fps: float
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    @property
    def short_label(self) -> str:
        return self.key.replace("__clip000", "").replace("__", " ")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def iter_pkl_files(folder: str | Path, pattern: str = "*.pkl"):
    folder = Path(folder)
    if folder.is_file():
        yield folder
        return
    yield from sorted(folder.rglob(pattern))


def _clip_from_raw(source: Path, key: str, raw: dict) -> MotionClip:
    root_pos = np.asarray(raw["root_trans_offset"], dtype=np.float64)
    root_quat = np.asarray(raw["root_quat"], dtype=np.float64)
    dof_pos = np.asarray(raw["dof_pos"], dtype=np.float64)
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    n = int(min(root_pos.shape[0], root_quat.shape[0], dof_pos.shape[0]))
    return MotionClip(source, key, root_pos[:n], root_quat[:n], dof_pos[:n], fps, n)


def load_clips(files: list[Path], dof_expected: int) -> list[MotionClip]:
    """Flatten UFO ``*_ufo.pkl`` motion dicts into a flat list of clips.

    Skips files that aren't UFO motion dicts and clips whose dof width doesn't
    match the robot (so mixing data for another robot won't crash the viewer).
    """
    clips: list[MotionClip] = []
    for f in files:
        try:
            data = joblib.load(f)
        except Exception as exc:  # corrupted / unsupported pkl
            print(f"[skip] {f.name}: could not load ({exc})", flush=True)
            continue
        if not isinstance(data, dict):
            print(f"[skip] {f.name}: not a UFO motion dict (type {type(data).__name__})", flush=True)
            continue
        for key, raw in data.items():
            if not isinstance(raw, dict) or "dof_pos" not in raw or "root_trans_offset" not in raw:
                continue
            clip = _clip_from_raw(f, str(key), raw)
            if clip.dof_pos.shape[1] != dof_expected:
                print(
                    f"[skip] {f.name}::{key}: dof {clip.dof_pos.shape[1]} != robot {dof_expected}",
                    flush=True,
                )
                continue
            clips.append(clip)
    return clips


def clip_speed(motion: MotionClip) -> float:
    if motion.num_frames < 2:
        return 0.0
    delta_xy = np.diff(motion.root_pos[:, :2], axis=0)
    return float(np.linalg.norm(delta_xy, axis=1).mean() * motion.fps)


# Clearance above the floor after FK grounding (same convention as
# tools/z1/_npz_viewer_core.py). Root z only.
_FLOOR_CLEARANCE = 0.02


def _lift_to_ground(model: mj.MjModel, data: mj.MjData) -> None:
    """Lift the root z just enough to clear floor penetration after FK.

    Same logic as ``_npz_viewer_core.lift_to_ground``: scan ``data.contact`` for
    ``dist < 0`` and raise only ``data.qpos[2]`` by the deepest penetration plus
    ``_FLOOR_CLEARANCE``, then re-forward. No-op when nothing penetrates.
    """
    deepest = 0.0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.dist < 0.0:
            deepest = max(deepest, float(-contact.dist))
    if deepest > 0.0:
        data.qpos[2] += deepest + _FLOOR_CLEARANCE
        mj.mj_forward(model, data)


class App:
    def __init__(
        self,
        pkl_files: list[Path],
        loop_each: bool = False,
        max_frames: int = 0,
        robot_config: Path | None = None,
    ) -> None:
        project_root = _project_root()
        robot_cfg_path = robot_config or (project_root / "configs" / "robots" / "z1_23dof.yaml")
        cfg = _load_yaml(robot_cfg_path)
        out_dir = project_root / "cache" / "motion_vis"
        out_dir.mkdir(parents=True, exist_ok=True)
        runtime_xml = _make_runtime_xml(project_root / cfg["xml_path"], out_dir)

        self.model = mj.MjModel.from_xml_path(str(runtime_xml))
        self.model.vis.global_.offwidth = RENDER_WIDTH
        self.model.vis.global_.offheight = RENDER_HEIGHT
        self.data = mj.MjData(self.model)
        self.dof_dim = int(self.model.nq - 7)  # free root joint = 7 qpos
        if self.dof_dim <= 0:
            raise RuntimeError(
                f"Unexpected Z1 MJCF: nq={self.model.nq} (expected a free root + N hinge joints)."
            )

        self.motions = load_clips(pkl_files, self.dof_dim)
        if not self.motions:
            raise SystemExit(
                f"No loadable Z1 clips found in {len(pkl_files)} file(s) "
                f"(robot dof={self.dof_dim}). Check --folder / --pkl."
            )
        self.speeds = [clip_speed(m) for m in self.motions]

        self.renderer = mj.Renderer(self.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.distance = DEFAULT_CAMERA_DISTANCE
        self.cam.elevation = DEFAULT_CAMERA_ELEVATION
        self.cam.azimuth = DEFAULT_CAMERA_AZIMUTH
        self.scene_option = mj.MjvOption()
        self.auto_orbit = False
        self.follow_heading = False
        self._drag_last: tuple[int, int] | None = None

        # Body to center the camera on; fall back to the root joint origin.
        self.lookat_body = -1
        for name in ("pelvis", "torso", "root", "base"):
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.lookat_body = bid
                break

        self.loop_each = loop_each
        self.max_frames = max_frames
        self.idx = 0
        self.frame = 0
        self.paused = False
        self._running = True
        self._photo: ImageTk.PhotoImage | None = None

        self.header_font = load_font(24)
        self.status_font = load_font(18)

        self.root = tk.Tk()
        self.root.title("Magicbot Z1 UFO Motion Browser")
        self.root.geometry("1660x860")
        self.root.configure(bg="#131722")

        self._build_layout()
        self._bind_keys()
        self._select_index(0, reset_frame=True)
        self._schedule()

    def _build_layout(self) -> None:
        container = tk.Frame(self.root, bg="#131722")
        container.pack(fill="both", expand=True)

        sidebar = tk.Frame(container, bg="#0f1320", width=380)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        viewer_panel = tk.Frame(container, bg="#131722")
        viewer_panel.pack(side="right", fill="both", expand=True)

        tk.Label(
            sidebar, text="Z1 UFO Motion Clips", bg="#0f1320", fg="#eef3ff",
            font=("Segoe UI", 16, "bold"), anchor="w", padx=16, pady=12,
        ).pack(fill="x")

        self.folder_var = tk.StringVar(value=str(self.motions[0].source.parent))
        tk.Label(
            sidebar, textvariable=self.folder_var, bg="#0f1320", fg="#7f8ca8",
            font=("Consolas", 9), anchor="w", justify="left", wraplength=340, padx=16,
        ).pack(fill="x")

        self.summary_var = tk.StringVar()
        tk.Label(
            sidebar, textvariable=self.summary_var, bg="#0f1320", fg="#b5bfd6",
            font=("Segoe UI", 10), anchor="w", justify="left", padx=16, pady=10,
        ).pack(fill="x")

        list_frame = tk.Frame(sidebar, bg="#0f1320")
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame, width=LIST_WIDTH, bg="#171d2d", fg="#dbe2f2",
            selectbackground="#2b6cff", selectforeground="#ffffff", activestyle="none",
            font=("Consolas", 10), yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        for clip, speed in zip(self.motions, self.speeds):
            seconds = clip.duration_seconds
            self.listbox.insert("end", f"{clip.short_label[:28]:<28.28}  {seconds:5.1f}s  {speed:4.2f}m/s")
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.info_var = tk.StringVar()
        tk.Label(
            sidebar, textvariable=self.info_var, bg="#0f1320", fg="#dbe2f2",
            font=("Consolas", 10), anchor="w", justify="left", padx=16, pady=10,
        ).pack(fill="x")

        tk.Label(
            sidebar,
            text=(
                "Space pause/resume\nLeft/Right prev/next clip\nUp/Down step frame when paused\n"
                "R restart clip   Home/End jump\nQ or Esc quit\n\nCamera orbit:\n"
                "drag = orbit/tilt  wheel = zoom\n[ ] azimuth   , . tilt   - = zoom\n"
                "O auto-orbit   F follow   C reset"
            ),
            bg="#0f1320", fg="#7f8ca8", font=("Segoe UI", 9), anchor="w", justify="left",
            padx=16, pady=12,
        ).pack(fill="x")

        self.image_label = tk.Label(viewer_panel, bg="#131722")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=12)
        self.image_label.bind("<ButtonPress-1>", self._on_mouse_down)
        self.image_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.image_label.bind("<MouseWheel>", self._on_mouse_wheel)
        self.image_label.bind("<Button-4>", lambda _: self.zoom_camera(-1))
        self.image_label.bind("<Button-5>", lambda _: self.zoom_camera(1))

        total_frames = sum(m.num_frames for m in self.motions)
        self.summary_var.set(f"{len(self.motions)} clips  |  {total_frames} total frames")

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _: self.toggle_pause())
        self.root.bind("<Left>", lambda _: self.next_clip(-1))
        self.root.bind("<Right>", lambda _: self.next_clip(1))
        self.root.bind("<Up>", lambda _: self.step_frame(1))
        self.root.bind("<Down>", lambda _: self.step_frame(-1))
        self.root.bind("<Home>", lambda _: self._select_index(0, reset_frame=True))
        self.root.bind("<End>", lambda _: self._select_index(len(self.motions) - 1, reset_frame=True))
        self.root.bind("r", lambda _: self.restart_clip())
        self.root.bind("q", lambda _: self.quit())
        self.root.bind("<Escape>", lambda _: self.quit())
        self.root.bind("bracketleft", lambda _: self.rotate_azimuth(-1))
        self.root.bind("bracketright", lambda _: self.rotate_azimuth(1))
        self.root.bind("comma", lambda _: self.rotate_elevation(-1))
        self.root.bind("period", lambda _: self.rotate_elevation(1))
        self.root.bind("minus", lambda _: self.zoom_camera(1))
        self.root.bind("equal", lambda _: self.zoom_camera(-1))
        self.root.bind("plus", lambda _: self.zoom_camera(-1))
        self.root.bind("o", lambda _: self.toggle_auto_orbit())
        self.root.bind("f", lambda _: self.toggle_follow_heading())
        self.root.bind("c", lambda _: self.reset_camera())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _on_listbox_select(self, _event) -> None:
        selection = self.listbox.curselection()
        if selection:
            self._select_index(selection[0], reset_frame=True)

    def _select_index(self, idx: int, reset_frame: bool) -> None:
        self.idx = idx % len(self.motions)
        if reset_frame:
            self.frame = 0
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.idx)
        self.listbox.activate(self.idx)
        self.listbox.see(self.idx)
        self._update_info_panel()

    def _update_info_panel(self) -> None:
        clip = self.motions[self.idx]
        speed = self.speeds[self.idx]
        seconds = clip.duration_seconds
        self.info_var.set(
            "\n".join(
                [
                    f"[{self.idx + 1}/{len(self.motions)}] {clip.key}",
                    f"source : {clip.source.name}",
                    f"length : {clip.num_frames} frames  ({seconds:.2f}s @ {clip.fps:.1f} fps)",
                    f"dof    : {clip.dof_pos.shape[1]}",
                    f"speed  : {speed:.3f} m/s",
                    f"cam    : {self._camera_mode_tag()} az {self.cam.azimuth:.0f} elev {self.cam.elevation:.0f} dist {self.cam.distance:.1f}",
                ]
            )
        )

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def restart_clip(self) -> None:
        self.frame = 0

    def next_clip(self, delta: int) -> None:
        self._select_index(self.idx + delta, reset_frame=True)

    def step_frame(self, delta: int) -> None:
        if not self.paused:
            self.paused = True
        clip = self.motions[self.idx]
        clip_frames = self._clip_frames(clip)
        self.frame = min(max(self.frame + delta, 0), clip_frames - 1)
        self.render_current()

    def quit(self) -> None:
        self._running = False
        try:
            self.renderer.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---- interactive camera (orbit / rotation) -----------------------------
    def _enter_manual_camera(self) -> None:
        self.auto_orbit = False
        self.follow_heading = False

    def rotate_azimuth(self, sign: int) -> None:
        self._enter_manual_camera()
        self.cam.azimuth = (self.cam.azimuth + sign * CAMERA_AZIMUTH_STEP) % 360.0

    def rotate_elevation(self, sign: int) -> None:
        self._enter_manual_camera()
        self.cam.elevation = float(
            np.clip(self.cam.elevation + sign * CAMERA_ELEVATION_STEP, CAMERA_MIN_ELEVATION, CAMERA_MAX_ELEVATION)
        )

    def zoom_camera(self, sign: int) -> None:
        self._enter_manual_camera()
        self.cam.distance = float(
            np.clip(self.cam.distance * (1.0 + sign * CAMERA_ZOOM_FACTOR), CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE)
        )

    def toggle_auto_orbit(self) -> None:
        self.follow_heading = False
        self.auto_orbit = not self.auto_orbit

    def toggle_follow_heading(self) -> None:
        self.auto_orbit = False
        self.follow_heading = not self.follow_heading

    def reset_camera(self) -> None:
        self.auto_orbit = False
        self.follow_heading = False
        self.cam.distance = DEFAULT_CAMERA_DISTANCE
        self.cam.elevation = DEFAULT_CAMERA_ELEVATION
        self.cam.azimuth = DEFAULT_CAMERA_AZIMUTH

    def _motion_heading_deg(self, clip: MotionClip, frame_idx: int) -> float | None:
        root_pos = clip.root_pos
        num = root_pos.shape[0]
        lo = max(0, frame_idx - FOLLOW_HEADING_WINDOW)
        hi = min(num - 1, frame_idx + FOLLOW_HEADING_WINDOW)
        dx = root_pos[hi, 0] - root_pos[lo, 0]
        dy = root_pos[hi, 1] - root_pos[lo, 1]
        if dx * dx + dy * dy < 1e-8:
            return None
        return float(np.degrees(np.arctan2(dy, dx)))

    def _camera_mode_tag(self) -> str:
        if self.auto_orbit:
            return "ORBIT"
        if self.follow_heading:
            return "FOLLOW"
        return "free"

    def _apply_camera_mode(self, clip: MotionClip) -> None:
        if self.follow_heading:
            heading = self._motion_heading_deg(clip, self.frame)
            if heading is not None:
                self.cam.azimuth = (heading + FOLLOW_HEADING_OFFSET_DEG) % 360.0
        elif self.auto_orbit:
            self.cam.azimuth = (self.cam.azimuth + AUTO_ORBIT_DEG_PER_FRAME) % 360.0

    def _on_mouse_down(self, event) -> None:
        self._enter_manual_camera()
        self._drag_last = (event.x, event.y)

    def _on_mouse_drag(self, event) -> None:
        if self._drag_last is None:
            self._drag_last = (event.x, event.y)
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        self.cam.azimuth = (self.cam.azimuth - dx * CAMERA_DRAG_AZIMUTH_GAIN) % 360.0
        self.cam.elevation = float(
            np.clip(self.cam.elevation + dy * CAMERA_DRAG_ELEVATION_GAIN, CAMERA_MIN_ELEVATION, CAMERA_MAX_ELEVATION)
        )

    def _on_mouse_release(self, _event) -> None:
        self._drag_last = None

    def _on_mouse_wheel(self, event) -> None:
        self.zoom_camera(-1 if event.delta > 0 else 1)

    def _clip_frames(self, clip: MotionClip) -> int:
        if self.max_frames > 0:
            return min(clip.num_frames, self.max_frames)
        return clip.num_frames

    def _advance_after_end(self) -> None:
        if self.loop_each:
            self.frame = 0
            return
        self._select_index(self.idx + 1, reset_frame=True)

    def _apply_motion_frame(self, clip: MotionClip, frame_idx: int) -> None:
        self.data.qpos[:3] = clip.root_pos[frame_idx]
        self.data.qpos[3:7] = clip.root_quat[frame_idx]
        self.data.qpos[7 : 7 + clip.dof_pos.shape[1]] = clip.dof_pos[frame_idx]
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)
        _lift_to_ground(self.model, self.data)

    def render_current(self) -> None:
        clip = self.motions[self.idx]
        clip_frames = self._clip_frames(clip)
        if self.frame >= clip_frames:
            self._advance_after_end()
            clip = self.motions[self.idx]
            clip_frames = self._clip_frames(clip)

        self._apply_motion_frame(clip, self.frame)
        if self.lookat_body >= 0:
            self.cam.lookat[:] = self.data.xpos[self.lookat_body]
        else:
            self.cam.lookat[:] = self.data.qpos[:3]
        self._apply_camera_mode(clip)
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.scene_option)
        rgb = self.renderer.render()

        header = f"[{self.idx + 1}/{len(self.motions)}] {clip.short_label}  |  {self.speeds[self.idx]:.2f} m/s"
        state = "PAUSED" if self.paused else "PLAY"
        status = (
            f"frame {self.frame + 1}/{clip_frames}  |  {state}  |  cam: {self._camera_mode_tag()}  |  "
            "[ ] orbit  drag=tilt  wheel=zoom  o auto  f follow  c reset  Q quit"
        )
        print("\r" + status[:140], end="", flush=True)

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        header_box = draw.textbbox((0, 0), header, font=self.header_font)
        draw.rectangle([10, 10, 18 + header_box[2], 18 + header_box[3]], fill=(0, 0, 0))
        draw.text((14, 12), header, fill=(255, 230, 0), font=self.header_font)

        status_box = draw.textbbox((0, 0), status, font=self.status_font)
        top = 24 + header_box[3]
        draw.rectangle([10, top, 18 + status_box[2], top + 8 + status_box[3]], fill=(0, 0, 0))
        draw.text((14, top + 2), status, fill=(210, 255, 210), font=self.status_font)

        self._photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self._photo)
        self._update_info_panel()

        if not self.paused:
            self.frame += 1

    def _schedule(self) -> None:
        if not self._running:
            return
        try:
            self.render_current()
        except Exception as exc:
            print(f"\nrender error: {exc}")
            self.quit()
            return

        fps = self.motions[self.idx].fps
        delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 33
        self.root.after(delay_ms, self._schedule)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--folder",
        default=str(project_root / "cache" / "motion_data" / "z1_mimic_clean"),
        help="Folder to scan for *_ufo.pkl motion dicts (default: cache/motion_data/z1_mimic_clean).",
    )
    parser.add_argument(
        "--pattern", default="*_full_ufo.pkl",
        help="File pattern under --folder (default *_full_ufo.pkl; use *_ufo.pkl to include _train_near10s clips too).",
    )
    parser.add_argument(
        "--pkl", nargs="+", type=Path, default=None,
        help="Explicit pkl file(s) to browse; overrides --folder.",
    )
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot yaml (default: configs/robots/z1_23dof.yaml).")
    parser.add_argument("--loop_each", action="store_true", help="Loop each clip until you switch.")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = full clip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.pkl if args.pkl else list(iter_pkl_files(args.folder, args.pattern))
    if not files:
        print(f"No PKL files found (folder={args.folder}, pattern={args.pattern}). Pass --pkl FILE [...].")
        return
    print(
        f"Scanning {len(files)} pkl file(s). Controls: Space pause  Left/Right clip  "
        "Up/Down frame  R restart  Q quit  |  Camera: drag orbit  wheel zoom  "
        "[ ] ,. tilt  o auto  f follow  c reset",
        flush=True,
    )
    App(
        pkl_files=files,
        loop_each=args.loop_each,
        max_frames=args.max_frames,
        robot_config=args.robot_config,
    ).run()
    print("\nDone!")


if __name__ == "__main__":
    main()
