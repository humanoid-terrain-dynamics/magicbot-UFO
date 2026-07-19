#!/usr/bin/env python3
"""Tkinter-hosted MuJoCo browser for UFO Z1 NPZ clips.

Renders one selected clip offscreen with MuJoCo and shows it inside a Tkinter
window. The left panel lists all `.npz` files under a folder, defaulting to the
converted UFO Z1 robot-state dataset.

Controls:
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
  python tools/z1/vis_npz_dataset_labeled.py
  python tools/z1/vis_npz_dataset_labeled.py --npz_folder humanoidverse/data/z1_mimic_robot_state_npz
  python tools/z1/vis_npz_dataset_labeled.py --loop_each
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk

from _npz_viewer_core import (
    DEFAULT_CAMERA_AZIMUTH,
    DEFAULT_CAMERA_DISTANCE,
    DEFAULT_CAMERA_ELEVATION,
    MotionClip,
    Z1_MJCF,
    apply_motion_frame,
    clip_kind,
    clip_speed,
    create_model_and_data,
    iter_npz_files,
    load_npz_motion,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_NPZ_FOLDER = REPO_ROOT / "humanoidverse" / "data" / "z1_mimic_robot_state_npz"
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
LIST_WIDTH = 38

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
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


class App:
    def __init__(self, npz_files: list[Path], loop_each: bool = False, max_frames: int = 0) -> None:
        self.files = npz_files
        self.loop_each = loop_each
        self.max_frames = max_frames
        self.motions = [load_npz_motion(path) for path in self.files]
        self.speeds = [clip_speed(motion) for motion in self.motions]
        self.idx = 0
        self.frame = 0
        self.paused = False
        self._running = True
        self._photo: ImageTk.PhotoImage | None = None

        self.model, self.data = create_model_and_data()
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

        self.header_font = load_font(24)
        self.status_font = load_font(18)

        self.root = tk.Tk()
        self.root.title("MagicBot Z1 AMP NPZ Browser")
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

        title = tk.Label(
            sidebar,
            text="AMP Clips",
            bg="#0f1320",
            fg="#eef3ff",
            font=("Segoe UI", 16, "bold"),
            anchor="w",
            padx=16,
            pady=12,
        )
        title.pack(fill="x")

        self.folder_var = tk.StringVar(value=str(self.files[0].parent if self.files else DEFAULT_NPZ_FOLDER))
        folder_label = tk.Label(
            sidebar,
            textvariable=self.folder_var,
            bg="#0f1320",
            fg="#7f8ca8",
            font=("Consolas", 9),
            anchor="w",
            justify="left",
            wraplength=340,
            padx=16,
            pady=0,
        )
        folder_label.pack(fill="x")

        self.summary_var = tk.StringVar()
        summary_label = tk.Label(
            sidebar,
            textvariable=self.summary_var,
            bg="#0f1320",
            fg="#b5bfd6",
            font=("Segoe UI", 10),
            anchor="w",
            justify="left",
            padx=16,
            pady=10,
        )
        summary_label.pack(fill="x")

        list_frame = tk.Frame(sidebar, bg="#0f1320")
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame,
            width=LIST_WIDTH,
            bg="#171d2d",
            fg="#dbe2f2",
            selectbackground="#2b6cff",
            selectforeground="#ffffff",
            activestyle="none",
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        for path, motion, speed in zip(self.files, self.motions, self.speeds):
            fps = motion.fps
            frames = motion.num_frames
            seconds = frames / fps if fps > 0 else 0.0
            self.listbox.insert(
                "end",
                f"{path.stem:<24.24}  {seconds:5.1f}s  {speed:4.2f}m/s",
            )
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.info_var = tk.StringVar()
        info = tk.Label(
            sidebar,
            textvariable=self.info_var,
            bg="#0f1320",
            fg="#dbe2f2",
            font=("Consolas", 10),
            anchor="w",
            justify="left",
            padx=16,
            pady=10,
        )
        info.pack(fill="x")

        controls = tk.Label(
            sidebar,
            text=(
                "Space pause/resume\n"
                "Left/Right prev/next clip\n"
                "Up/Down step frame when paused\n"
                "R restart clip   Home/End jump\n"
                "Q or Esc quit\n"
                "\n"
                "Camera orbit:\n"
                "drag = orbit/tilt  wheel = zoom\n"
                "[ ] azimuth   , . tilt   - = zoom\n"
                "O auto-orbit   F follow   C reset"
            ),
            bg="#0f1320",
            fg="#7f8ca8",
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            padx=16,
            pady=12,
        )
        controls.pack(fill="x")

        self.image_label = tk.Label(viewer_panel, bg="#131722")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=12)

        total_frames = sum(motion.num_frames for motion in self.motions)
        self.summary_var.set(f"{len(self.files)} clips  |  {total_frames} total frames")

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _: self.toggle_pause())
        self.root.bind("<Left>", lambda _: self.next_clip(-1))
        self.root.bind("<Right>", lambda _: self.next_clip(1))
        self.root.bind("<Up>", lambda _: self.step_frame(1))
        self.root.bind("<Down>", lambda _: self.step_frame(-1))
        self.root.bind("<Home>", lambda _: self._select_index(0, reset_frame=True))
        self.root.bind("<End>", lambda _: self._select_index(len(self.files) - 1, reset_frame=True))
        self.root.bind("r", lambda _: self.restart_clip())
        self.root.bind("q", lambda _: self.quit())
        self.root.bind("<Escape>", lambda _: self.quit())
        # Camera orbit / rotation
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
        self.image_label.bind("<ButtonPress-1>", self._on_mouse_down)
        self.image_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.image_label.bind("<MouseWheel>", self._on_mouse_wheel)
        self.image_label.bind("<Button-4>", lambda _: self.zoom_camera(-1))
        self.image_label.bind("<Button-5>", lambda _: self.zoom_camera(1))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _on_listbox_select(self, _event) -> None:
        selection = self.listbox.curselection()
        if selection:
            self._select_index(selection[0], reset_frame=True)

    def _select_index(self, idx: int, reset_frame: bool) -> None:
        self.idx = idx % len(self.files)
        if reset_frame:
            self.frame = 0
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.idx)
        self.listbox.activate(self.idx)
        self.listbox.see(self.idx)
        self._update_info_panel()

    def _update_info_panel(self) -> None:
        path = self.files[self.idx]
        motion: MotionClip = self.motions[self.idx]
        speed = self.speeds[self.idx]
        fps = motion.fps
        frames = motion.num_frames
        seconds = frames / fps if fps > 0 else 0.0
        kind = clip_kind(path.stem) or "generic"
        self.info_var.set(
            "\n".join(
                [
                    f"[{self.idx + 1}/{len(self.files)}] {path.name}",
                    f"type   : {kind}",
                    f"length : {frames} frames  ({seconds:.2f}s @ {fps:.1f} fps)",
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
        motion = self.motions[self.idx]
        clip_frames = self._clip_frames(motion)
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
            np.clip(
                self.cam.elevation + sign * CAMERA_ELEVATION_STEP,
                CAMERA_MIN_ELEVATION,
                CAMERA_MAX_ELEVATION,
            )
        )

    def zoom_camera(self, sign: int) -> None:
        self._enter_manual_camera()
        self.cam.distance = float(
            np.clip(
                self.cam.distance * (1.0 + sign * CAMERA_ZOOM_FACTOR),
                CAMERA_MIN_DISTANCE,
                CAMERA_MAX_DISTANCE,
            )
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

    def _motion_heading_deg(self, motion: MotionClip, frame_idx: int) -> float | None:
        root_pos = motion.root_pos
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

    def _apply_camera_mode(self, motion: MotionClip) -> None:
        if self.follow_heading:
            heading = self._motion_heading_deg(motion, self.frame)
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
            np.clip(
                self.cam.elevation + dy * CAMERA_DRAG_ELEVATION_GAIN,
                CAMERA_MIN_ELEVATION,
                CAMERA_MAX_ELEVATION,
            )
        )

    def _on_mouse_release(self, _event) -> None:
        self._drag_last = None

    def _on_mouse_wheel(self, event) -> None:
        # Windows wheel: event.delta in multiples of 120, up = positive = zoom in
        self.zoom_camera(-1 if event.delta > 0 else 1)

    def _clip_frames(self, motion: MotionClip) -> int:
        if self.max_frames > 0:
            return min(motion.num_frames, self.max_frames)
        return motion.num_frames

    def _advance_after_end(self) -> None:
        if self.loop_each:
            self.frame = 0
            return
        self._select_index(self.idx + 1, reset_frame=True)

    def render_current(self) -> None:
        motion = self.motions[self.idx]
        clip_frames = self._clip_frames(motion)
        if self.frame >= clip_frames:
            self._advance_after_end()
            motion = self.motions[self.idx]
            clip_frames = self._clip_frames(motion)

        root_pos = apply_motion_frame(self.model, self.data, motion, self.frame)
        self.cam.lookat[:] = root_pos
        self._apply_camera_mode(motion)
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.scene_option)
        rgb = self.renderer.render()

        path = self.files[self.idx]
        kind = clip_kind(path.stem)
        header = f"[{self.idx + 1}/{len(self.files)}] {path.stem}"
        if kind:
            header += f"  |  {kind}"
        header += f"  |  {self.speeds[self.idx]:.2f} m/s"

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz_folder",
        default=str(DEFAULT_NPZ_FOLDER),
        help="Folder containing AMP NPZ files (default: output/amp)",
    )
    parser.add_argument("--pattern", default="*.npz", help="File pattern to match")
    parser.add_argument("--loop_each", action="store_true", help="Loop each clip until you switch")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = full clip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_files = list(iter_npz_files(args.npz_folder, args.pattern))
    if not npz_files:
        print(f"No NPZ files found in {args.npz_folder}")
        return

    print(
        f"Found {len(npz_files)} clips under {args.npz_folder}. "
        "Controls: Space pause  Left/Right clip  Up/Down frame  R restart  Q quit  |  "
        "Camera: drag orbit  wheel zoom  [ ] ,. tilt  o auto  f follow  c reset"
    )
    App(npz_files=npz_files, loop_each=args.loop_each, max_frames=args.max_frames).run()
    print("\nDone!")


if __name__ == "__main__":
    main()
