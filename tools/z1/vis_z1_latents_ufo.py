#!/usr/bin/env python3
"""Tkinter viewer for Z1 UFO latent-z trajectories (``zs_*.pkl``).

Each ``zs_i.pkl`` written by ``humanoidverse.tracking_inference`` is a
``(T, 256)`` float32 array = the per-timestep FB skill latent for one motion.
These are NOT poses, so this viewer does not render a robot — it visualizes the
latent itself:

  * Heatmap (256 latent dims x T timesteps), blue<->red diverging, with an
    animated playhead sweeping timesteps and a temporal-energy strip
    (mean |z| over dims).
  * Per-clip vs global normalization toggle.
  * PCA-2D trajectory toggle: each clip's T points projected onto the top-2
    PCs (fit globally over all clips), current frame highlighted.

Controls:
  Left / Right  previous / next latent
  Space         pause / resume playhead
  R             restart playhead at frame 0
  N             toggle per-clip / global normalization
  P             toggle heatmap / PCA-2D view
  Q / Esc       quit

Usage:
  python tools/z1/vis_z1_latents_ufo.py
  python tools/z1/vis_z1_latents_ufo.py --folder <dir-with-zs_*.pkl>
  python tools/z1/vis_z1_latents_ufo.py --pkl path/to/zs_0.pkl path/to/zs_1.pkl
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDER = (
    REPO_ROOT.parent
    / "checkpoints"
    / "Z1_UFO"
    / "z1_mimic_clean_fb_2xa100_1024env_20260716_044447"
    / "tracking_inference"
)

RENDER_WIDTH = 1120
RENDER_HEIGHT = 720
LIST_WIDTH = 34
ENERGY_STRIP_H = 70   # top band for the temporal-energy line plot
PLAYHEAD_MS = 33      # ~30 fps playhead advance


@dataclass
class Latent:
    path: Path
    index: int
    z: np.ndarray  # (T, 256) float32

    @property
    def num_frames(self) -> int:
        return int(self.z.shape[0])

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.z ** 2)))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def iter_pkl_files(folder: str | Path, pattern: str = "zs_*.pkl"):
    folder = Path(folder)
    if folder.is_file():
        yield folder
        return
    yield from sorted(folder.rglob(pattern), key=lambda p: p.name)


def load_latents(files: list[Path]) -> list[Latent]:
    out: list[Latent] = []
    for f in files:
        try:
            arr = np.asarray(joblib.load(f), dtype=np.float32)
        except Exception as exc:
            print(f"[skip] {f.name}: {exc}", flush=True)
            continue
        if arr.ndim != 2:
            print(f"[skip] {f.name}: expected 2-D (T, z_dim), got shape {arr.shape}", flush=True)
            continue
        idx = _index_from_name(f.stem)
        out.append(Latent(f, idx, arr))
    out.sort(key=lambda lat: lat.index)
    return out


def _index_from_name(stem: str) -> int:
    tail = stem.split("_")[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


def build_diverging_lut() -> np.ndarray:
    """256-entry RGB LUT: blue (negative) -> dark (zero) -> red (positive)."""
    stops = [(0.0, (45, 70, 180)), (0.5, (12, 12, 28)), (1.0, (215, 85, 45))]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(len(stops) - 1):
            t0, c0 = stops[k]
            t1, c1 = stops[k + 1]
            if t0 <= t <= t1:
                a = (t - t0) / max(t1 - t0, 1e-9)
                lut[i] = [int(c0[j] + (c1[j] - c0[j]) * a) for j in range(3)]
                break
    return lut


class App:
    def __init__(self, latents: list[Latent]) -> None:
        if not latents:
            raise SystemExit("No zs_*.pkl latents loaded.")
        self.latents = latents
        self.lut = build_diverging_lut()

        # Global stats + PCA (fit over all clips pooled).
        all_rows = np.concatenate([lat.z for lat in latents], axis=0)
        self.g_min = float(all_rows.min())
        self.g_max = float(all_rows.max())
        self.g_mu = all_rows.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(all_rows - self.g_mu, full_matrices=False)
            self.pc = vt[:2]                      # (2, 256)
            self.pc_proj = [(lat.z - self.g_mu) @ self.pc.T for lat in latents]  # each (T, 2)
            self.pc_range = self._pc_extent()
        except Exception:
            self.pc = None
            self.pc_proj = None

        self.idx = 0
        self.frame = 0
        self.paused = False
        self.norm_global = False
        self.view_pca = False
        self._running = True
        self._photo: ImageTk.PhotoImage | None = None
        self._base_cache: dict[tuple, Image.Image] = {}

        self.header_font = load_font(22)
        self.status_font = load_font(16)
        self.small_font = load_font(13)

        self.root = tk.Tk()
        self.root.title("Z1 UFO Latent-z Viewer")
        self.root.geometry("1500x820")
        self.root.configure(bg="#131722")
        self._build_layout()
        self._bind_keys()
        self._select_index(0, reset_frame=True)
        self._schedule()

    # ---- PCA helpers --------------------------------------------------------
    def _pc_extent(self) -> tuple[float, float, float, float]:
        pts = np.concatenate(self.pc_proj, axis=0)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    # ---- UI -----------------------------------------------------------------
    def _build_layout(self) -> None:
        container = tk.Frame(self.root, bg="#131722")
        container.pack(fill="both", expand=True)
        sidebar = tk.Frame(container, bg="#0f1320", width=360)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        viewer = tk.Frame(container, bg="#131722")
        viewer.pack(side="right", fill="both", expand=True)

        tk.Label(
            sidebar, text="Latent-z Clips", bg="#0f1320", fg="#eef3ff",
            font=("Segoe UI", 16, "bold"), anchor="w", padx=16, pady=12,
        ).pack(fill="x")
        tk.Label(
            sidebar, textvariable=tk.StringVar(value=str(self.latents[0].path.parent)),
            bg="#0f1320", fg="#7f8ca8", font=("Consolas", 9), anchor="w",
            justify="left", wraplength=340, padx=16,
        ).pack(fill="x")

        self.summary_var = tk.StringVar()
        tk.Label(
            sidebar, textvariable=self.summary_var, bg="#0f1320", fg="#b5bfd6",
            font=("Segoe UI", 10), anchor="w", justify="left", padx=16, pady=8,
        ).pack(fill="x")

        list_frame = tk.Frame(sidebar, bg="#0f1320")
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        sb = tk.Scrollbar(list_frame, orient="vertical")
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(
            list_frame, width=LIST_WIDTH, bg="#171d2d", fg="#dbe2f2",
            selectbackground="#2b6cff", selectforeground="#ffffff", activestyle="none",
            font=("Consolas", 10), yscrollcommand=sb.set,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.listbox.yview)
        for lat in self.latents:
            self.listbox.insert("end", f"zs_{lat.index:<2d}  T={lat.num_frames:<4d}  rms={lat.rms:.2f}")
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.info_var = tk.StringVar()
        tk.Label(
            sidebar, textvariable=self.info_var, bg="#0f1320", fg="#dbe2f2",
            font=("Consolas", 10), anchor="w", justify="left", padx=16, pady=8,
        ).pack(fill="x")

        tk.Label(
            sidebar,
            text=(
                "Left/Right prev/next latent\nSpace pause/resume playhead\n"
                "R restart  N norm(per/global)  P view(heat/PCA)\nQ or Esc quit"
            ),
            bg="#0f1320", fg="#7f8ca8", font=("Segoe UI", 9), anchor="w",
            justify="left", padx=16, pady=12,
        ).pack(fill="x")

        self.image_label = tk.Label(viewer, bg="#131722")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=12)

        total = sum(lat.num_frames for lat in self.latents)
        self.summary_var.set(f"{len(self.latents)} latents  |  {total} total frames  |  z_dim={self.latents[0].z.shape[1]}")

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _: self._select_index(self.idx - 1, reset_frame=True))
        self.root.bind("<Right>", lambda _: self._select_index(self.idx + 1, reset_frame=True))
        self.root.bind("<space>", lambda _: self.toggle_pause())
        self.root.bind("r", lambda _: self._restart())
        self.root.bind("n", lambda _: self._toggle_norm())
        self.root.bind("p", lambda _: self._toggle_view())
        self.root.bind("q", lambda _: self.quit())
        self.root.bind("<Escape>", lambda _: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _on_listbox_select(self, _e) -> None:
        sel = self.listbox.curselection()
        if sel:
            self._select_index(sel[0], reset_frame=True)

    def _select_index(self, idx: int, reset_frame: bool) -> None:
        self.idx = idx % len(self.latents)
        if reset_frame:
            self.frame = 0
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.idx)
        self.listbox.activate(self.idx)
        self.listbox.see(self.idx)

    def _restart(self) -> None:
        self.frame = 0

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def _toggle_norm(self) -> None:
        self.norm_global = not self.norm_global
        self._base_cache.clear()

    def _toggle_view(self) -> None:
        self.view_pca = not self.view_pca
        self._base_cache.clear()

    def quit(self) -> None:
        self._running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---- rendering ----------------------------------------------------------
    def _norm_bounds(self, lat: Latent) -> tuple[float, float]:
        if self.norm_global:
            return self.g_min, self.g_max
        return float(lat.z.min()), float(lat.z.max())

    def _heatmap_base(self, lat: Latent) -> Image.Image:
        vmin, vmax = self._norm_bounds(lat)
        a = lat.z.T  # (z_dim, T) -> rows=dims, cols=timesteps
        norm = np.clip((a - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
        rgb = self.lut[(norm * 255).astype(np.uint8)]  # (z_dim, T, 3)
        img = Image.fromarray(rgb, "RGB").resize((RENDER_WIDTH, RENDER_HEIGHT - ENERGY_STRIP_H), Image.NEAREST)
        # top energy strip (dark), overwritten with the line plot
        strip = Image.new("RGB", (RENDER_WIDTH, ENERGY_STRIP_H), (12, 12, 28))
        out = Image.new("RGB", (RENDER_WIDTH, RENDER_HEIGHT), (12, 12, 28))
        out.paste(strip, (0, 0))
        out.paste(img, (0, ENERGY_STRIP_H))
        return out

    def _draw_energy(self, base: Image.Image, lat: Latent) -> None:
        energy = np.sqrt(np.mean(lat.z ** 2, axis=1))  # (T,)
        T = lat.num_frames
        x0, y0 = 8, 6
        x1, y1 = RENDER_WIDTH - 8, ENERGY_STRIP_H - 6
        emin, emax = float(energy.min()), float(energy.max())
        rng = max(emax - emin, 1e-8)
        pts = [
            (
                x0 + (t / max(T - 1, 1)) * (x1 - x0),
                y1 - ((energy[t] - emin) / rng) * (y1 - y0),
            )
            for t in range(T)
        ]
        draw = ImageDraw.Draw(base)
        draw.line(pts, fill=(120, 200, 255), width=2)
        draw.text((x0, y0 - 2), f"mean|z|  [{emin:.2f}, {emax:.2f}]", fill=(170, 190, 220), font=self.small_font)

    def _pca_base(self, lat: Latent) -> Image.Image:
        img = Image.new("RGB", (RENDER_WIDTH, RENDER_HEIGHT), (12, 12, 28))
        if self.pc is None or self.pc_proj is None:
            return img
        draw = ImageDraw.Draw(img)
        lox, hix, loy, hiy = self.pc_range
        padx = 60
        pady = 60
        def to_px(p):
            x = padx + (p[0] - lox) / max(hix - lox, 1e-8) * (RENDER_WIDTH - 2 * padx)
            y = pady + (hiy - p[1]) / max(hiy - loy, 1e-8) * (RENDER_HEIGHT - 2 * pady)
            return (float(x), float(y))
        # other clips faint
        for j, proj in enumerate(self.pc_proj):
            if j == self.idx:
                continue
            pts = [to_px(proj[t]) for t in range(len(proj))]
            draw.line(pts, fill=(50, 60, 90), width=1)
        # current clip bold, colored along time
        proj = self.pc_proj[self.idx]
        T = len(proj)
        for t in range(1, T):
            c = int(80 + 175 * t / max(T - 1, 1))
            draw.line([to_px(proj[t - 1]), to_px(proj[t])], fill=(c, 200 - c // 2, 255 - c), width=2)
        draw.text((padx, 16), "PCA-2D trajectory (PC1 vs PC2, global fit) — faint = other latents", fill=(170, 190, 220), font=self.small_font)
        return img

    def _base(self, lat: Latent) -> Image.Image:
        key = (self.idx, self.view_pca, self.norm_global)
        cached = self._base_cache.get(key)
        if cached is None:
            cached = self._pca_base(lat) if self.view_pca else self._heatmap_base(lat)
            self._base_cache[key] = cached
        return cached

    def render_current(self) -> None:
        lat = self.latents[self.idx]
        T = lat.num_frames
        self.frame = min(self.frame, T - 1)
        base = self._base(lat).copy()
        if not self.view_pca:
            self._draw_energy(base, lat)

        draw = ImageDraw.Draw(base)
        # playhead
        if not self.view_pca:
            x = 0 + (self.frame / max(T - 1, 1)) * (RENDER_WIDTH - 1)
            draw.rectangle([int(x) - 1, ENERGY_STRIP_H, int(x) + 1, RENDER_HEIGHT], fill=(255, 235, 80))
        else:
            proj = self.pc_proj[self.idx]
            px, py = proj[self.frame]
            lox, hix, loy, hiy = self.pc_range
            padx = pady = 60
            cx = padx + (px - lox) / max(hix - lox, 1e-8) * (RENDER_WIDTH - 2 * padx)
            cy = pady + (hiy - py) / max(hiy - loy, 1e-8) * (RENDER_HEIGHT - 2 * pady)
            draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 235, 80), width=3)
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 235, 80))

        vmin, vmax = self._norm_bounds(lat)
        view = "PCA" if self.view_pca else "HEAT"
        norm = "global" if self.norm_global else "per-clip"
        header = f"zs_{lat.index}  |  T={T}  z_dim={lat.z.shape[1]}  rms={lat.rms:.2f}  range=[{vmin:.2f},{vmax:.2f}]"
        status = f"frame {self.frame + 1}/{T}  |  {'PAUSED' if self.paused else 'PLAY'}  |  view={view}  norm={norm}  |  N norm  P view  R restart  Q quit"
        print("\r" + status[:140], end="", flush=True)

        hb = draw.textbbox((0, 0), header, font=self.header_font)
        draw.rectangle([8, RENDER_HEIGHT - 56, 16 + hb[2], RENDER_HEIGHT - 36], fill=(0, 0, 0))
        draw.text((12, RENDER_HEIGHT - 54), header, fill=(255, 235, 80), font=self.header_font)
        sb = draw.textbbox((0, 0), status, font=self.status_font)
        draw.rectangle([8, RENDER_HEIGHT - 30, 16 + sb[2], RENDER_HEIGHT - 12], fill=(0, 0, 0))
        draw.text((12, RENDER_HEIGHT - 28), status, fill=(210, 255, 210), font=self.status_font)

        self._photo = ImageTk.PhotoImage(base)
        self.image_label.config(image=self._photo)
        self.info_var.set(
            "\n".join(
                [
                    f"[{self.idx + 1}/{len(self.latents)}] {lat.path.name}",
                    f"shape : {tuple(lat.z.shape)}  dtype={lat.z.dtype}",
                    f"frames: {T}   rms={lat.rms:.3f}",
                    f"min/max: {float(lat.z.min()):.2f} / {float(lat.z.max()):.2f}",
                    f"view={view}  norm={norm}  {'PAUSED' if self.paused else 'PLAY'}",
                ]
            )
        )

        if not self.paused:
            self.frame = (self.frame + 1) % T

    def _schedule(self) -> None:
        if not self._running:
            return
        try:
            self.render_current()
        except Exception as exc:
            print(f"\nrender error: {exc}")
            self.quit()
            return
        self.root.after(PLAYHEAD_MS, self._schedule)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default=str(DEFAULT_FOLDER), help="Folder of zs_*.pkl latents.")
    parser.add_argument("--pattern", default="zs_*.pkl", help="File pattern under --folder.")
    parser.add_argument("--pkl", nargs="+", type=Path, default=None, help="Explicit zs_*.pkl file(s); overrides --folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.pkl if args.pkl else list(iter_pkl_files(args.folder, args.pattern))
    if not files:
        print(f"No zs_*.pkl found (folder={args.folder}, pattern={args.pattern}). Pass --pkl FILE [...].")
        return
    latents = load_latents(files)
    print(
        f"Loaded {len(latents)} latent(s) from {len(files)} file(s). "
        "Left/Right switch  Space pause  N norm  P view  R restart  Q quit",
        flush=True,
    )
    App(latents).run()
    print("\nDone!")


if __name__ == "__main__":
    main()
