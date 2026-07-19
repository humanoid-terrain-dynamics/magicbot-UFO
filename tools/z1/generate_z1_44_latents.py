"""Generate official UFO z latents for the 44 curated Z1 training motions.

This wraps ``humanoidverse/tracking_inference.py`` with ``--max-steps 0``. The
tracking script saves the backward-encoder z before the policy rollout, so this
builds a deploy-friendly latent cache without rendering videos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib


GROUP_FILES = [
    "z1_cyclic_locomotion_train_near10s_ufo.pkl",
    "z1_atomic_skills_train_near10s_ufo.pkl",
    "z1_acrobatics_recovery_train_near10s_ufo.pkl",
    "z1_pose_low_motion_train_near10s_ufo.pkl",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_dir(project_root: Path) -> Path:
    return project_root.parent / "checkpoints" / "Z1_UFO" / "compare_hand_vs_nohand" / "fakehand_32M"


def _checkpoint_model_folder(project_root: Path) -> Path:
    return project_root.parent / "checkpoints" / "Z1_UFO" / "z1_mimic_clean_fb_2xa100_1024env_20260716_044447"


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=_checkpoint_model_folder(project_root))
    parser.add_argument("--artifact-dir", type=Path, default=_artifact_dir(project_root))
    parser.add_argument("--cache-root", type=Path, default=project_root / "cache" / "motion_data" / "z1_mimic_clean")
    parser.add_argument("--robot-config", type=Path, default=project_root / "configs" / "robots" / "z1_23dof.yaml")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    out_dir = (args.artifact_dir / "z44_latents").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = args.model_folder / "tracking_inference"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    global_index = 0
    python = sys.executable
    tracking_script = project_root / "humanoidverse" / "tracking_inference.py"

    for group_name in GROUP_FILES:
        group_path = (args.cache_root / group_name).resolve()
        motions = joblib.load(group_path)
        keys = list(motions.keys())
        motion_ids = [str(i) for i in range(len(keys))]
        print(f"[z44] generating group={group_name} count={len(keys)}", flush=True)

        cmd = [
            python,
            str(tracking_script),
            "--model-folder",
            str(args.model_folder.resolve()),
            "--data-path",
            str(group_path),
            "--robot-config",
            str(args.robot_config.resolve()),
            "--device",
            str(args.device),
            "--save-mp4",
            "false",
            "--headless",
            "true",
            "--disable-dr",
            "true",
            "--disable-obs-noise",
            "true",
            "--motion-list",
            *motion_ids,
            "--export-onnx",
            "false",
            "--log-every-steps",
            "0",
            "--max-steps",
            "0",
        ]
        env = os.environ.copy()
        env["MUJOCO_GL"] = "glfw"
        subprocess.run(cmd, cwd=str(project_root), env=env, check=True)

        for local_index, key in enumerate(keys):
            source = tracking_dir / f"zs_{local_index}.pkl"
            if not source.exists():
                raise FileNotFoundError(f"tracking_inference did not produce {source}")
            z_name = f"{global_index:02d}_{_safe_name(key)}_zs.pkl"
            dest = out_dir / z_name
            shutil.copy2(source, dest)
            manifest.append(
                {
                    "index": global_index,
                    "group": group_name,
                    "group_index": local_index,
                    "motion_key": key,
                    "z_file": z_name,
                    "source_motion_file": str(group_path),
                }
            )
            global_index += 1

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[z44] wrote {len(manifest)} latents to {out_dir}", flush=True)
    print(f"[z44] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
