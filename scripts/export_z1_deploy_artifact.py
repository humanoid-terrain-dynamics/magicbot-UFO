#!/usr/bin/env python3
"""Stage and export a deployable Z1 UFO model artifact.

The input checkpoint remains immutable.  Large source files are hard-linked into
the deploy bundle when possible, otherwise copied, so every runtime input lives
under UFO_deploy after this command completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) == sha256(destination):
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            link_or_copy(path, destination / path.relative_to(source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the self-contained Z1 ONNX deploy artifact.")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2] / "UFO")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "checkpoints/Z1_UFO/z1_mimic_clean_full_20_fb_1xa100_gpu1_1024env_nowandb_20260727_0110",
    )
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    bundle_root = args.bundle_root.resolve()
    if not (source_root / "humanoidverse").is_dir():
        raise FileNotFoundError(f"Not a UFO source root: {source_root}")

    source_model = checkpoint_root / "checkpoint/model/model_best_mpjpe_l_963p743_global48009216.safetensors"
    if not source_model.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {source_model}")

    model_dir = bundle_root / "model/z1_policy"
    checkpoint_dir = model_dir / "checkpoint/model"
    exported_dir = model_dir / "exported"
    latent_dir = model_dir / "tracking_inference"
    robot_dir = bundle_root / "robot"
    for source, target in (
        (checkpoint_root / "config.json", checkpoint_dir / "config.json"),
        (checkpoint_root / "init_kwargs.json", checkpoint_dir / "init_kwargs.json"),
        (source_model, checkpoint_dir / "model.safetensors"),
    ):
        link_or_copy(source, target)
    for latent in sorted((checkpoint_root / "tracking_inference").glob("*.pkl")):
        link_or_copy(latent, latent_dir / latent.name)

    robot_yaml = source_root / "configs/robots/z1_23dof.yaml"
    # This is the training-config asset: 23 named actuators, no head actuator.
    # The nearby z1/mjcf variant contains an additional head actuator (nu=24).
    robot_xml = source_root / "humanoidverse/data/robots/magicbot_z1_description/mjcf/MAGICBOTZ1.xml"
    if not robot_xml.is_file():
        raise FileNotFoundError(robot_xml)
    link_or_copy(robot_yaml, robot_dir / "z1_23dof.yaml")
    link_or_copy(robot_xml, robot_dir / "mjcf/MAGICBOTZ1.xml")
    copy_tree(robot_xml.parent.parent / "meshes", robot_dir / "meshes")

    # The deployed runner uses this normalized, resolved control contract.
    robot_cfg = yaml.safe_load(robot_yaml.read_text(encoding="utf-8"))
    control_names = list(robot_cfg["control_joints"]["names"])
    control = robot_cfg["training"]["control"]
    control_contract = {
        "control_joint_names": control_names,
        "default_dof_pos": [robot_cfg["default_dof_pos"][name] for name in control_names],
        "stiffness": [control["stiffness"][name] for name in control_names],
        "damping": [control["damping"][name] for name in control_names],
        "effort_limit": list(control["effort_limit"]),
        "action_scale": control["action_scale"],
        "normalize_action_to": control["normalize_action_to"],
        "root_quat_order": robot_cfg["root_quat_order"],
        "sim_dt": 0.002,
        "control_hz": 50,
    }
    contract_path = model_dir / "z1_control_contract.json"
    contract_path.write_text(json.dumps(control_contract, indent=2) + "\n", encoding="utf-8")

    sys.path.insert(0, str(source_root))
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.export.backward_encoder import export_backward_encoder_from_model
    from humanoidverse.utils.helpers import export_meta_policy_as_onnx

    model = load_model_from_checkpoint_dir(model_dir / "checkpoint", device=args.device)
    model.eval()
    export_spec = export_meta_policy_as_onnx(model, exported_dir, "policy.onnx", z_dim=int(model.cfg.archi.z_dim))
    # Z1's backward map uses state and privileged_state only. ONNX correctly
    # eliminates last_action; the shared export verifier currently assumes that
    # optional input is retained, so validate the retained inputs below instead.
    export_backward_encoder_from_model(model, exported_dir / "backward_encoder.onnx", verify=False)
    if export_spec["actor_obs_dim"] != 631 or export_spec["output_action_dim"] != 23:
        raise ValueError(f"Unexpected Z1 actor interface: {export_spec}")
    import onnxruntime as ort

    backward_inputs = {item.name for item in ort.InferenceSession(str(exported_dir / "backward_encoder.onnx")).get_inputs()}
    if not {"state", "privileged_state"}.issubset(backward_inputs):
        raise ValueError(f"Unexpected Z1 backward encoder inputs: {sorted(backward_inputs)}")
    export_spec.update({"robot_xml": "robot/mjcf/MAGICBOTZ1.xml", "control_contract": "model/z1_policy/z1_control_contract.json"})
    (exported_dir / "policy.meta.json").write_text(json.dumps(export_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    required = [
        model_dir / "checkpoint/model/model.safetensors",
        exported_dir / "policy.onnx",
        exported_dir / "backward_encoder.onnx",
        exported_dir / "policy.meta.json",
        contract_path,
        robot_dir / "mjcf/MAGICBOTZ1.xml",
    ]
    manifest = {
        "format_version": 1,
        "robot": "z1_23dof",
        "checkpoint_source": str(checkpoint_root),
        "files": {str(path.relative_to(bundle_root)).replace("\\", "/"): sha256(path) for path in required},
        "tracking_latents": sorted(path.name for path in latent_dir.glob("*.pkl")),
    }
    (model_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Z1 deploy artifact exported: {model_dir}")


if __name__ == "__main__":
    main()
