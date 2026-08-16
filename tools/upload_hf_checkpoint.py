#!/usr/bin/env python3
"""Upload a completed local training checkpoint to a Hugging Face dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-in-repo", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint_dir}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=checkpoint_dir,
        path_in_repo=args.path_in_repo,
        commit_message=f"Upload checkpoint {args.path_in_repo.rsplit('/', 1)[-1]}",
    )


if __name__ == "__main__":
    main()
