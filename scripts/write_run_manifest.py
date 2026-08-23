#!/usr/bin/env python3
"""Write run manifest for CloudEdge training run."""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=str, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project_root).decode().strip()

    manifest = {
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "timestamp_start": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit,
        "bootstrap_model": args.bootstrap,
        "base_model": "artifacts/pi05_libero_base_official_aligned",
        "steps": 120000,
        "global_batch_size": 8,
        "per_gpu_batch_size": 2,
        "num_gpus": 4,
        "seed": 7,
        "precision": "bfloat16",
        "learning_rate": 2.5e-5,
        "lr_warmup_steps": 1000,
        "lr_decay_steps": 30000,
        "lr_decay_floor": 2.5e-6,
        "history_window": 21,
        "stale_delay_distribution": "Uniform(1..20)",
        "stale_loss_weight_max": 0.5,
        "stale_loss_warmup_steps": 10000,
        "save_interval": 5000,
        "wandb": {
            "enable": True,
            "project": "cloudedge_pi05_libero_aligned"
        }
    }
    manifest_path = args.run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"RUN_MANIFEST_WRITTEN: {manifest_path}")


if __name__ == "__main__":
    main()
