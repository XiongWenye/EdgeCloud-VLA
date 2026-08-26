#!/usr/bin/env python3
"""Static validation of repository files, python syntax, configuration manifest, and sbatch headers."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    # 1. Check Python syntax
    python_files = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "scripts").glob("*.py"))
        + sorted((ROOT / "tests").glob("*.py"))
    )
    for path in python_files:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in {path}: {e}")

    # 2. Check manifest
    manifest_path = ROOT / "configs" / "experiment.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["training"]["history_window"] == 21
    assert manifest["training"]["steps"] == 120000
    assert manifest["training"]["global_batch_size"] == 8
    assert manifest["protocol"]["chunk_size"] == 10
    assert manifest["protocol"]["n_action_steps"] == 5
    assert manifest["protocol"]["num_inference_steps"] == 10
    assert manifest["evaluation"]["core_delay_max_steps"] == [0, 10, 20, 40]
    assert manifest["evaluation"]["cloudedge_seeds"] == [7, 8, 9]
    assert manifest["evaluation"]["trials_per_task"] == 50
    assert manifest["artifacts"]["base_model"] == "artifacts/pi05_libero_base_official_aligned"

    # 3. Check patch file
    patch_file = ROOT / "patches" / "0001-align-lerobot-pi05-with-openpi-libero.patch"
    assert patch_file.exists(), f"Missing alignment patch file {patch_file}"

    # 4. Check sbatch files
    sbatch_files = sorted((ROOT / "slurm").glob("*.sbatch"))
    assert sbatch_files, "No sbatch files found in slurm/"
    for path in sbatch_files:
        text = path.read_text()
        assert "#SBATCH --partition=kempner" in text, f"Missing partition in {path}"
        assert "#SBATCH --account=kempner_ydu_lab" in text, f"Missing account in {path}"
        assert "set -euo pipefail" in text, f"Missing strict mode in {path}"

    print(
        f"STATIC_VALIDATE_OK: python_files={len(python_files)} sbatch_files={len(sbatch_files)} manifest=valid"
    )


if __name__ == "__main__":
    main()
