#!/usr/bin/env python3
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    python_files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))

    manifest = json.loads((ROOT / "configs" / "experiment.json").read_text())
    assert manifest["training"]["history_window"] == 21
    assert manifest["training"]["steps"] == 120000
    assert manifest["evaluation"]["delay_max_steps"] == [0, 10, 20, 40]
    assert manifest["evaluation"]["cloudedge_seeds"] == [7, 8, 9]
    assert manifest["evaluation"]["trials_per_task"] == 50

    sbatch_files = sorted((ROOT / "slurm").glob("*.sbatch"))
    assert sbatch_files
    for path in sbatch_files:
        text = path.read_text()
        assert "#SBATCH --partition=kempner" in text, path
        assert "#SBATCH --account=kempner_ydu_lab" in text, path
        assert "set -euo pipefail" in text, path

    print(f"python_files={len(python_files)} sbatch_files={len(sbatch_files)} manifest=valid")


if __name__ == "__main__":
    main()

