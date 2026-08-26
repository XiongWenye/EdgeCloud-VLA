#!/usr/bin/env python3
"""Aggregate evaluation results across all 4 LIBERO suites for Aligned LeRobot pi05."""

import argparse
import json
import pathlib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=pathlib.Path("/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/results/aligned_lerobot"),
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=pathlib.Path("/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/results/aligned_lerobot_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=pathlib.Path,
        default=pathlib.Path("/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/results/aligned_lerobot_summary.md"),
    )
    args = parser.parse_args()

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suite_names = {
        "libero_spatial": "LIBERO-Spatial",
        "libero_object": "LIBERO-Object",
        "libero_goal": "LIBERO-Goal",
        "libero_10": "LIBERO-10 (Long Horizon)",
    }

    published = {
        "libero_spatial": 98.8,
        "libero_object": 98.2,
        "libero_goal": 98.0,
        "libero_10": 92.4,
    }
    openpi_cluster = {
        "libero_spatial": 98.4,
        "libero_object": 99.0,
        "libero_goal": 97.8,
        "libero_10": 93.4,
    }
    unaligned_lerobot = {
        "libero_spatial": 95.2,
        "libero_object": 96.0,
        "libero_goal": 94.4,
        "libero_10": 84.4,
    }

    results = {}
    total_episodes = 0
    total_successes = 0

    for suite in suites:
        eval_info_file = args.results_dir / suite / "eval_info.json"
        if eval_info_file.exists():
            data = json.loads(eval_info_file.read_text())
            overall = data.get("overall", {})
            per_group = data.get("per_group", {}).get(suite, {})
            n_episodes = overall.get("n_episodes", per_group.get("n_episodes", 0))
            pc_success = overall.get("pc_success", per_group.get("pc_success", 0.0))
            per_task = data.get("per_task", [])
            task_successes = []
            for t in per_task:
                succs = t.get("metrics", {}).get("successes", [])
                task_successes.append(float(np.mean(succs) * 100.0) if succs else 0.0)

            results[suite] = {
                "name": suite_names[suite],
                "n_episodes": n_episodes,
                "success_rate": float(pc_success),
                "success_count": int(round((pc_success / 100.0) * n_episodes)),
                "per_task_success": task_successes,
                "status": "COMPLETED",
            }
            total_episodes += n_episodes
            total_successes += int(round((pc_success / 100.0) * n_episodes))
        else:
            results[suite] = {
                "name": suite_names[suite],
                "n_episodes": 0,
                "success_rate": 0.0,
                "success_count": 0,
                "per_task_success": [],
                "status": "NOT_FOUND",
            }

    completed_suites = [s for s in suites if results[s]["status"] == "COMPLETED"]
    macro_avg = float(np.mean([results[s]["success_rate"] for s in completed_suites])) if completed_suites else 0.0

    summary = {
        "suites": results,
        "completed_suites": len(completed_suites),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "aligned_lerobot_macro_avg": macro_avg,
        "published_macro_avg": 96.85,
        "openpi_cluster_macro_avg": 97.15,
        "unaligned_lerobot_macro_avg": 92.50,
        "gain_over_unaligned": macro_avg - 92.50,
    }

    # Generate Markdown Table
    md = [
        "# Aligned LeRobot pi05_libero Evaluation Summary",
        "",
        "Full 4-suite closed-loop evaluation on Harvard Kempner H200 SLURM cluster (50 episodes/task $\times$ 10 tasks/suite = 2,000 episodes total, seed 7, wait steps 10, chunk 10, replan 5).",
        "",
        "| Benchmark Suite | Published OpenPI | Cluster OpenPI Ref | Unaligned LeRobot | **Aligned LeRobot (Ours)** | Successes | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in suites:
        r = results[s]
        pub = f"{published[s]:.1f}%"
        ref = f"{openpi_cluster[s]:.1f}%"
        unal = f"{unaligned_lerobot[s]:.1f}%"
        if r["status"] == "COMPLETED":
            aln = f"**{r['success_rate']:.2f}%**"
            succ_str = f"{r['success_count']}/{r['n_episodes']}"
            stat = "✅ Completed"
        else:
            aln = "*Running...*"
            succ_str = "-"
            stat = "⏳ Running"
        md.append(f"| {suite_names[s]} | {pub} | {ref} | {unal} | {aln} | {succ_str} | {stat} |")

    pub_macro = "96.85%"
    ref_macro = "97.15%"
    unal_macro = "92.50%"
    aln_macro = f"**{macro_avg:.2f}%**"
    total_succ_str = f"**{total_successes}/{total_episodes}**"
    md.append(f"| **Macro Average** | **{pub_macro}** | **{ref_macro}** | **{unal_macro}** | {aln_macro} | {total_succ_str} | **All 4 Suites** |")
    md.append("")

    # Add Per-Task Breakdown
    md.extend([
        "## Per-Task Success Breakdown",
        "",
    ])
    for s in suites:
        r = results[s]
        if r["status"] == "COMPLETED":
            md.append(f"### {suite_names[s]} ({r['success_rate']:.2f}%)")
            md.append("")
            md.append("| Task ID | Task Success Rate | Episodes |")
            md.append("|---|---|---|")
            for tid, acc in enumerate(r["per_task_success"]):
                md.append(f"| Task {tid} | {acc:.1f}% | 50 |")
            md.append("")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + chr(10))
    args.output_md.write_text(chr(10).join(md) + chr(10))
    print(f"Summary written to {args.output_json} and {args.output_md}")
    print(chr(10).join(md))


if __name__ == "__main__":
    main()
