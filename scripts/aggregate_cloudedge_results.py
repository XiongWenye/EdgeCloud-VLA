#!/usr/bin/env python3
"""Aggregate and validate CloudEdge pi05 evaluation matrix across suites, delays, and seeds."""

import argparse
import json
from pathlib import Path
import numpy as np


def compute_auc(delays, values):
    """Compute normalized trapezoidal AUC over delay range [0, max(delays)]."""
    if len(delays) < 2:
        return 0.0
    sorted_pairs = sorted(zip(delays, values))
    d_sorted = [p[0] for p in sorted_pairs]
    v_sorted = [p[1] for p in sorted_pairs]
    total_area = 0.0
    for i in range(len(d_sorted) - 1):
        h = d_sorted[i + 1] - d_sorted[i]
        avg_val = (v_sorted[i] + v_sorted[i + 1]) / 2.0
        total_area += h * avg_val
    max_d = d_sorted[-1] - d_sorted[0]
    return total_area / max_d if max_d > 0 else 0.0


def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ce-dir",
        type=Path,
        default=ROOT / "results/cloudedge_aligned/raw",
        help="Path to CloudEdge raw results",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=ROOT / "results/baseline_aligned_delayed/raw",
        help="Path to Delayed Baseline raw results",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results/cloudedge_aligned_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "results/cloudedge_aligned_summary.md",
    )
    parser.add_argument(
        "--full-curve",
        action="store_true",
        help="Include full 8-point curve delays [0, 5, 10, 15, 20, 25, 30, 40]",
    )
    args = parser.parse_args()

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suite_names = {
        "libero_spatial": "LIBERO-Spatial",
        "libero_object": "LIBERO-Object",
        "libero_goal": "LIBERO-Goal",
        "libero_10": "LIBERO-10",
    }
    delays = [0, 5, 10, 15, 20, 25, 30, 40] if args.full_curve else [0, 10, 20, 40]
    ce_seeds = [7, 8, 9]

    # Load baseline delayed records
    baseline_records = {}  # (suite, d_max) -> float
    baseline_counts = {}  # (suite, d_max) -> (success, total)
    for suite in suites:
        for d in delays:
            p = args.base_dir / f"dmax_{d}" / suite / "seed_7" / "eval_info.json"
            if p.exists():
                data = json.loads(p.read_text())
                ov = data.get("overall", {})
                succ_rate = ov.get("pc_success", 0.0)
                n_ep = ov.get("n_episodes", 0)
                baseline_records[(suite, d)] = float(succ_rate)
                baseline_counts[(suite, d)] = (int(round(succ_rate * n_ep / 100.0)), n_ep)
            else:
                baseline_records[(suite, d)] = None

    # Load CloudEdge records
    ce_records = {}  # (suite, d_max, seed) -> float
    ce_counts = {}  # (suite, d_max, seed) -> (success, total)
    for suite in suites:
        for d in delays:
            for s in ce_seeds:
                p = args.ce_dir / f"dmax_{d}" / suite / f"seed_{s}" / "eval_info.json"
                if p.exists():
                    data = json.loads(p.read_text())
                    ov = data.get("overall", {})
                    succ_rate = ov.get("pc_success", 0.0)
                    n_ep = ov.get("n_episodes", 0)
                    ce_records[(suite, d, s)] = float(succ_rate)
                    ce_counts[(suite, d, s)] = (int(round(succ_rate * n_ep / 100.0)), n_ep)
                else:
                    ce_records[(suite, d, s)] = None

    # Compute stats
    ce_mean = {}
    ce_std = {}
    for suite in suites:
        for d in delays:
            vals = [ce_records[(suite, d, s)] for s in ce_seeds if ce_records[(suite, d, s)] is not None]
            if len(vals) == len(ce_seeds):
                ce_mean[(suite, d)] = float(np.mean(vals))
                ce_std[(suite, d)] = float(np.std(vals, ddof=1))
            elif len(vals) > 0:
                ce_mean[(suite, d)] = float(np.mean(vals))
                ce_std[(suite, d)] = 0.0
            else:
                ce_mean[(suite, d)] = None
                ce_std[(suite, d)] = None

    # Macro averages per delay
    ce_macro_mean = {}
    ce_macro_std = {}
    baseline_macro = {}
    for d in delays:
        # CE macro
        seed_macros = []
        for s in ce_seeds:
            vals = [ce_records[(suite, d, s)] for suite in suites if ce_records[(suite, d, s)] is not None]
            if len(vals) == len(suites):
                seed_macros.append(float(np.mean(vals)))
        if len(seed_macros) == len(ce_seeds):
            ce_macro_mean[d] = float(np.mean(seed_macros))
            ce_macro_std[d] = float(np.std(seed_macros, ddof=1))
        elif len(seed_macros) > 0:
            ce_macro_mean[d] = float(np.mean(seed_macros))
            ce_macro_std[d] = 0.0
        else:
            ce_macro_mean[d] = None
            ce_macro_std[d] = None

        # Baseline macro
        b_vals = [baseline_records[(suite, d)] for suite in suites if baseline_records[(suite, d)] is not None]
        baseline_macro[d] = float(np.mean(b_vals)) if len(b_vals) == len(suites) else None

    # Compute AUC & Retention
    valid_ce_delays = [d for d in delays if ce_macro_mean[d] is not None]
    valid_ce_macro_vals = [ce_macro_mean[d] for d in valid_ce_delays]
    ce_auc = compute_auc(valid_ce_delays, valid_ce_macro_vals) if len(valid_ce_delays) >= 2 else 0.0
    ce_retention_40 = (
        (ce_macro_mean[40] / ce_macro_mean[0] * 100.0)
        if (0 in ce_macro_mean and 40 in ce_macro_mean and ce_macro_mean[0] and ce_macro_mean[40])
        else 0.0
    )

    valid_base_delays = [d for d in delays if baseline_macro[d] is not None]
    valid_base_macro_vals = [baseline_macro[d] for d in valid_base_delays]
    base_auc = compute_auc(valid_base_delays, valid_base_macro_vals) if len(valid_base_delays) >= 2 else 0.0
    base_retention_40 = (
        (baseline_macro[40] / baseline_macro[0] * 100.0)
        if (0 in baseline_macro and 40 in baseline_macro and baseline_macro[0] and baseline_macro[40])
        else 0.0
    )

    # Build Markdown Summary Table
    md = [
        "# CloudEdge π0.5 Aligned Full Reproduction Summary",
        "",
        "Evaluation across LIBERO benchmarks under simulated observation delay.",
        "Core protocol: 50 episodes/task $\\times$ 10 tasks = 500 episodes/suite, chunk 10, replan 5, flow steps 10.",
        "",
        "## Macro Summary & Metrics",
        "",
        "| Metric | Delayed Baseline (Seed 7) | **CloudEdge Aligned (Seeds 7, 8, 9)** | Paper Reference (OpenVLA-OFT) |",
        "|---|---|---|---|",
    ]

    for d in delays:
        b_str = f"{baseline_macro[d]:.2f}%" if baseline_macro[d] is not None else "*Pending*"
        if ce_macro_mean[d] is not None:
            c_str = f"**{ce_macro_mean[d]:.2f}% ± {ce_macro_std[d]:.2f}%**"
        else:
            c_str = "*Pending*"
        md.append(f"| Macro Success ($d_{{max}} = {d}$) | {b_str} | {c_str} | - |")

    md.append(f"| **Delay AUC (0 → 40)** | {base_auc:.2f}% | **{ce_auc:.2f}%** | 90.8% |")
    md.append(f"| **Retention at $d_{{max}}=40$** | {base_retention_40:.2f}% | **{ce_retention_40:.2f}%** | 76.5% |")
    md.append("")

    # Per-suite breakdown tables
    md.append("## Per-Suite Success Rates")
    md.append("")
    header = "| Benchmark Suite | Method | " + " | ".join([f"$d_{{max}}={d}$" for d in delays]) + " | AUC | Retention (40) |"
    sep = "|---|---|" + "|".join(["---" for _ in delays]) + "|---|---|"
    md.append(header)
    md.append(sep)

    for suite in suites:
        s_name = suite_names[suite]
        # Baseline row
        b_cols = []
        b_vals = []
        for d in delays:
            v = baseline_records[(suite, d)]
            b_cols.append(f"{v:.1f}%" if v is not None else "-")
            if v is not None:
                b_vals.append(v)
        b_auc_suite = compute_auc([d for d in delays if baseline_records[(suite, d)] is not None], b_vals)
        b_ret_suite = (
            (baseline_records[(suite, 40)] / baseline_records[(suite, 0)] * 100.0)
            if (baseline_records[(suite, 0)] and baseline_records[(suite, 40)])
            else 0.0
        )
        md.append(f"| {s_name} | Delayed Baseline | " + " | ".join(b_cols) + f" | {b_auc_suite:.1f}% | {b_ret_suite:.1f}% |")

        # CloudEdge row
        c_cols = []
        c_vals = []
        for d in delays:
            m = ce_mean[(suite, d)]
            s = ce_std[(suite, d)]
            c_cols.append(f"**{m:.1f}±{s:.1f}%**" if m is not None else "-")
            if m is not None:
                c_vals.append(m)
        c_auc_suite = compute_auc([d for d in delays if ce_mean[(suite, d)] is not None], c_vals)
        c_ret_suite = (
            (ce_mean[(suite, 40)] / ce_mean[(suite, 0)] * 100.0)
            if (ce_mean[(suite, 0)] and ce_mean[(suite, 40)])
            else 0.0
        )
        md.append(f"| {s_name} | **CloudEdge (Ours)** | " + " | ".join(c_cols) + f" | **{c_auc_suite:.1f}%** | **{c_ret_suite:.1f}%** |")

    md.append("")

    summary_data = {
        "delays": delays,
        "suites": suites,
        "ce_macro_mean": ce_macro_mean,
        "ce_macro_std": ce_macro_std,
        "baseline_macro": baseline_macro,
        "ce_auc": ce_auc,
        "ce_retention_40": ce_retention_40,
        "base_auc": base_auc,
        "base_retention_40": base_retention_40,
        "ce_mean": {f"{k[0]}_d{k[1]}": v for k, v in ce_mean.items()},
        "ce_std": {f"{k[0]}_d{k[1]}": v for k, v in ce_std.items()},
        "baseline": {f"{k[0]}_d{k[1]}": v for k, v in baseline_records.items()},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary_data, indent=2) + "\n")
    args.output_md.write_text("\n".join(md) + "\n")
    print(f"Aggregated summary written to {args.output_json} and {args.output_md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
