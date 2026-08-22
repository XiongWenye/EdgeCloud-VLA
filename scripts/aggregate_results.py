#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path


NAME = re.compile(
    r"(?P<method>cloudedge|pi05)_d(?P<delay>\d+)_(?P<suite>libero_[a-z0-9]+)_s(?P<seed>\d+)"
)


def mean(values):
    return sum(values) / len(values)


def stdev(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--validate-paper-grid", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("eval_info.json")):
        match = NAME.search(path.parent.name)
        if not match:
            continue
        with path.open() as handle:
            payload = json.load(handle)
        success = float(payload["overall"]["pc_success"])
        if success <= 1.0:
            success *= 100.0
        row = match.groupdict()
        row["delay"] = int(row["delay"])
        row["seed"] = int(row["seed"])
        row["success_percent"] = success
        row["source"] = str(path)
        rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["delay"], row["suite"])].append(row["success_percent"])

    if args.validate_paper_grid:
        methods = {"cloudedge": 3, "pi05": 1}
        delays = (0, 10, 20, 40)
        suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        expected = set(itertools.product(methods, delays, suites))
        actual = set(grouped)
        if actual != expected:
            raise RuntimeError(
                f"Evaluation grid mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for key, values in grouped.items():
            expected_seeds = methods[key[0]]
            if len(values) != expected_seeds:
                raise RuntimeError(
                    f"Expected {expected_seeds} seeds for {key}, found {len(values)}"
                )

    summary_rows = []
    for key, values in sorted(grouped.items()):
        summary_rows.append([*key, mean(values), stdev(values), len(values)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "d_max", "suite", "mean_success_percent", "std", "n_seeds"])
        for method, delay, suite, avg, spread, n_seeds in summary_rows:
            writer.writerow([method, delay, suite, f"{avg:.4f}", f"{spread:.4f}", n_seeds])

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "## Closed-loop LIBERO results",
            "",
            "Success rate over 50 episodes per task. CloudEdge reports mean and sample standard deviation over seeds 7, 8, and 9; the π0.5 delayed baseline uses seed 7.",
            "",
            "| Method | d_max | Suite | Success (%) | Std | Seeds |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for method, delay, suite, avg, spread, n_seeds in summary_rows:
            lines.append(
                f"| {method} | {delay} | {suite} | {avg:.2f} | {spread:.2f} | {n_seeds} |"
            )
        args.markdown_output.write_text("\n".join(lines) + "\n")
    print(f"runs={len(rows)} groups={len(grouped)} output={args.output}")


if __name__ == "__main__":
    main()
