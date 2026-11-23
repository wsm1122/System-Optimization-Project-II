#!/usr/bin/env python3
"""Summarize public process-level R8-B and R9 evidence CSVs."""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = (
    "mean_ms",
    "p95_ms",
    "task_seconds",
    "rtf",
    "queries",
    "cer_percent",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(path):
    rows = read_rows(path)
    groups = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)

    print(f"[{path.name}] process_rows={len(rows)}")
    for variant, items in sorted(groups.items()):
        print(f"  {variant}: n_processes={len(items)}")
        for metric in METRICS:
            values = [float(item[metric]) for item in items if item.get(metric)]
            if not values:
                continue
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"    {metric}: mean={mean:.9g}, sd={sd:.9g}")


def main():
    for name in ("r8b_process_summary_public.csv", "r9_process_summary_public.csv"):
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"missing evidence file: {path.name}")
        summarize(path)


if __name__ == "__main__":
    main()
