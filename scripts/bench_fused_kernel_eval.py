#!/usr/bin/env python3
"""
Benchmark harness for comparing baseline vs fused-kernel serving behavior.

This script runs three benchmark profiles:
  1) low_noise_decode_focus
  2) moderate_load_sanity
  3) realism_stress

For each profile/concurrency pair, it runs both variants (baseline/fused)
for N repeats (default 5), saves per-run results to CSV, emits aggregated
summary CSV, and prints clear conclusions at the end.

This script intentionally uses bench_serving defaults for endpoint selection
(no --base-url / --host / --port overrides).

Use it as a single-run collector and execute it multiple times with labels:
  --variant baseline|fused
  --kv-mode fp4|fp8

Example:
  python3 scripts/bench_fused_kernel_eval.py --variant baseline --kv-mode fp4
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    num_prompt: int
    random_input: int
    random_output: int
    random_range_ratio: float
    max_concurrency_values: list[int]


PROFILES: list[Profile] = [
    Profile(
        name="low_noise_decode_focus",
        num_prompt=256,
        random_input=16,
        random_output=2048,
        random_range_ratio=0.0,
        max_concurrency_values=[1, 4],
    ),
    Profile(
        name="realism_stress",
        num_prompt=512,
        random_input=1024,
        random_output=1024,
        random_range_ratio=1.0,
        max_concurrency_values=[64],
    ),
]


RAW_FIELDS = [
    "profile",
    "max_concurrency",
    "variant",
    "kv_mode",
    "run_tag",
    "repeat_idx",
    "duration_s",
    "request_throughput",
    "output_throughput",
    "total_throughput",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_ttft_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
    "completed",
    "total_output_tokens",
]

SUMMARY_FIELDS = [
    "profile",
    "max_concurrency",
    "variant",
    "kv_mode",
    "run_tag",
    "runs",
    "median_output_throughput",
    "mean_output_throughput",
    "stdev_output_throughput",
    "median_mean_tpot_ms",
    "mean_mean_tpot_ms",
    "stdev_mean_tpot_ms",
    "median_mean_ttft_ms",
    "mean_mean_itl_ms",
    "mean_mean_e2e_latency_ms",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    rank = (len(s) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def stdev_or_zero(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def run_once(
    profile: Profile,
    concurrency: int,
) -> dict[str, Any]:
    # bench_serving appends one JSON line to output_file; use a unique temp file per run.
    with tempfile.NamedTemporaryFile(
        prefix="bench_serving_", suffix=".jsonl", delete=False
    ) as tf:
        output_file = Path(tf.name)

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--dataset-name",
        "random",
        "--num-prompt",
        str(profile.num_prompt),
        "--random-input",
        str(profile.random_input),
        "--random-output",
        str(profile.random_output),
        "--random-range-ratio",
        str(profile.random_range_ratio),
        "--max-concurrency",
        str(concurrency),
        "--output-file",
        str(output_file),
    ]

    started = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=False,
        check=False,
    )
    elapsed = time.time() - started

    if proc.returncode != 0:
        raise RuntimeError(f"bench_serving failed with exit code {proc.returncode}")

    try:
        lines = output_file.read_text().strip().splitlines()
        if not lines:
            raise RuntimeError("bench_serving produced empty output file")
        result = json.loads(lines[-1])
    finally:
        try:
            output_file.unlink(missing_ok=True)
        except OSError:
            pass

    result["duration_s"] = elapsed
    return result


def aggregate_variant(rows: list[dict[str, Any]]) -> dict[str, float]:
    out_thr = [float(r["output_throughput"]) for r in rows]
    mean_tpot = [float(r["mean_tpot_ms"]) for r in rows]
    mean_ttft = [float(r["mean_ttft_ms"]) for r in rows]
    mean_itl = [float(r["mean_itl_ms"]) for r in rows]
    mean_e2e = [float(r["mean_e2e_latency_ms"]) for r in rows]

    return {
        "runs": len(rows),
        "median_output_throughput": statistics.median(out_thr),
        "mean_output_throughput": statistics.mean(out_thr),
        "stdev_output_throughput": stdev_or_zero(out_thr),
        "median_mean_tpot_ms": statistics.median(mean_tpot),
        "mean_mean_tpot_ms": statistics.mean(mean_tpot),
        "stdev_mean_tpot_ms": stdev_or_zero(mean_tpot),
        "median_mean_ttft_ms": statistics.median(mean_ttft),
        "mean_mean_itl_ms": statistics.mean(mean_itl),
        "mean_mean_e2e_latency_ms": statistics.mean(mean_e2e),
    }


def print_block_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_progress_line(
    profile_name: str,
    concurrency: int,
    variant: str,
    rep: int,
    repeats: int,
):
    print(
        f"[RUN] profile={profile_name:22s} conc={concurrency:4d} variant={variant:8s} "
        f"repeat={rep}/{repeats}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect repeated bench_serving runs for one labeled variant."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["baseline", "fused"],
        help="Label for this run set.",
    )
    parser.add_argument(
        "--kv-mode",
        required=True,
        choices=["fp4", "fp8"],
        help="KV mode label for this run set.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional extra tag (e.g., branch or commit).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per (profile, concurrency).",
    )
    parser.add_argument(
        "--output-dir",
        default="bench_fused_eval_results",
        help="Directory for CSV outputs.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("Please use repeats >= 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.variant}_{args.kv_mode}"
    if args.run_tag:
        suffix += f"_{args.run_tag}"
    raw_csv_path = output_dir / f"raw_runs_{suffix}.csv"
    summary_csv_path = output_dir / f"summary_{suffix}.csv"

    all_raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    total_jobs = sum(len(p.max_concurrency_values) for p in PROFILES) * args.repeats
    done_jobs = 0

    print_block_header("Fused Kernel Benchmark Harness")
    print("backend          : sglang (hard-coded)")
    print("endpoint         : bench_serving defaults (no base-url override)")
    print(f"variant label    : {args.variant}")
    print(f"kv mode label    : {args.kv_mode}")
    print(f"run tag          : {args.run_tag or '(none)'}")
    print(f"repeats          : {args.repeats}")
    print(f"output_dir       : {output_dir}")
    print(f"total runs       : {total_jobs}")

    for profile in PROFILES:
        print_block_header(
            f"Profile: {profile.name} | Variant: {args.variant} | KV: {args.kv_mode}"
        )
        print(
            "Config: "
            f"num_prompt={profile.num_prompt}, "
            f"random_input={profile.random_input}, "
            f"random_output={profile.random_output}, "
            f"random_range_ratio={profile.random_range_ratio}, "
            f"concurrency={profile.max_concurrency_values}"
        )
        for concurrency in profile.max_concurrency_values:
            print(
                f"\n[BEGIN] profile={profile.name}, conc={concurrency}, "
                f"variant={args.variant}, kv_mode={args.kv_mode}"
            )
            for rep in range(1, args.repeats + 1):
                print_progress_line(
                    profile.name, concurrency, args.variant, rep, args.repeats
                )
                result = run_once(
                    profile=profile,
                    concurrency=concurrency,
                )
                raw_row = {
                    "profile": profile.name,
                    "max_concurrency": concurrency,
                    "variant": args.variant,
                    "kv_mode": args.kv_mode,
                    "run_tag": args.run_tag,
                    "repeat_idx": rep,
                    "duration_s": float(result["duration_s"]),
                    "request_throughput": float(result["request_throughput"]),
                    "output_throughput": float(result["output_throughput"]),
                    "total_throughput": float(result["total_throughput"]),
                    "mean_tpot_ms": float(result["mean_tpot_ms"]),
                    "median_tpot_ms": float(result["median_tpot_ms"]),
                    "p99_tpot_ms": float(result["p99_tpot_ms"]),
                    "mean_ttft_ms": float(result["mean_ttft_ms"]),
                    "mean_itl_ms": float(result["mean_itl_ms"]),
                    "mean_e2e_latency_ms": float(result["mean_e2e_latency_ms"]),
                    "completed": int(result["completed"]),
                    "total_output_tokens": int(result["total_output_tokens"]),
                }
                all_raw_rows.append(raw_row)
                done_jobs += 1
                print(
                    f"      -> out_thr={raw_row['output_throughput']:.2f} tok/s, "
                    f"mean_tpot={raw_row['mean_tpot_ms']:.3f} ms, "
                    f"mean_ttft={raw_row['mean_ttft_ms']:.2f} ms "
                    f"[{done_jobs}/{total_jobs}]"
                )

            rows_for_group = [
                r
                for r in all_raw_rows
                if r["profile"] == profile.name and r["max_concurrency"] == concurrency
            ]
            stats = aggregate_variant(rows_for_group)
            summary_rows.append(
                {
                    "profile": profile.name,
                    "max_concurrency": concurrency,
                    "variant": args.variant,
                    "kv_mode": args.kv_mode,
                    "run_tag": args.run_tag,
                    **stats,
                }
            )
            print(
                f"[END] profile={profile.name}, conc={concurrency}, variant={args.variant}, "
                f"kv_mode={args.kv_mode} | "
                f"median_out_thr={stats['median_output_throughput']:.2f}, "
                f"median_mean_tpot={stats['median_mean_tpot_ms']:.3f} ms"
            )

    with raw_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(all_raw_rows)

    with summary_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print_block_header("Run Summary (This Variant Only)")
    header = (
        f"{'profile':24s} {'conc':>6s} {'med_out_thr':>12s} {'mean_out_thr':>12s} "
        f"{'med_tpot_ms':>12s} {'mean_tpot_ms':>13s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['profile']:24s} {int(row['max_concurrency']):6d} "
            f"{row['median_output_throughput']:12.2f} {row['mean_output_throughput']:12.2f} "
            f"{row['median_mean_tpot_ms']:12.3f} {row['mean_mean_tpot_ms']:13.3f}"
        )

    overall_out_thr = [float(r["output_throughput"]) for r in all_raw_rows]
    overall_tpot = [float(r["mean_tpot_ms"]) for r in all_raw_rows]
    print_block_header("Overall Stats (This Variant Only)")
    print(
        f"output_throughput tok/s: mean={statistics.mean(overall_out_thr):.2f}, "
        f"median={statistics.median(overall_out_thr):.2f}, "
        f"stdev={stdev_or_zero(overall_out_thr):.2f}"
    )
    print(
        f"mean_tpot_ms:            mean={statistics.mean(overall_tpot):.3f}, "
        f"median={statistics.median(overall_tpot):.3f}, "
        f"stdev={stdev_or_zero(overall_tpot):.3f}"
    )
    print(
        "Conclusion: this script run captures one labeled variant only. "
        "Run all four variants and compare CSVs side-by-side."
    )

    print(f"\nWrote raw runs CSV:      {raw_csv_path}")
    print(f"Wrote summary CSV:       {summary_csv_path}")


if __name__ == "__main__":
    main()
