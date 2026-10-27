"""CPU controls for paired-window arithmetic and whole-block inference."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("mlp_analyzer", args.analyzer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []

    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append(label)

    def reject(label, function):
        try:
            function()
        except (ValueError, KeyError, IndexError):
            checks.append(label)
        else:
            raise AssertionError(label)

    schedule = json.loads(args.schedule.read_text())
    module.validate_schedule(schedule)
    checks.append("actual prespecified schedule accepted")
    changed = copy.deepcopy(schedule)
    changed["entries"][1] = changed["entries"][0]
    reject(
        "duplicate worker assignment rejected",
        lambda: module.validate_schedule(changed),
    )
    orders = schedule["round_schedules"]["0"]["eager"]
    pairs = []
    for i, order in enumerate(orders):
        natural = (1.0, 10.0, 100.0)[i % 3]
        default = (1.0, 2.0, 100.0)[i % 3]
        rows = []
        for position, role in enumerate(order):
            wall = natural if role == "natural" else default
            gpu = default * (3 if role == "natural" else 1)
            rows.append(
                {
                    "role": role,
                    "iterations": 2,
                    "wall_batch_ms": wall * 2,
                    "wall_per_call_ms": wall,
                    "gpu_event_batch_ms": gpu * 2,
                    "gpu_event_per_call_ms": gpu,
                    "started_ns": i * 100 + position * 10,
                    "ended_ns": i * 100 + position * 10 + 5,
                }
            )
        pairs.append({"index": i, "order": order, "measurements": rows})
    eager = module.paired_logs(pairs, orders, "eager")
    graph = module.paired_logs(pairs, orders, "cuda_graph")
    check(
        "paired log median differs from ratio of medians",
        eager["natural_over_default"] == 1
        and eager["median_absolute_ms"]["natural"]
        / eager["median_absolute_ms"]["default"]
        == 5,
    )
    check("graph selects GPU clock", math.isclose(graph["natural_over_default"], 3))
    noisy = copy.deepcopy(pairs)
    for pair in noisy:
        rows = {row["role"]: row for row in pair["measurements"]}
        for suffix in ("batch_ms", "per_call_ms"):
            rows["natural"]["wall_" + suffix] = 1.1 * rows["default"]["wall_" + suffix]
    check(
        "independent same-default captures retain measured ratio",
        math.isclose(
            module.paired_logs(noisy, orders, "eager")["natural_over_default"], 1.1
        ),
    )
    reject(
        "missing timing pair rejected",
        lambda: module.paired_logs(pairs[:-1], orders, "eager"),
    )
    reject(
        "unknown clock mode rejected",
        lambda: module.paired_logs(pairs, orders, "other"),
    )
    mutations = (
        ("pair ordinal drift", lambda p: p[1].update(index=0)),
        ("execution order drift", lambda p: p[0]["measurements"].reverse()),
        (
            "mismatched paired batch length",
            lambda p: p[0]["measurements"][0].update(iterations=3),
        ),
        ("zero batch length", lambda p: p[0]["measurements"][0].update(iterations=0)),
        (
            "per-call value inconsistent with raw batch",
            lambda p: p[0]["measurements"][0].update(wall_per_call_ms=12.345),
        ),
        (
            "unordered sample endpoints",
            lambda p: p[0]["measurements"][0].update(ended_ns=-1),
        ),
        (
            "overlapping samples within pair",
            lambda p: p[0]["measurements"][1].update(started_ns=1),
        ),
        (
            "overlapping samples across pairs",
            lambda p: p[1]["measurements"][0].update(started_ns=1),
        ),
    )
    for label, mutate in mutations:
        broken = copy.deepcopy(pairs)
        mutate(broken)
        reject(label + " rejected", lambda: module.paired_logs(broken, orders, "eager"))
    for bad in (0, -1, math.inf, math.nan, True):
        broken = copy.deepcopy(pairs)
        broken[0]["measurements"][0]["wall_batch_ms"] = bad
        reject(
            f"invalid timing {bad!r} rejected",
            lambda: module.paired_logs(broken, orders, "eager"),
        )
    check("type7 interpolated quantile", module.quantile([0, 10, 20, 30], 0.25) == 7.5)
    estimate = module.estimates([math.log(2), math.log(8)], [[0, 0], [1, 1]])
    for label, resamples in (
        ("missing resamples", []),
        ("partial block resample", [[0]]),
        ("out-of-range block index", [[0, 2]]),
    ):
        reject(
            label + " rejected",
            lambda: module.estimates([math.log(2), math.log(8)], resamples),
        )
    check(
        "geometric mean is computed in log space",
        math.isclose(estimate["geometric_mean_ratio"], 4),
    )
    check(
        "whole-block resample interval",
        all(
            math.isclose(a, b)
            for a, b in zip(estimate["exploratory_block_bootstrap_95"], [2.15, 7.85])
        ),
    )

    processes = []
    for trial in range(6):
        for arm in module.ARMS:
            for phase in ("cold", "warm"):
                ratio = {"default": 1.01, "v1": 2 ** (trial + 1), "v2": 2**trial}[arm]
                latency = {"default": 9, "v1": 10 + trial, "v2": 20 + 2 * trial}[arm]
                processes.append(
                    {
                        "trial": trial,
                        "arm": arm,
                        "phase": phase,
                        "modes": {
                            mode: {
                                "median_log_ratio": math.log(ratio),
                                "median_absolute_ms": {
                                    "natural": latency,
                                    "default": latency / ratio,
                                },
                            }
                            for mode in module.MODES
                        },
                    }
                )
    reject("partial cohort rejected", lambda: module.summarize(processes[:-1]))
    duplicated = copy.deepcopy(processes)
    duplicated[1] = duplicated[0]
    reject("duplicate cohort worker rejected", lambda: module.summarize(duplicated))
    original_estimates = module.estimates
    observed = []

    def bounded_estimates(logs, resamples):
        # Inspect the actual full resample plan; a short prefix suffices to test
        # wiring. The production analyzer retains all 100,000 bootstrap means.
        observed.append((list(logs), resamples))
        return original_estimates(logs, resamples[:24])

    module.estimates = bounded_estimates
    try:
        summary = module.summarize(processes)
    finally:
        module.estimates = original_estimates
    check(
        "same paired block resamples shared across all metrics",
        len(observed) == 20 and all(row[1] is observed[0][1] for row in observed),
    )
    resamples = observed[0][1]
    check(
        "bootstrap unit is six blocks, not replay windows",
        len(resamples) == 100000
        and all(
            len(indices) == 6 and all(type(i) is int and 0 <= i < 6 for i in indices)
            for indices in resamples
        ),
    )
    rng = random.Random(202610270000)
    check(
        "prespecified resampling seed retained",
        resamples[:24] == [[rng.randrange(6) for _ in range(6)] for _ in range(24)],
    )
    for phase in ("warm", "cold"):
        for mode in module.MODES:
            value = summary["estimates"][phase][mode]
            check(
                f"{phase} {mode} normalization precedes contrast",
                math.isclose(
                    value["v2_over_v1_normalized_to_own_default"][
                        "geometric_mean_ratio"
                    ],
                    0.5,
                ),
            )
            check(
                f"{phase} {mode} absolute latency kept separate",
                math.isclose(
                    value["v2_over_v1_absolute_natural_latency_secondary"][
                        "geometric_mean_ratio"
                    ],
                    2,
                ),
            )
            check(
                f"{phase} {mode} default noise survives cohort summary",
                math.isclose(
                    value["natural_over_own_default"]["default"][
                        "geometric_mean_ratio"
                    ],
                    1.01,
                ),
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "checks": len(checks),
                "analyzer_sha256": module.sha(args.analyzer),
                "passed": checks,
                "scope": "CPU arithmetic, malformed-data and shared whole-block resampling controls; shortened bootstrap evaluation in summary wiring control only",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "passed", "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
