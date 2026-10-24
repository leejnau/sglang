#!/usr/bin/env python3
"""Validate the fixed objective-by-values cohort and estimate block contrasts."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

BATCH = "test/manual/kernels/results/flashinfer_autotune_selection/20261024T180000Z-h100-objective-values"
CELLS = ("auto_synthetic", "graph_synthetic", "auto_captured", "graph_captured")
GRAPHS = ("winner", "default")
BOOTSTRAP_SEED = 202610241803
BOOTSTRAP_RESAMPLES = 200_000
SCHEDULE_SEED = 202610241801
CAPTURE_SEED = 202610241802
PROPOSAL_SHA256 = "347b905b8b421f3f6d1160d3b40f3b7aa4348fbaf4fc0602bf5608f42932a45b"
RUNTIME_SHA256 = {
    "flashinfer/autotuner/autotuner.py": "0db42ad90d1a3fec6f81509a2c520c1892cea0a433f5dbd7f43468a2a10243ca",
    "flashinfer/autotune_cache.py": "1ab2ed7208ed1bf1007b1f74de29fc818c93a5ab091beea0833474177fbb29ad",
    "flashinfer/autotuner/_graph_timing.py": "5f3be31e17501ba11ee95327176ac4a4b95eeaee7be92c2bc7795bcbd30cefe3",
    "flashinfer/gemm/gemm_base.py": "0bba9240029a5e09f331e7a2267f7067e126d57a7a341ca535f7c9557599adca",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def positive(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label)
    require(math.isfinite(value) and value > 0, label)
    return float(value)


def quantile(values, probability):
    """Hyndman-Fan type7, including singleton and exact endpoints."""
    ordered = sorted(values)
    require(bool(ordered) and 0 <= probability <= 1, "quantile domain")
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def describe(values):
    require(bool(values), "nonempty independent values")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "range": [min(values), max(values)],
        "values": list(values),
    }


def exact_binomial_interval(k, n, confidence=0.95):
    """Equal-tail Clopper-Pearson interval and one-sided upper bound."""
    require(
        isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0,
        "binomial counts",
    )
    alpha = 1 - confidence

    def cdf(p, count):
        return sum(
            math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(count + 1)
        )

    def solve(count, target):
        low, high = 0.0, 1.0
        for _ in range(100):
            middle = (low + high) / 2
            if cdf(middle, count) > target:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    lower = 0.0 if k == 0 else solve(k - 1, 1 - alpha / 2)
    upper = 1.0 if k == n else solve(k, alpha / 2)
    one_upper = 1.0 if k == n else solve(k, alpha)
    return {
        "count": k,
        "n": n,
        "fraction": k / n,
        "confidence": confidence,
        "two_sided_clopper_pearson": [lower, upper],
        "one_sided_upper": one_upper,
        "assumption_limit": "Fixed-input process repetitions; binomial stationarity/independence is not established by this pilot.",
    }


def block_contrasts(worker_ratios):
    require(set(worker_ratios) == set(CELLS), "complete four-cell block")
    a, b, c, d = (math.log(positive(worker_ratios[cell], cell)) for cell in CELLS)
    return {
        "objective": ((b - a) + (d - c)) / 2,
        "values": ((c - a) + (d - b)) / 2,
        "interaction": (d - c) - (b - a),
    }


def summarize_statistics(workers, *, resamples=BOOTSTRAP_RESAMPLES):
    by_block = collections.defaultdict(dict)
    for worker in workers:
        block, cell = worker["block"], worker["cell"]
        require(cell not in by_block[block], "duplicate block/cell")
        by_block[block][cell] = worker
    require(
        sorted(by_block) == list(range(len(by_block))), "contiguous complete blocks"
    )
    contrasts = []
    for block in sorted(by_block):
        values = by_block[block]
        require(set(values) == set(CELLS), "incomplete four-cell block")
        contrasts.append(
            {
                "block": block,
                **block_contrasts(
                    {cell: values[cell]["event_ratio"] for cell in CELLS}
                ),
            }
        )
    result = {
        "independent_blocks": len(contrasts),
        "worker_count": len(workers),
        "block_log_contrasts": contrasts,
        "cells": {},
    }
    for cell in CELLS:
        rows = [by_block[b][cell] for b in sorted(by_block)]
        result["cells"][cell] = {
            "event_winner_over_default": describe([r["event_ratio"] for r in rows]),
            "queued_wall_ratio_diagnostic_only": describe(
                [r["wall_ratio"] for r in rows]
            ),
            "default_selections": exact_binomial_interval(
                sum(r["selected_default"] for r in rows), len(rows)
            ),
            "explicit_selections": exact_binomial_interval(
                sum(not r["selected_default"] for r in rows), len(rows)
            ),
            "material_regressions_ratio_gt_1_10": exact_binomial_interval(
                sum(r["event_ratio"] > 1.10 for r in rows), len(rows)
            ),
            "worker_outcomes": rows,
        }
        if all("capture_order" in row for row in rows):
            result["cells"][cell]["capture_order_strata"] = {
                order: describe(
                    [
                        row["event_ratio"]
                        for row in rows
                        if row["capture_order"] == order
                    ]
                )
                for order in ("winner-first", "default-first")
                if any(row["capture_order"] == order for row in rows)
            }
        if all("order_strata_event_ratios" in row for row in rows):
            result["cells"][cell]["within_round_order_strata_worker_medians"] = {
                order + "_first": describe(
                    [
                        row["order_strata_event_ratios"][order + "_first"]["median"]
                        for row in rows
                    ]
                )
                for order in GRAPHS
            }
    result["factor_estimates"] = {}
    for factor in ("objective", "values", "interaction"):
        logs = [c[factor] for c in contrasts]
        result["factor_estimates"][factor] = {
            "ratio_of_ratios": math.exp(statistics.mean(logs)),
            "block_log_contrasts": describe(logs),
            "block_ratio_contrasts": describe([math.exp(value) for value in logs]),
            "primary": factor != "interaction",
        }
    if len(contrasts) == 8:
        names = ("objective", "values", "interaction")
        samples = {name: [] for name in names}
        rng = random.Random(BOOTSTRAP_SEED)
        for _ in range(resamples):
            indices = [rng.randrange(8) for _ in range(8)]
            for name in names:
                samples[name].append(
                    sum(contrasts[index][name] for index in indices) / 8
                )
        for name in names:
            probabilities = (
                (0.0125, 0.9875) if name != "interaction" else (0.025, 0.975)
            )
            result["factor_estimates"][name]["bootstrap_interval"] = {
                "confidence": 0.975 if name != "interaction" else 0.95,
                "bounds": [math.exp(quantile(samples[name], p)) for p in probabilities],
                "resamples": resamples,
                "seed": BOOTSTRAP_SEED,
                "unit": "whole four-cell block",
                "quantile": "Hyndman-Fan type7",
                "family": "two prespecified factors"
                if name != "interaction"
                else "secondary interaction",
                "interpretation": "Exploratory percentile interval; no equivalence, safety or superiority claim.",
            }
    else:
        result["bootstrap_status"] = (
            "Not estimated at the first-block integrity gate. Eight complete blocks are required."
        )
    result["interpretation_limits"] = [
        "Twenty-four paired rounds are repeated observations within one worker, not independent trials.",
        "The synthetic initializer is a matched clone/copy control. Source contents/addresses and physical history can differ from captured-values mode.",
        "Queued wall time includes prequeue delay and is diagnostic only.",
        "Small same-default favorable ratios are unresolved by this calibration and do not demonstrate a serving speedup.",
        "No historical pooling, rare-regression safety, production policy or persistence conclusion.",
    ]
    return result


def expected_round_orders(block):
    rows = [["winner", "default"]] * 12 + [["default", "winner"]] * 12
    random.Random(180100 + block).shuffle(rows)
    return rows


def validate_schedule(schedule):
    require(
        schedule["schema_version"] == 1 and schedule["cells"] == list(CELLS),
        "schedule cell mapping",
    )
    require(
        schedule["independent_blocks"] == 8 and schedule["workers"] == 32,
        "fixed schedule size",
    )
    sequences = ("ABDC", "BCAD", "CDBA", "DACB")
    rng = random.Random(SCHEDULE_SEED)
    indices = []
    for _ in range(2):
        repeat = list(range(4))
        rng.shuffle(repeat)
        indices.extend(repeat)
    randomization = schedule["launch_randomization"]
    require(
        randomization["seed"] == SCHEDULE_SEED
        and randomization["sequence_indices"] == indices,
        "frozen Williams assignment",
    )
    require(
        randomization["sequences"] == list(sequences), "Williams sequence definitions"
    )
    entries = schedule["entries"]
    require(len(entries) == 32, "all32 scheduled workers")
    bits = schedule["capture_randomization"]["winner_first_bits_in_cell_order"]
    require(
        schedule["capture_randomization"]["seed"] == CAPTURE_SEED,
        "independent capture seed",
    )
    require(
        len(bits) == 8 and all(len(row) == 4 and set(row) <= {0, 1} for row in bits),
        "capture matrix",
    )
    for block in range(8):
        require(sum(bits[block]) == 2, "two captures of each order per block")
        order = [CELLS["ABCD".index(letter)] for letter in sequences[indices[block]]]
        rows = entries[block * 4 : block * 4 + 4]
        for position, (cell, row) in enumerate(zip(order, rows)):
            objective, values = cell.split("_")
            require(
                row
                == {
                    "block": block,
                    "williams_sequence": sequences[indices[block]],
                    "launch_position": position,
                    "launch_index": block * 4 + position,
                    "cell": cell,
                    "objective": objective,
                    "values": values,
                    "capture_order": "winner-first"
                    if bits[block][CELLS.index(cell)]
                    else "default-first",
                    "replay_order_seed": 180100 + block,
                    "raw_name": f"block{block}-{cell}-timing.json",
                    "trace_name": f"block{block}-{cell}-timing.trace.json",
                    "bootstrap_name": f"block{block}-{cell}-timing.bootstrap.json",
                },
                "exact scheduled worker",
            )
    for cell in range(4):
        require(
            sum(bits[b][cell] for b in range(8)) == 4,
            "capture order balanced within cell",
        )
        require(
            sum(bits[b][cell] for b in range(4)) == 2,
            "capture order balanced within Williams repeat",
        )
        require(
            sum(bits[b][cell] for b in range(0, 8, 2)) == 2,
            "capture order balanced across block parity",
        )
    for position in range(4):
        require(
            sum(
                row["capture_order"] == "winner-first"
                for row in entries
                if row["launch_position"] == position
            )
            == 4,
            "capture order balanced across launch positions",
        )
    require(
        schedule["round_schedules"]
        == [
            {"block": b, "seed": 180100 + b, "orders": expected_round_orders(b)}
            for b in range(8)
        ],
        "frozen24paired rounds",
    )
    # Verify first-order carryover balance over each complete Williams repeat.
    for repeat in range(2):
        transitions = collections.Counter()
        positions = collections.Counter()
        for block in range(repeat * 4, repeat * 4 + 4):
            order = [row["cell"] for row in entries[block * 4 : block * 4 + 4]]
            transitions.update(zip(order, order[1:]))
            positions.update((position, cell) for position, cell in enumerate(order))
        require(
            set(transitions.values()) == {1} and len(transitions) == 12,
            "Williams ordered-pair balance",
        )
        require(
            set(positions.values()) == {1} and len(positions) == 16,
            "Williams launch-position balance",
        )
    return entries


def validate_plan(plan, schedule, schedule_path):
    require(
        plan["schema_version"] == 1 and plan["batch_id"] == Path(BATCH).name,
        "fixed study batch",
    )
    require(plan["proposal_sha256"] == PROPOSAL_SHA256, "frozen reviewed proposal")
    require(plan["schedule_sha256"] == sha256(schedule_path), "plan/schedule binding")
    require(
        plan["source_contract"]["flashinfer_commit"]
        == "27c62fdf0ab1cd4014df8a15b66a975541fda83c",
        "fixed current FlashInfer source",
    )
    require(
        plan["source_contract"]["runtime_sha256"] == RUNTIME_SHA256,
        "unchanged tuner/cache/timer/FP8 source",
    )
    protocol = plan["protocol"]
    require(
        (
            protocol["independent_blocks"],
            protocol["timing_workers"],
            protocol["rounds"],
            protocol["replays_per_sample"],
            protocol["prequeue_cycles"],
        )
        == (8, 32, 24, 16, 10_000_000),
        "fixed replication/replay recipe",
    )
    require(
        protocol["actual_profile_ms"] == [1, 2, 4, 8, 16, 32, 64, 128, 144]
        and protocol["runtime_cache_m"] == 256,
        "actual endpoint versus normalized bucket",
    )
    require(
        protocol["startup_history"] is False
        and protocol["inline_gpu_fingerprints"] is False
        and protocol["cache_persistence"] is False,
        "fixed unobserved startup/memory-only cache",
    )
    require(
        protocol["objective_auto"] is None
        and protocol["objective_graph"] == {"execution_mode": "cuda_graph"},
        "only objective policy override",
    )
    bootstrap = plan["analysis"]["bootstrap"]
    require(
        bootstrap["resamples"] == BOOTSTRAP_RESAMPLES
        and bootstrap["seed"] == BOOTSTRAP_SEED
        and bootstrap["interval_level"] == 0.975
        and bootstrap["tail_probabilities"] == [0.0125, 0.9875],
        "frozen adjusted whole-block inference",
    )
    return validate_schedule(schedule)


def tactic_key(tactic):
    if tactic == -1:
        require(type(tactic) is int, "integer default tactic")
        return "-1"
    require(
        isinstance(tactic, (list, tuple)) and len(tactic) == 8,
        "complete64byte explicit descriptor",
    )
    require(
        all(type(word) is int and 0 <= word < 2**64 for word in tactic),
        "uint64 descriptor words",
    )
    return json.dumps(tactic, separators=(",", ":"))


def trace_signature(event):
    args = event["args"]
    signature = {
        "name": event["name"],
        **{
            key: args[key]
            for key in (
                "grid",
                "block",
                "registers per thread",
                "shared memory",
                "device",
            )
        },
    }
    require(signature["device"] == 0, "trace exclusive visible device")
    require(
        all(
            isinstance(signature[key], list)
            and len(signature[key]) == 3
            and all(type(v) is int and v > 0 for v in signature[key])
            for key in ("grid", "block")
        ),
        "complete positive CUDA launch dimensions",
    )
    require(
        all(
            type(signature[key]) is int and signature[key] >= 0
            for key in ("registers per thread", "shared memory")
        ),
        "complete CUDA launch resource signature",
    )
    return signature


def parse_retained_trace(trace):
    events = trace["traceEvents"]
    result = {}
    runtime_correlations = []
    native_graph_ids = []
    observed_kernel_count = 0
    for graph in GRAPHS:
        markers = [
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event.get("ph") == "X"
            and event.get("name") == "retained_" + graph
        ]
        require(len(markers) == 1, "one original graph trace scope")
        marker = markers[0]
        begin, end = marker["ts"], marker["ts"] + marker["dur"]
        launches = sorted(
            [
                event
                for event in events
                if event.get("cat") == "cuda_runtime"
                and "cudaGraphLaunch" in event.get("name", "")
                and begin <= event["ts"] <= end
            ],
            key=lambda event: event["ts"],
        )
        require(len(launches) == 3, "three actual retained graph launches")
        correlations = [launch["args"]["correlation"] for launch in launches]
        require(len(set(correlations)) == 3, "distinct graph-launch correlations")
        runtime_correlations.extend(correlations)
        cycles = []
        graph_ids = set()
        for correlation in correlations:
            kernels = sorted(
                [
                    event
                    for event in events
                    if event.get("cat") == "kernel"
                    and event.get("args", {}).get("correlation") == correlation
                ],
                key=lambda event: event["ts"],
            )
            require(len(kernels) >= 2, "retained full-layer quantizer and GEMM")
            require(
                all(
                    begin <= event["ts"] and event["ts"] + event["dur"] <= end + 1
                    for event in kernels
                ),
                "trace kernel completion inside retained scope",
            )
            require(
                "static_quant_fp8" in kernels[0]["name"]
                and any(
                    "static_quant_fp8" not in event["name"] for event in kernels[1:]
                ),
                "original static quantizer before matrix operation",
            )
            graph_ids.update(event["args"]["graph id"] for event in kernels)
            cycles.append([trace_signature(event) for event in kernels])
            observed_kernel_count += len(kernels)
        require(
            cycles[0] == cycles[1] == cycles[2],
            "same ordered kernel signatures across all original replays",
        )
        require(
            len(graph_ids) == 1
            and type(next(iter(graph_ids))) is int
            and next(iter(graph_ids)) > 0,
            "actual CUDA graph identity",
        )
        native_graph_ids.extend(graph_ids)
        result[graph] = {
            "cuda_graph_id": next(iter(graph_ids)),
            "graph_launches": 3,
            "ordered_launch_signature": cycles[0],
            "kernel_events": sum(len(cycle) for cycle in cycles),
        }
    require(len(set(native_graph_ids)) == 2, "two independently retained CUDA graphs")
    require(
        len(set(runtime_correlations)) == 6,
        "distinct runtime launches across retained scopes",
    )
    require(
        observed_kernel_count
        == len([event for event in events if event.get("cat") == "kernel"]),
        "no unreported CUDA kernels in posttiming trace",
    )
    result["winner_default_signatures_equal"] = (
        result["winner"]["ordered_launch_signature"]
        == result["default"]["ordered_launch_signature"]
    )
    return result


def normalize_provider_identity(bootstrap):
    providers = bootstrap["native"]["providers"]
    require(
        set(providers) == {"gemm", "trtllm_utils"}, "qualified GEMM and delay providers"
    )
    identities = {}
    for name, provider in providers.items():
        require(
            all(
                isinstance(provider[key], str)
                and len(provider[key]) == 64
                and all(c in "0123456789abcdef" for c in provider[key])
                for key in ("library_sha256", "ninja_sha256")
            ),
            "actual native artifact hashes",
        )
        require(
            provider["linkage"]["returncode"] == 0
            and "not found" not in provider["linkage"]["stdout"],
            "resolved native dependencies",
        )
        require(
            "/cached_ops/" in provider["library_path"]
            and "/site-packages/" not in provider["library_path"],
            "explicitly built current source provider",
        )
        expected_exports = (
            {"bmm_fp8", "bmm_fp8_get_algos", "bmm_fp8_run_with_descriptor"}
            if name == "gemm"
            else {"delay_kernel"}
        )
        require(
            set(provider["exports_checked"]) == expected_exports,
            "actual required native exports",
        )
        sources = {
            path.split("/csrc/", 1)[-1]: digest
            for path, digest in provider["sources"].items()
        }
        require(
            len(sources) == (3 if name == "gemm" else 5),
            "complete native translation-unit source set",
        )
        if name == "gemm":
            require(
                sources["bmm_fp8.cu"]
                == "ac00c1cdba76e653ead25a77d2b1ca9d07f35a89f65221334fbbab1a695a4709"
                and sources["flashinfer_gemm_binding.cu"]
                == "4cfdde4e2d7928ffa442d4a386c8f545de61d68820a6d01e4bf78c1b4a490e8e",
                "qualified FP8 native port",
            )
        commands = provider["compile_commands"]
        require(len(commands) == len(sources), "all native compile commands preserved")
        import shlex

        compiled_sources = []
        for row in commands:
            path = Path(row["file"])
            require(
                str(path).startswith("/study/input/current/source/csrc/"),
                "actual current-source translation-unit path",
            )
            relative = str(path).split("/csrc/", 1)[1]
            require(
                relative in sources, "compiled source belongs to qualified provider"
            )
            compiled_sources.append(relative)
            argv = shlex.split(row["command"])
            require(
                argv.count("-c") == 1 and argv[argv.index("-c") + 1] == str(path),
                "compile command names its recorded translation unit",
            )
            compiler = Path(argv[0]).name
            if path.suffix == ".cu":
                require(
                    compiler == "nvcc"
                    and "-gencode=arch=compute_90a,code=sm_90a" in argv,
                    "current-source H100 CUDA compilation",
                )
            elif path.suffix == ".cpp":
                require(
                    compiler in {"c++", "g++", "clang++"}
                    and not any(
                        arg.startswith(("-gencode", "--generate-code")) for arg in argv
                    ),
                    "current-source host C++ compilation",
                )
            else:
                raise ValueError("unexpected native translation-unit type")
        require(
            len(set(compiled_sources)) == len(sources)
            and set(compiled_sources) == set(sources),
            "complete one-to-one native translation-unit compilation",
        )
        identities[name] = {
            "library_sha256": provider["library_sha256"],
            "ninja_sha256": provider["ninja_sha256"],
            "sources": sources,
            "compile_commands": commands,
            "exports_checked": sorted(expected_exports),
        }
    return identities


def validate_bootstrap(bootstrap, data, raw_path, source_root):
    require(
        bootstrap["status"] == "passed"
        and bootstrap["native_identity_passthrough"] is True,
        "same unwrapped native bootstrap provider",
    )
    require(
        bootstrap["pid"] == data["pid"]
        and bootstrap["result_sha256"] == sha256(raw_path),
        "same process and exact worker result",
    )
    require(
        bootstrap["worker_sha256"]
        == data["worker_sha256"]
        == sha256(
            source_root / "test/manual/kernels/bench_flashinfer_objective_values.py"
        ),
        "actual frozen worker source",
    )
    argv = bootstrap["worker_args"]
    for flag, value in (
        ("--mode", data["mode"]),
        ("--block", str(data["block"])),
        ("--cell", data["cell"]),
        ("--capture-order", data["capture_order"]),
    ):
        require(
            argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
            "actual bootstrap worker treatment arguments",
        )
    require(
        Path(argv[argv.index("--output") + 1]).name == raw_path.name,
        "actual bootstrap worker result path",
    )
    require(
        bootstrap["helper_sha256"]
        == "b8d0f6bbc85eaca0e3ef1419ab99a97d5340c96b3da382cba81d7896768287ba",
        "qualified native bootstrap helper",
    )
    source = bootstrap["source"]
    require(
        source["source_commit"] == "27c62fdf0ab1cd4014df8a15b66a975541fda83c"
        and source["variant"] == "fp8-runtime-callback"
        and source["manifest_sha256"]
        == "9a4c51562d7410d785cc544c004bc4f92486f40311cb5c7a7328c2c1213160d5",
        "qualified current-source manifest",
    )
    require(
        source["source_sha256"]["include/flashinfer/gemm/bmm_fp8.cuh"]
        == "5a35da070efbc028e565721110e059c5b143d6a49697995e3c70fdae258eae72",
        "qualified actual FP8 native header",
    )
    require(
        all(
            source["source_sha256"][path] == digest
            for path, digest in RUNTIME_SHA256.items()
        ),
        "unchanged actual source/timer/cache/FP8 implementation",
    )
    imports = bootstrap["import"]["source_modules"]
    for module, path in (
        ("flashinfer.autotuner.autotuner", "flashinfer/autotuner/autotuner.py"),
        ("flashinfer.autotune_cache", "flashinfer/autotune_cache.py"),
        ("flashinfer.gemm.gemm_base", "flashinfer/gemm/gemm_base.py"),
    ):
        require(
            imports[module]["sha256"] == RUNTIME_SHA256[path]
            and imports[module]["path"].startswith("/study/input/current/source/"),
            "actual Python source import",
        )
    require(
        bootstrap["import"]["flashinfer_runtime_version"] == "0.0.0+unknown"
        and bootstrap["import"]["flashinfer_runtime_commit"] == "unknown",
        "truthful source package metadata",
    )
    return {
        "source_sha256": source["source_sha256"],
        "imports_sha256": hashlib.sha256(
            json.dumps(
                {name: row["sha256"] for name, row in imports.items()}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "native_providers": normalize_provider_identity(bootstrap),
    }


def policy_for(objective):
    return (
        None
        if objective == "auto"
        else {
            "execution_mode": "cuda_graph",
            "cold_l2": None,
            "_timer": "auto",
            "graph_timing": None,
        }
    )


def validate_bare(state, label):
    require(
        state["active_v2"] is False
        and state["is_tuning_mode"] is False
        and state["tuning_contexts"] == 0
        and state["measurement_policy"] is None
        and state["managed_store"] is False
        and state["on_caller_stream"] is True,
        label + ": policy/tuning/store/stream restored",
    )


def validate_tensor(metadata, shape, dtype, label):
    require(
        metadata["shape"] == shape
        and metadata["dtype"] == dtype
        and metadata["device"] == "cuda:0",
        label + ": actual tensor shape/dtype/device",
    )
    pointer = metadata["data_ptr"]
    require(
        type(pointer) is int
        and pointer > 0
        and metadata["alignment_mod256"] == pointer % 256
        and metadata["alignment_mod4096"] == pointer % 4096,
        label + ": actual pointer/alignment",
    )
    require(
        len(metadata["stride"]) == len(shape)
        and all(type(v) is int and v >= 0 for v in metadata["stride"]),
        label + ": tensor strides",
    )


def metadata_without_addresses(metadata):
    return {
        k: v
        for k, v in metadata.items()
        if k not in ("data_ptr", "alignment_mod256", "alignment_mod4096")
    }


def parse_file_key(text):
    """Read the real key repr without evaluating Torch attributes or code."""
    import ast

    allowed = {"torch.float8_e4m3fn", "torch.bfloat16"}

    def convert(node):
        if isinstance(node, ast.Tuple):
            return tuple(convert(value) for value in node.elts)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) is int
        ):
            return -node.operand.value
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and "torch." + node.attr in allowed
        ):
            return "torch." + node.attr
        raise ValueError("unsupported actual cache-key representation")

    key = convert(ast.parse(text, mode="eval").body)
    require(
        len(key) == 4
        and key[3] == ("torch.float8_e4m3fn", "torch.float8_e4m3fn", "torch.bfloat16"),
        "actual dtype-bearing native cache key",
    )
    return key


def validate_profiles(data):
    startup, objective = data["startup"], data["objective"]
    require(
        startup["profile_attempts"] == 81 and len(startup["profiles"]) == 81,
        "all81actual startup measurements",
    )
    require(startup["started_ns"] < startup["ended_ns"], "startup chronology")
    rows = startup["profiles"]
    require(
        [row["index"] for row in rows] == list(range(81)),
        "complete profiling attempt sequence",
    )
    require(
        [row["m"] for row in rows]
        == [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 144) for _ in range(9)],
        "nine unmodified actual profile endpoints",
    )
    require(
        set(startup["candidate_order"])
        == {str(m) for m in (1, 2, 4, 8, 16, 32, 64, 128, 144)},
        "all endpoint-specific candidate pools",
    )
    groups = []
    for offset in range(0, 81, 9):
        group = rows[offset : offset + 9]
        tactics = [tactic_key(row["tactic"]) for row in group]
        require(
            tactics[0] == "-1" and tactics.count("-1") == 1 and len(set(tactics)) == 9,
            "default and eight unique ordered explicit candidates",
        )
        require(
            [row["tactic"] for row in group]
            == startup["candidate_order"][str(group[0]["m"])],
            "original ordered candidate pool at each endpoint",
        )
        for row in group:
            require(
                row["error"] is None,
                "preserved profiling failure prevents performance qualification",
            )
            positive(row["elapsed_ms"], "positive finite actual profile duration")
            require(
                row["fingerprinted"] == (data["mode"] == "admission"),
                "admission-only byte observation",
            )
            require(
                ("fingerprints" in row) == (data["mode"] == "admission"),
                "no hidden cohort synthetic byte fingerprints",
            )
            settings = row["settings"]
            require(
                {
                    key: settings[key]
                    for key in (
                        "warmup",
                        "repeat",
                        "stream_delay_micro_secs",
                        "cuda_graph_delay_micro_secs",
                        "use_global_timer",
                    )
                }
                == {
                    "warmup": 3,
                    "repeat": 10,
                    "stream_delay_micro_secs": 5000,
                    "cuda_graph_delay_micro_secs": 100,
                    "use_global_timer": False,
                },
                "unchanged actual profiler settings",
            )
            require(
                settings["measurement_policy"] == policy_for(objective)
                and settings["effective_policy_timer"] in (None, "events"),
                "actual objective policy and configured timer",
            )
            require(
                settings["config"]
                == {
                    "use_cuda_graph": objective == "graph",
                    "use_cold_l2_cache": False,
                    "use_cuda_graph_individual_timing": False,
                    "cuda_graph_profile_replays": 1,
                    "profiling_repeat": None,
                    "effective_profiling_repeat": 10,
                },
                "only execution objective differs",
            )
            inputs = row["metadata"]
            require(len(inputs) == 6, "actual six public runner inputs")
            validate_tensor(
                inputs[0],
                [1, row["m"], 1024],
                "torch.float8_e4m3fn",
                "profile activation",
            )
            validate_tensor(
                inputs[1], [1, 1024, 4096], "torch.float8_e4m3fn", "profile weight"
            )
            for index in (2, 3):
                validate_tensor(inputs[index], [1], "torch.float32", "profile scale")
            validate_tensor(
                inputs[4], [1, row["m"], 4096], "torch.bfloat16", "profile output"
            )
            require(
                inputs[5]["shape"] == [41943040]
                and inputs[5]["dtype"] == "torch.uint8",
                "fixed actual workspace contract",
            )
            key = parse_file_key(row["key"])
            require(
                key[0] == "fp8_gemm"
                and key[1] == "CublasFp8GemmRunner"
                and key[2][0] == (1, 256 if row["m"] == 144 else row["m"], 1024),
                "actual profile endpoint versus normalized cache bucket",
            )
        fastest = min(group, key=lambda row: row["elapsed_ms"])
        explicit = min(group[1:], key=lambda row: row["elapsed_ms"])
        ranked = sorted(group, key=lambda row: row["elapsed_ms"])
        groups.append(
            {
                "actual_m": group[0]["m"],
                "key": group[0]["key"],
                "default_ms": group[0]["elapsed_ms"],
                "best_explicit_ms": explicit["elapsed_ms"],
                "best_explicit_tactic": explicit["tactic"],
                "best_explicit_over_default": explicit["elapsed_ms"]
                / group[0]["elapsed_ms"],
                "profile_winner": fastest["tactic"],
                "profile_winner_ms": fastest["elapsed_ms"],
                "runner_up_ms": ranked[1]["elapsed_ms"],
                "winner_over_runner_up": fastest["elapsed_ms"]
                / ranked[1]["elapsed_ms"],
                "winner_margin_fraction_of_runner_up": 1
                - fastest["elapsed_ms"] / ranked[1]["elapsed_ms"],
                "candidate_order": [row["tactic"] for row in group],
            }
        )
    require(
        data["natural"]["tactic"] == groups[-1]["profile_winner"]
        and data["natural"]["key"] == groups[-1]["key"]
        and data["natural"]["cache_hit"] is True
        and data["natural"]["lookup_tier"] == "memory",
        "natural runtime winner is actual first finite profile minimum and real memory hit",
    )
    calls = startup["initializer_calls"]
    require(
        len(calls) == 9
        and [row["shape"] for row in calls]
        == [[1, m, 1024] for m in (1, 2, 4, 8, 16, 32, 64, 128, 144)],
        "nine original activation draws",
    )
    for index, row in enumerate(calls):
        targeted = index == 8
        require(
            row["dtype"] == "torch.float8_e4m3fn"
            and row["device"] == "cuda:0"
            and row["targeted"] is targeted
            and row["cloned"] is targeted
            and row["copied"] is targeted,
            "exact-M144 matched allocation/copy control",
        )
        require(
            ("generated_sha256" in row) == (data["mode"] == "admission")
            and ("returned_sha256" in row) == (data["mode"] == "admission"),
            "admission hashes absent from timing initializer",
        )
    require(
        startup["incidental_legacy_entries"] == (9 if objective == "auto" else 0),
        "incidental legacy save separated from memory-only graph policy",
    )
    records = startup["scope_records"]
    stages = {"startup", "natural_lookup", "winner_capture", "default_capture"}
    if data["mode"] == "admission":
        stages.add("admission_explicit")
    require(
        collections.Counter((row["stage"], row["event"]) for row in records)
        == collections.Counter(
            (stage, event)
            for stage in stages
            for event in ("enter", "before_exit", "after_exit")
        ),
        "complete policy scope evidence",
    )
    for row in records:
        if row["event"] == "after_exit":
            if row["stage"] != "startup":
                validate_bare(row, "scope exit")
            require(
                row["active_v2"] is False
                and row["measurement_policy"] is None
                and row["managed_store"] is False,
                "measurement policy restored on startup/capture exit",
            )
        else:
            require(
                row["active_v2"] is True
                and row["measurement_policy"] == policy_for(objective)
                and row["managed_store"] is False,
                "matching memory-only policy scope",
            )
            require(
                row["is_tuning_mode"] is (row["stage"] == "startup"),
                "only actual startup can tune",
            )
            require(
                row["winner_identity"]
                == (
                    "None"
                    if objective == "auto"
                    else "('__policy__', ('measure_execution_mode', 'cuda_graph'))"
                ),
                "policy-specific winner bank identity",
            )
    return groups


def correctness_index(data):
    rows = data["correctness"]
    require(
        len({row["check"] for row in rows}) == len(rows),
        "unique numerical check labels",
    )
    require(
        all(
            row["passed"] is True
            and math.isfinite(row["max_abs_error"])
            and row["max_abs_error"] >= 0
            for row in rows
        ),
        "all numerical checks passed",
    )
    for row in rows:
        if row["exact_zero_required"]:
            require(row["max_abs_error"] == 0, "exact-zero changed-input control")
    return {row["check"]: row for row in rows}


def check_owner(checks, label, graph, *, zero=False):
    row = checks[label]
    require(
        row["graph_id"] == graph["id"]
        and row["output_data_ptr"] == graph["output"]["data_ptr"]
        and row["exact_zero_required"] is zero,
        "numerical guard belongs to original graph output",
    )


def validate_graphs(data, checks):
    graph_rows = data["graphs"]
    expected = list(
        GRAPHS if data["capture_order"] == "winner-first" else reversed(GRAPHS)
    )
    if data["mode"] == "admission":
        expected.append("admission_explicit")
    require(
        [row["id"] for row in graph_rows] == expected,
        "independent scheduled graph capture order",
    )
    graphs = {row["id"]: row for row in graph_rows}
    require(
        len({row["graph_object_id"] for row in graph_rows}) == len(graph_rows)
        and len({row["output"]["data_ptr"] for row in graph_rows}) == len(graph_rows),
        "distinct retained graph objects and output owners",
    )
    for name, graph in graphs.items():
        require(
            graph["capture_started_ns"] < graph["capture_ended_ns"],
            "graph capture chronology",
        )
        inputs = graph["inputs"]
        require(len(inputs) == 6, "retained six input/workspace owners")
        validate_tensor(
            inputs[0], [1, 144, 1024], "torch.float8_e4m3fn", "capture activation"
        )
        validate_tensor(
            inputs[1], [1, 1024, 4096], "torch.float8_e4m3fn", "capture weight"
        )
        require(
            inputs[0]["stride"][-2:] == [1024, 1]
            and inputs[1]["stride"][-2:] == [1, 1024],
            "actual row/column-major public operands",
        )
        for index in (2, 3):
            validate_tensor(inputs[index], [1], "torch.float32", "capture static scale")
        validate_tensor(
            inputs[4], [1, 144, 4096], "torch.bfloat16", "capture native output"
        )
        validate_tensor(
            graph["output"], [144, 4096], "torch.bfloat16", "full-layer output"
        )
        validate_tensor(
            graph["linear_input"],
            [144, 1024],
            "torch.bfloat16",
            "full-layer BF16 input",
        )
        require(
            inputs[4]["data_ptr"] == graph["output"]["data_ptr"],
            "facade output aliases exact native graph-owned output",
        )
        require(
            len(graph["read_operand_hashes"]) == 4
            and graph["read_operand_hashes"][0]
            == data["fixture"]["quantized_input_sha256"],
            "actual public capture quantized operand",
        )
        expected_tactic = (
            -1
            if name == "default"
            else data["natural"]["tactic"]
            if name == "winner"
            else data["known_explicit_control"]["tactic"]
        )
        require(
            graph["tactic"] == expected_tactic
            and graph["lookup_tier"] == ("miss" if name == "default" else "memory"),
            "actual selected/default graph submission",
        )
        selection = graph["selection"]
        require(
            selection["capturing"] is True
            and selection["cache_hit"] is (name != "default")
            and selection["tactic"] == expected_tactic
            and selection["lookup_tier"] == graph["lookup_tier"]
            and selection["stage"] == name + "_capture",
            "real capture hit versus removed-cache miss",
        )
        require(
            selection["active_v2"] is True
            and selection["is_tuning_mode"] is False
            and selection["measurement_policy"] == policy_for(data["objective"])
            and selection["managed_store"] is False,
            "capture uses matching replay-only memory policy",
        )
        for suffix in ("poisoned", "restored_input"):
            check_owner(checks, name + "_" + suffix, graph)
        check_owner(checks, name + "_changed_input", graph, zero=True)
        check_owner(checks, "final_" + name, graph)
    require(
        graphs["winner"]["read_operand_hashes"]
        == graphs["default"]["read_operand_hashes"],
        "same actual serving operands across independent graphs",
    )
    expected_misses = 2 if data["mode"] == "admission" else 1
    require(
        len(data["miss_scopes"]) == expected_misses,
        "actual reversible default/known-control cache scopes",
    )
    for row in data["miss_scopes"]:
        require(
            row["restored"] is True
            and row["winner_entries_before"] == 9
            and row["winner_entries_after_removal"]
            == row["file_entries_before"]
            == row["file_entries_after_removal"]
            == 0
            and row["cache_sha256_before"]
            == row["cache_sha256_after"]
            == data["startup"]["cache_sha256"],
            "real cache removal and full restoration",
        )
    return graphs


def validate_timing(data, graphs, checks):
    if data["mode"] == "admission":
        require(
            data["rounds"] == []
            and "selected_over_default_graph_event_median" not in data,
            "admissions are not performance trials",
        )
        return None
    warmups = data["initial_warmups"]
    require(
        [row["graph_id"] for row in warmups] == list(GRAPHS)
        and all(row["actual_ms"] >= 500 and row["calls"] >= 5120 for row in warmups),
        "initial500ms per-graph warmup",
    )
    require(
        data["round_schedule"] == expected_round_orders(data["block"]),
        "frozen paired replay order",
    )
    rounds = data["rounds"]
    require(len(rounds) == 24, "all24paired rounds retained")
    previous_end = max(graph["capture_ended_ns"] for graph in graphs.values())
    ratios, wall_ratios, samples, order_strata = (
        [],
        [],
        [],
        collections.defaultdict(list),
    )
    query_counts = collections.Counter()
    graph_latencies = {name: [] for name in GRAPHS}
    for index, group in enumerate(rounds):
        require(
            group["index"] == index
            and group["order"] == expected_round_orders(data["block"])[index],
            "scheduled complete round",
        )
        rows = group["measurements"]
        require(
            [row["graph_id"] for row in rows] == group["order"],
            "both graphs measured exactly once in scheduled order",
        )
        for row in rows:
            name = row["graph_id"]
            graph = graphs[name]
            require(
                row["graph_object_id"] == graph["graph_object_id"]
                and row["output_data_ptr"] == graph["output"]["data_ptr"],
                "sample uses original retained object/output",
            )
            require(
                row["replays"] == 16
                and row["prequeue_cycles"] == 10_000_000
                and row["rewarm"]["actual_ms"] >= 20
                and row["rewarm"]["calls"] >= 5120,
                "fixed replay/depth/rewarm recipe",
            )
            require(
                row["profile_attempts_before"] == row["profile_attempts_after"] == 81,
                "no profiling inside timing",
            )
            require(
                row["start_query_error"] is None
                and type(row["start_event_complete_after_enqueue"]) is bool,
                "actual pre-sync start query outcome",
            )
            pending = not row["start_event_complete_after_enqueue"]
            require(
                row["queue_evidence"]
                == (
                    "start_pending_after_enqueue"
                    if pending
                    else "start_complete_inconclusive"
                ),
                "honest queued/inconclusive classification",
            )
            query_counts[row["queue_evidence"]] += 1
            require(
                previous_end
                < row["wall_started_ns"]
                <= row["enqueue_started_ns"]
                < row["enqueue_ended_ns"]
                <= row["ended_ns"],
                "nonoverlapping postcapture timing chronology",
            )
            previous_end = row["ended_ns"]
            batch = positive(
                row["gpu_event_batch_ms"], "positive finite GPU event batch"
            )
            per_replay = positive(
                row["gpu_event_per_replay_ms"],
                "positive finite per-replay event duration",
            )
            require(
                math.isclose(per_replay, batch / 16, rel_tol=1e-12, abs_tol=1e-15),
                "correct16-replay normalization",
            )
            wall = positive(
                row["wall_including_prequeue_ms"], "positive queued wall diagnostic"
            )
            require(
                math.isclose(
                    wall,
                    (row["ended_ns"] - row["wall_started_ns"]) / 1e6,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and row["wall_diagnostic_only"] is True
                and row["enqueue_span_includes_start_query"] is True,
                "wall scope includes deliberate delay and query",
            )
            require(
                math.isclose(
                    row["enqueue_cpu_ms"],
                    (row["enqueue_ended_ns"] - row["enqueue_started_ns"]) / 1e6,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "actual enqueue diagnostic scope",
            )
            require(
                row["correctness_check"] == f"round_{index:02d}_{name}",
                "sample-specific numerical check",
            )
            check_owner(checks, row["correctness_check"], graph)
            graph_latencies[name].append(per_replay)
            samples.append(row)
        paired = {row["graph_id"]: row for row in rows}
        ratio = (
            paired["winner"]["gpu_event_per_replay_ms"]
            / paired["default"]["gpu_event_per_replay_ms"]
        )
        ratios.append(ratio)
        wall_ratios.append(
            paired["winner"]["wall_including_prequeue_ms"]
            / paired["default"]["wall_including_prequeue_ms"]
        )
        order_strata[group["order"][0]].append(ratio)
    event_ratio = statistics.median(ratios)
    require(
        math.isclose(
            data["selected_over_default_graph_event_median"],
            event_ratio,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "worker primary estimate independently recomputed",
    )
    return {
        "event_ratio": event_ratio,
        "wall_ratio": statistics.median(wall_ratios),
        "within_round_event_ratios": ratios,
        "within_round_wall_ratios_diagnostic_only": wall_ratios,
        "event_per_replay_ms": {
            name: describe(values) for name, values in graph_latencies.items()
        },
        "order_strata_event_ratios": {
            name + "_first": describe(values) for name, values in order_strata.items()
        },
        "query_outcomes_all_retained": dict(query_counts),
        "samples_retained": len(samples),
        "last_timing_ended_ns": previous_end,
    }


def validate_acceptance(row, graph, checks, earliest):
    require(
        earliest < row["started_ns"] < row["ended_ns"],
        "raw native acceptance strictly outside and after primary timing",
    )
    require(
        row["tactic"] == graph["tactic"]
        and row["captured_inputs"] == graph["inputs"]
        and row["pointers_unchanged"] is True
        and row["original_graph_replayed"] is True,
        "exact original captured inputs/descriptor accepted and original graph replayed",
    )
    if graph["tactic"] == -1:
        require(
            row["route"] == "native_default"
            and row["used_descriptor"] is None
            and row["descriptor_acceptance"] == "not_applicable_default_route",
            "default route does not claim explicit acceptance",
        )
    else:
        require(
            row["route"] == "native_descriptor"
            and row["used_descriptor"] == 1
            and row["descriptor_acceptance"] == "accepted",
            "actual submitted selected descriptor accepted",
        )
    require(
        row["graph_id"] == graph["id"]
        and row["graph_object_id"] == graph["graph_object_id"]
        and row["output_data_ptr"] == graph["output"]["data_ptr"],
        "native acceptance bound to original measured graph",
    )
    for key in ("raw_numerical_check", "original_graph_check"):
        check = checks[row[key]]
        require(
            check["graph_id"] == graph["id"]
            and check["output_data_ptr"] == graph["output"]["data_ptr"]
            and check["exact_zero_required"] is False,
            "native and original graph correctness on same owned output",
        )
    return {
        "route": row["route"],
        "tactic": row["tactic"],
        "used_descriptor": row["used_descriptor"],
        "descriptor_acceptance": row["descriptor_acceptance"],
        "limit": "Posttiming viability on retained actual capture inputs; not direct observation of native return during earlier capture.",
    }


def validate_trace(data, graphs, checks, raw_root, expected_name):
    evidence = data["trace"]
    before, after = data["pre_trace_checks"], data["final"]
    require(
        data["post_timing_acceptance"]["ended_ns"]
        < before["ended_ns"]
        < evidence["started_ns"]
        < evidence["ended_ns"]
        < after["ended_ns"],
        "trace strictly after primary/native/final numerical guards",
    )
    path = raw_root / Path(evidence["path"]).name
    require(
        path.name == expected_name
        and path.is_file()
        and sha256(path) == evidence["sha256"]
        and evidence["calls_per_graph"] == 3,
        "exact raw posttiming trace file",
    )
    observed = parse_retained_trace(load(path))
    require(
        [row["graph_id"] for row in evidence["graphs"]] == list(GRAPHS),
        "both original graph trace bindings",
    )
    for row in evidence["graphs"]:
        name, graph = row["graph_id"], graphs[row["graph_id"]]
        require(
            row["graph_object_id"] == graph["graph_object_id"]
            and row["output_data_ptr"] == graph["output"]["data_ptr"],
            "traced object is the original measured graph",
        )
        sequence = observed[name]["ordered_launch_signature"] * 3
        unique = []
        indices = []
        for signature in sequence:
            if signature not in unique:
                unique.append(signature)
            indices.append(unique.index(signature))
        require(
            row["graph_launch_events"] == 3
            and row["kernel_events"] == len(sequence)
            and row["signatures"] == unique
            and row["kernel_sequence"] == indices,
            "producer trace summary independently verified",
        )
        check_owner(checks, "traced_" + name, graph)
    return {
        "sha256": evidence["sha256"],
        "original_graphs": observed,
        "timing_inference": False,
    }


def validate_worker(data, assignment, args, plan, outer):
    mode = "admission" if args.admission else "timing"
    require(
        data["schema_version"] == 1
        and data["status"] == "success"
        and data["mode"] == mode,
        "complete successful worker outcome",
    )
    require(
        data["block"] == data["trial"] == assignment["block"]
        and data["cell"] == assignment["cell"]
        and data["capture_order"] == assignment["capture_order"]
        and data["objective"] == assignment["objective"]
        and data["values"] == assignment["values"],
        "actual assigned fresh-process treatment",
    )
    require(
        data["assignment"] == assignment
        and data["round_schedule"] == expected_round_orders(assignment["block"]),
        "actual frozen worker schedule",
    )
    require(
        data["plan_sha256"] == sha256(args.plan)
        and data["schedule_sha256"] == sha256(args.schedule),
        "actual worker plan/schedule source",
    )
    require(
        not data["protocol_errors"]
        and not data["forbidden_profile_attempts"]
        and data.get("pending_sample") is None,
        "no failed/omitted/unfinished measurement",
    )
    require(
        data["initializer_config_restored"] is True
        and data["synthetic_input_references_retained"] is False
        and data["persistence_under_test"] is False
        and data["all_samples_retained"] is True
        and data["queue_query_is_exclusion_criterion"] is False,
        "scoped initializer restoration and all outcomes retained",
    )
    require(
        data["profiling_byte_attestation"]
        == (
            "admission_full_hashes"
            if args.admission
            else "metadata_only_no_synthetic_byte_attestation"
        ),
        "admission and cohort observation scopes separated",
    )
    source = data["source"]
    require(
        source["commit"] == plan["source_contract"]["flashinfer_commit"]
        and source["critical_sha256"] == RUNTIME_SHA256
        and source["bootstrap_helper_sha256"]
        == plan["qualified_source"]["container_preflight_sha256"]
        and source["candidate_manifest_sha256"]
        == plan["qualified_source"]["candidate_manifest_sha256"]
        and source["native_identity_passthrough"] is True,
        "actual qualified source/native contract",
    )
    require(
        all(
            outer["source_sha256"][name] == digest == sha256(args.source_root / name)
            for name, digest in source["sglang_source_sha256"].items()
        ),
        "unchanged actual SGLang fixture production source",
    )
    wrapper = "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
    require(
        source["startup_wrapper_sha256"]
        == outer["source_sha256"][wrapper]
        == sha256(args.source_root / wrapper),
        "actual SGLang startup wrapper",
    )
    fixture = data["fixture"]
    require(
        fixture["manifest_sha256"]
        == "e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1"
        and fixture["tensor_file_sha256"]
        == "c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b"
        and fixture["shape"] == [144, 4096, 1024],
        "same exact frozen fixture",
    )
    require(
        fixture["tensors"] == args.fixture_manifest_data["tensors"]
        and fixture["quantized_input_sha256"]
        == args.fixture_manifest_data["quantization"]["independent_input_fp8_sha256"],
        "actual fixture tensor and serving quantized bytes",
    )
    environment = data["environment"]
    require(
        environment["torch"] == "2.14.1+cu130"
        and environment["cuda"] == "13.0"
        and environment["gpu"] == "NVIDIA H100 NVL"
        and environment["gpu_uuid"].lower().removeprefix("gpu-")
        == plan["environment"]["uuid"],
        "pinned actual GPU/Torch stack",
    )
    require(
        environment["inference_mode"] is True
        and environment["grad_enabled"] is False
        and environment["allow_tf32"] is False
        and environment["flashinfer_version"] == "0.0.0+unknown",
        "actual inference/numerical/source metadata",
    )
    layer = data["actual_layer"]
    require(
        {key: layer[key] for key in ("function", "custom_op", "quantizer", "backend")}
        == {
            "function": "apply_fp8_linear_bmm_flashinfer",
            "custom_op": "flashinfer_bmm_fp8",
            "quantizer": "static_quant_fp8",
            "backend": "cublas",
        },
        "actual original full-layer public facade",
    )
    require(
        layer["inference_mode"] is True
        and layer["custom_op_direct_wrapper"] is True
        and not any(
            layer[key]
            for key in (
                "grad_enabled",
                "dispatch_modes",
                "function_modes",
                "torch_profiler_enabled",
                "compiling",
                "kernel_api_log_level",
            )
        ),
        "normal uninstrumented inference custom-op path",
    )
    require(
        data["delay_helper"]["prime_cycles"] == 10_000_000
        and data["delay_helper"]["primed_before_startup"] is True,
        "same real delay helper primed before startup",
    )
    profiles = validate_profiles(data)
    checks = correctness_index(data)
    graphs = validate_graphs(data, checks)
    timings = validate_timing(data, graphs, checks)
    earliest = (
        max(
            graph["capture_ended_ns"]
            for graph in graphs.values()
            if graph["id"] in GRAPHS
        )
        if args.admission
        else timings["last_timing_ended_ns"]
    )
    acceptance = validate_acceptance(
        data["post_timing_acceptance"], graphs["winner"], checks, earliest
    )
    if args.admission:
        control = data["known_explicit_control"]
        require(
            control["accepted"] is True
            and tactic_key(control["tactic"]) != "-1"
            and control["attempts"]
            and control["attempts"][-1]["used_descriptor"] == 1,
            "separate known explicit-control admission",
        )
        validate_acceptance(
            control["actual_acceptance"],
            graphs["admission_explicit"],
            checks,
            graphs["admission_explicit"]["capture_ended_ns"],
        )
    memory_hash = data["startup"]["cache_state_sha256"]
    require(
        data["pre_capture_cache_state_sha256"] == memory_hash,
        "same live winner banks before capture",
    )
    for label in ("pre_trace_checks", "final"):
        state = data[label]
        validate_bare(state, label)
        require(
            state["profile_attempts"] == 81
            and not state["forbidden_profile_attempts"]
            and state["cache_sha256"] == data["startup"]["cache_sha256"]
            and state["cache_state_sha256"] == memory_hash
            and state["cache_unchanged"] is True
            and state["operands_unchanged"] is True,
            "final live cache/operand/profile preservation",
        )
        for name, graph in graphs.items():
            check_owner(
                checks,
                ("pre_trace_" if label == "pre_trace_checks" else "final_") + name,
                graph,
            )
    raw_name = (
        f"admission-{assignment['cell']}.json"
        if args.admission
        else assignment["raw_name"]
    )
    trace_name = Path(raw_name).stem + ".trace.json"
    trace = validate_trace(data, graphs, checks, args.raw_root, trace_name)
    result = {
        "block": data["block"],
        "cell": data["cell"],
        "capture_order": data["capture_order"],
        "selected_default": data["natural"]["tactic"] == -1,
        "selected_tactic": data["natural"]["tactic"],
        "profile_margins_by_actual_m": profiles,
        "native_acceptance": acceptance,
        "trace": trace,
    }
    if timings:
        result.update(timings)
    return result


def validate_comparability(raw_workers, *, admission):
    first = raw_workers[0]
    common_sources = first["source"]
    candidate_order = first["startup"]["candidate_order"]
    actual_serving_hashes = {
        row["id"]: row["read_operand_hashes"]
        for row in first["graphs"]
        if row["id"] in GRAPHS
    }
    for data in raw_workers:
        require(
            data["source"] == common_sources,
            "same qualified runtime/SGLang sources across all treatments",
        )
        require(
            data["startup"]["candidate_order"] == candidate_order,
            "same ordered candidate pool across all treatments",
        )
        require(
            data["delay_helper"] == first["delay_helper"],
            "same prepared real delay helper",
        )
        require(
            {
                row["id"]: row["read_operand_hashes"]
                for row in data["graphs"]
                if row["id"] in GRAPHS
            }
            == actual_serving_hashes,
            "same actual serving operands across all cells",
        )
        require(
            data["fixture"] == first["fixture"], "same frozen fixture across all cells"
        )
        require(
            data["startup"]["rng_before"] == first["startup"]["rng_before"]
            and data["startup"]["rng_after"] == first["startup"]["rng_after"],
            "same original RNG draws and final RNG state",
        )
        for left, right in zip(
            first["startup"]["profiles"], data["startup"]["profiles"]
        ):
            require(
                left["m"] == right["m"]
                and left["tactic"] == right["tactic"]
                and left["key"] == right["key"],
                "same actual profile/order/cache-domain inputs",
            )
            require(
                [metadata_without_addresses(t) for t in left["metadata"]]
                == [metadata_without_addresses(t) for t in right["metadata"]],
                "same profile tensor shape/stride/dtype/device contract",
            )
    if not admission:
        return {
            "profile_bytes_attested": False,
            "scope": "Cohort metadata/RNG/source comparability only. Separate instrumented admissions establish the controlled initializer mechanism, not cohort byte equality or identical physical history.",
        }
    require(
        len(raw_workers) == 4 and {data["cell"] for data in raw_workers} == set(CELLS),
        "all four independent admission cells",
    )
    by_cell = {data["cell"]: data for data in raw_workers}
    generated = [
        row["generated_sha256"] for row in first["startup"]["initializer_calls"]
    ]
    baseline = by_cell["auto_synthetic"]
    captured_sha = first["fixture"]["quantized_input_sha256"]
    for data in raw_workers:
        calls = data["startup"]["initializer_calls"]
        require(
            [row["generated_sha256"] for row in calls] == generated,
            "identical actual original random draws in all admission cells",
        )
        for index, call in enumerate(calls):
            expected = (
                captured_sha
                if index == 8 and data["values"] == "captured"
                else call["generated_sha256"]
            )
            require(
                call["returned_sha256"] == expected,
                "only captured-values exactM144 activation is replaced",
            )
        for row, base in zip(
            data["startup"]["profiles"], baseline["startup"]["profiles"]
        ):
            hashes = row["fingerprints"]
            require(
                len(hashes) == 4,
                "actual first four profiling read operands fingerprinted",
            )
            expected = list(base["fingerprints"])
            if row["m"] == 144 and data["values"] == "captured":
                expected[0] = captured_sha
            require(
                hashes == expected,
                "only exactM144 profile activation differs; weights/scales/other endpoints unchanged",
            )
    require(
        by_cell["auto_synthetic"]["startup"]["initializer_calls"][-1]["returned_sha256"]
        != captured_sha,
        "admission exercises a real activation-values difference",
    )
    return {
        "profile_bytes_attested": True,
        "all_four_cells": True,
        "original_draws_rng_order_shapes_keys_candidates_preserved": True,
        "only_profile_M144_A_changed": True,
        "profile_weights_scales_unchanged": True,
        "actual_serving_operands_equal": True,
        "limit": "Instrumented disposable admissions, excluded from performance cohort; matching operations do not prove identical physical copy history.",
    }


def validate_outer(args, plan):
    import subprocess

    outer = load(args.raw_root / "manifest.json")
    require(
        outer["schema_version"] == 1
        and outer["plan_sha256"] == sha256(args.plan)
        and outer["schedule_sha256"] == sha256(args.schedule),
        "outer source/plan/schedule binding",
    )
    require(
        outer["fixture_manifest_sha256"]
        == sha256(args.fixture_manifest)
        == "e13a86fab5db5796b22d8dab0f84080b8057b036ca7aef2c7be8832e76505bf1"
        and outer["fixture_tensor_file_sha256"]
        == "c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b",
        "outer exact fixture identity",
    )
    require(
        outer["candidate_manifest_sha256"]
        == plan["qualified_source"]["candidate_manifest_sha256"]
        and outer["helper_sha256"]
        == plan["qualified_source"]["container_preflight_sha256"]
        and outer["private_source_archive_sha256"]
        == "209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91",
        "outer qualified current-source provider identity",
    )
    commit = outer["execution_commit"]
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(c in "0123456789abcdef" for c in commit),
        "frozen execution commit",
    )
    sources = outer["source_sha256"]
    required = {
        BATCH + "/analyze.py",
        BATCH + "/run_current.py",
        BATCH + "/plan.json",
        BATCH + "/schedule.json",
        "test/manual/kernels/bench_flashinfer_objective_values.py",
        "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
        *args.fixture_manifest_data["source"]["sglang_source_sha256"],
    }
    require(
        required <= set(sources),
        "all worker/bootstrap/analyzer/plan/production sources bound",
    )
    for name in sorted(required):
        require(
            sources[name] == sha256(args.source_root / name),
            "actual analysis checkout source: " + name,
        )
        blob = subprocess.run(
            ["git", "-C", str(args.source_root), "show", commit + ":" + name],
            check=True,
            capture_output=True,
        ).stdout
        require(
            hashlib.sha256(blob).hexdigest() == sources[name],
            "committed execution source: " + name,
        )
    gpu = outer["gpu"]
    normalized_gpu = {key.strip(): value.strip() for key, value in gpu.items()}
    require(
        normalized_gpu["name"] == plan["environment"]["gpu"]
        and normalized_gpu["uuid"].lower().removeprefix("gpu-")
        == plan["environment"]["uuid"]
        and normalized_gpu["driver_version"] == plan["environment"]["driver"]
        and float(normalized_gpu["power.limit [W]"].split()[0])
        == plan["environment"]["power_limit_watts"],
        "outer pinned H100 UUID/driver/power",
    )
    image = outer["image"]
    require(
        plan["environment"]["image"] in image["RepoDigests"]
        and image["Architecture"] == "amd64"
        and image["Os"] == "linux"
        and image["Id"] == plan["environment"]["image_id"],
        "actual immutable container image",
    )
    return outer


def analyze(args):
    admission_result = None
    if not args.admission:
        admission_args = copy.copy(args)
        admission_args.admission = True
        admission_result = analyze(admission_args)
    plan, schedule = load(args.plan), load(args.schedule)
    entries = validate_plan(plan, schedule, args.schedule)
    args.fixture_manifest_data = load(args.fixture_manifest)
    outer = validate_outer(args, plan)
    require(args.blocks in (1, 8), "only the complete first-block or full-cohort gate")
    if args.admission:
        entries = [row for row in entries if row["block"] == 0]
    else:
        entries = [row for row in entries if row["block"] < args.blocks]
    if not args.admission:
        expected_raw = {row["raw_name"] for row in entries}
        require(
            {path.name for path in args.raw_root.glob("block*-timing.json")}
            == expected_raw,
            "all and only fixed cohort worker outcomes",
        )
    raw_workers, workers, provider_identities, containers = [], [], [], []
    for assignment in entries:
        name = (
            f"admission-{assignment['cell']}.json"
            if args.admission
            else assignment["raw_name"]
        )
        raw_path = args.raw_root / name
        data = load(raw_path)
        stem = raw_path.stem
        bootstrap_path = raw_path.with_suffix(".bootstrap.json")
        bootstrap = load(bootstrap_path)
        identity = validate_bootstrap(bootstrap, data, raw_path, args.source_root)
        worker = validate_worker(data, assignment, args, plan, outer)
        container_path = args.raw_root / (stem + ".container-id")
        container = container_path.read_text().strip()
        require(
            len(container) == 64 and all(c in "0123456789abcdef" for c in container),
            "actual fresh-container process identity",
        )
        containers.append(container)
        require(
            (args.raw_root / (stem + ".exit-code")).read_text().strip() == "0",
            "actual worker process exited successfully",
        )
        worker.update(
            raw_name=name,
            raw_sha256=sha256(raw_path),
            bootstrap_sha256=sha256(bootstrap_path),
            container_id=container,
        )
        raw_workers.append(data)
        workers.append(worker)
        provider_identities.append(identity)
    require(
        len(set(containers)) == len(containers),
        "every worker in a distinct fresh container",
    )
    require(
        all(identity == provider_identities[0] for identity in provider_identities),
        "same actual loaded native ELF/ninja/translation units/imports across fresh processes",
    )
    comparability = validate_comparability(raw_workers, admission=args.admission)
    reference = raw_workers[0]
    reference_fields = {
        "source": reference["source"],
        "fixture": reference["fixture"],
        "delay_helper": reference["delay_helper"],
        "rng_before": reference["startup"]["rng_before"],
        "rng_after": reference["startup"]["rng_after"],
        "candidate_order": reference["startup"]["candidate_order"],
        "profiles": [
            {
                "m": row["m"],
                "tactic": row["tactic"],
                "key": row["key"],
                "metadata": [metadata_without_addresses(t) for t in row["metadata"]],
            }
            for row in reference["startup"]["profiles"]
        ],
    }
    reference_sha = hashlib.sha256(
        json.dumps(reference_fields, sort_keys=True).encode()
    ).hexdigest()
    output = {
        "schema_version": 1,
        "status": "passed",
        "comparability_reference_sha256": reference_sha,
        "mode": "admission" if args.admission else "timing",
        "plan_sha256": sha256(args.plan),
        "schedule_sha256": sha256(args.schedule),
        "execution_commit": outer["execution_commit"],
        "analysis_python": sys.version,
        "validated_workers": len(workers),
        "independent_blocks": 0 if args.admission else args.blocks,
        "comparability": comparability,
        "workers": workers,
        "provider_identity_sha256": hashlib.sha256(
            json.dumps(provider_identities[0], sort_keys=True).encode()
        ).hexdigest(),
        "admission_limit": "Correctness-only instrumented admissions are not counted as cohort trials.",
    }
    if not args.admission:
        require(
            admission_result["provider_identity_sha256"]
            == output["provider_identity_sha256"]
            and admission_result["comparability_reference_sha256"] == reference_sha,
            "cohort matches separately qualified four-cell admission/source/native/initializer domain",
        )
        require(
            not set(containers)
            & {worker["container_id"] for worker in admission_result["workers"]},
            "admission and timing are separate fresh processes",
        )
        output["admission"] = {
            "validated_workers": 4,
            "raw_sha256": {
                worker["cell"]: worker["raw_sha256"]
                for worker in admission_result["workers"]
            },
            "counted_as_timing_trials": False,
        }
        output["statistics"] = summarize_statistics(workers)
    return output


def self_test():
    import importlib.util

    path = Path(__file__).with_name("analyzer_controls.py")
    spec = importlib.util.spec_from_file_location(
        "objective_values_analyzer_controls", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(sys.modules[__name__])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[6]
    )
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("plan.json")
    )
    parser.add_argument(
        "--schedule", type=Path, default=Path(__file__).with_name("schedule.json")
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=Path(__file__).with_name("fixture-manifest.json"),
    )
    parser.add_argument("--blocks", type=int, choices=(1, 8), default=8)
    parser.add_argument("--admission", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = self_test()
    else:
        parser.error("--raw-root is required") if args.raw_root is None else None
        try:
            result = analyze(args)
        except Exception as error:
            result = {
                "schema_version": 1,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "available_outcomes": [],
            }
            for path in sorted(args.raw_root.glob("*.json")):
                if path.name.startswith(
                    ("block", "admission-")
                ) and not path.name.endswith((".bootstrap.json", ".trace.json")):
                    try:
                        data = load(path)
                        result["available_outcomes"].append(
                            {
                                "file": path.name,
                                "sha256": sha256(path),
                                "status": data.get("status"),
                                "error": data.get("error"),
                                "cell": data.get("cell"),
                                "block": data.get("block"),
                                "natural": data.get("natural"),
                                "profile_attempts": data.get("startup", {}).get(
                                    "profile_attempts"
                                ),
                                "rounds_retained": len(data.get("rounds", [])),
                            }
                        )
                    except Exception:
                        result["available_outcomes"].append(
                            {
                                "file": path.name,
                                "sha256": sha256(path),
                                "status": "unparsable",
                            }
                        )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
