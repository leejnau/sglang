"""CPU-only schema/statistics fault controls; no simulated GPU qualification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def run(a):
    root = Path(__file__).resolve().parents[6]
    batch = Path(__file__).resolve().parent
    plan, schedule = a.load(batch / "plan.json"), a.load(batch / "schedule.json")
    fixture_path = batch / "fixture-manifest.json"
    if not fixture_path.exists():
        fixture_path = (
            batch.parent
            / "20261024T090000Z-h100-replay-calibration/fixture-manifest.json"
        )
    fixture = a.load(fixture_path)
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, call):
        try:
            call()
        except (ValueError, KeyError, TypeError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError(name)

    a.validate_plan(plan, schedule, batch / "schedule.json")
    checks.append("actual_fixed_plan_and_Williams_capture_balance")
    check(
        "exact_zero_of_eight_upper",
        math.isclose(
            a.exact_binomial_interval(0, 8)["one_sided_upper"],
            1 - 0.05 ** (1 / 8),
            abs_tol=1e-14,
        ),
    )
    check(
        "exact_binomial_symmetry",
        math.isclose(
            a.exact_binomial_interval(2, 8)["two_sided_clopper_pearson"][0],
            1 - a.exact_binomial_interval(6, 8)["two_sided_clopper_pearson"][1],
            abs_tol=1e-14,
        ),
    )
    check("linear_type7_quantile", a.quantile([0, 10, 20, 30], 0.125) == 3.75)
    workers = [
        {
            "block": b,
            "cell": cell,
            "event_ratio": value,
            "wall_ratio": value,
            "selected_default": value == 1,
        }
        for b in range(8)
        for cell, value in zip(a.CELLS, (1, 2, 3, 6))
    ]
    summary = a.summarize_statistics(workers, resamples=2000)
    for factor, value in (("objective", 2), ("values", 3), ("interaction", 1)):
        row = summary["factor_estimates"][factor]
        check(
            factor + "_known_factorial_estimator",
            math.isclose(row["ratio_of_ratios"], value, rel_tol=1e-14),
        )
        check(
            factor + "_constant_bootstrap",
            all(
                math.isclose(v, value, rel_tol=1e-14)
                for v in row["bootstrap_interval"]["bounds"]
            ),
        )
    check(
        "adjusted_primary_interval_level",
        summary["factor_estimates"]["objective"]["bootstrap_interval"]["confidence"]
        == 0.975,
    )
    check(
        "interaction_is_secondary",
        summary["factor_estimates"]["interaction"]["primary"] is False,
    )
    rare = [
        {
            "block": b,
            "cell": cell,
            "event_ratio": math.exp(0.4) if b == 0 and cell.startswith("graph_") else 1,
            "wall_ratio": 1,
            "selected_default": True,
        }
        for b in range(8)
        for cell in a.CELLS
    ]
    rare_summary = a.summarize_statistics(rare, resamples=2000)
    check(
        "rare_bad_block_retained_by_mean",
        math.isclose(
            rare_summary["factor_estimates"]["objective"]["ratio_of_ratios"],
            math.exp(0.05),
            rel_tol=1e-14,
        ),
    )
    check(
        "all32_independent_outcomes_retained",
        sum(len(v["worker_outcomes"]) for v in rare_summary["cells"].values()) == 32,
    )
    check(
        "bad_selection_frequency",
        rare_summary["cells"]["graph_synthetic"]["material_regressions_ratio_gt_1_10"][
            "count"
        ]
        == 1,
    )
    check(
        "first_block_integrity_only",
        "bootstrap_interval"
        not in a.summarize_statistics(rare[:4])["factor_estimates"]["objective"],
    )
    rejects("incomplete_block", lambda: a.summarize_statistics(rare[:-1]))
    rejects("duplicate_block_cell", lambda: a.summarize_statistics(rare + [rare[0]]))
    for value in (0, -1, float("nan"), float("inf")):
        rejects(
            "invalid_primary_ratio_" + repr(value),
            lambda value=value: a.block_contrasts(dict(zip(a.CELLS, (1, value, 1, 1)))),
        )

    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    def metadata(shape, dtype, pointer, stride=None):
        if stride is None:
            stride = []
            product = 1
            for size in reversed(shape):
                stride.insert(0, product)
                product *= size
        return {
            "shape": shape,
            "stride": stride,
            "dtype": dtype,
            "device": "cuda:0",
            "data_ptr": pointer,
            "alignment_mod256": pointer % 256,
            "alignment_mod4096": pointer % 4096,
        }

    def inputs(m, base):
        return [
            metadata([1, m, 1024], "torch.float8_e4m3fn", base),
            metadata(
                [1, 1024, 4096], "torch.float8_e4m3fn", base + 256, [1024, 1, 1024]
            ),
            metadata([1], "torch.float32", base + 512),
            metadata([1], "torch.float32", base + 768),
            metadata([1, m, 4096], "torch.bfloat16", base + 1024),
            metadata([41943040], "torch.uint8", base + 1280),
        ]

    tactics = [-1] + [[14400 + i] + [0] * 7 for i in range(1, 9)]
    source = {
        "commit": plan["source_contract"]["flashinfer_commit"],
        "critical_sha256": a.RUNTIME_SHA256,
        "sglang_source_sha256": fixture["source"]["sglang_source_sha256"],
        "bootstrap_helper_sha256": plan["qualified_source"][
            "container_preflight_sha256"
        ],
        "candidate_manifest_sha256": plan["qualified_source"][
            "candidate_manifest_sha256"
        ],
        "native_identity_passthrough": True,
        "startup_wrapper_sha256": a.sha256(
            root / "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
        ),
        "facade_source_sha256": digest("fixed facade CPU fixture"),
    }
    gpu_environment = {
        "torch": "2.14.1+cu130",
        "cuda": "13.0",
        "gpu": "NVIDIA H100 NVL",
        "gpu_uuid": plan["environment"]["uuid"],
        "inference_mode": True,
        "grad_enabled": False,
        "allow_tf32": False,
        "flashinfer_version": "0.0.0+unknown",
    }
    bare = {
        "active_v2": False,
        "is_tuning_mode": False,
        "tuning_contexts": 0,
        "measurement_policy": None,
        "winner_identity": "None",
        "managed_store": False,
        "on_caller_stream": True,
        "on_forward_stream": False,
    }
    actual_layer = {
        "function": "apply_fp8_linear_bmm_flashinfer",
        "custom_op": "flashinfer_bmm_fp8",
        "quantizer": "static_quant_fp8",
        "backend": "cublas",
        "inference_mode": True,
        "custom_op_direct_wrapper": True,
        **dict.fromkeys(
            (
                "grad_enabled",
                "dispatch_modes",
                "function_modes",
                "torch_profiler_enabled",
                "compiling",
                "kernel_api_log_level",
            ),
            False,
        ),
    }

    def key(m):
        return f"('fp8_gemm', 'CublasFp8GemmRunner', ((1, {256 if m == 144 else m}, 1024), (1, 1024, 4096), (1,), (1,), (1, -1, 4096), (-1,)), (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16))"

    check(
        "real_dtype_cache_key_repr", a.parse_file_key(key(144))[2][0] == (1, 256, 1024)
    )
    rejects(
        "cache_key_code_execution",
        lambda: a.parse_file_key("__import__('os').system('false')"),
    )

    def payload(assignment, mode, raw_root):
        objective, values = assignment["objective"], assignment["values"]
        policy = a.policy_for(objective)
        state = {
            **bare,
            "active_v2": True,
            "measurement_policy": policy,
            "winner_identity": "None"
            if objective == "auto"
            else "('__policy__', ('measure_execution_mode', 'cuda_graph'))",
        }
        settings = {
            "warmup": 3,
            "repeat": 10,
            "stream_delay_micro_secs": 5000,
            "cuda_graph_delay_micro_secs": 100,
            "use_global_timer": False,
            "measurement_policy": policy,
            "effective_policy_timer": None if policy is None else "events",
            "config": {
                "use_cuda_graph": objective == "graph",
                "use_cold_l2_cache": False,
                "use_cuda_graph_individual_timing": False,
                "cuda_graph_profile_replays": 1,
                "profiling_repeat": None,
                "effective_profiling_repeat": 10,
            },
        }
        d = {
            "schema_version": 1,
            "status": "success",
            "mode": mode,
            "trial": assignment["block"],
            "block": assignment["block"],
            "cell": assignment["cell"],
            "capture_order": assignment["capture_order"],
            "objective": objective,
            "values": values,
            "assignment": assignment,
            "round_schedule": a.expected_round_orders(assignment["block"]),
            "pid": 1,
            "worker_sha256": a.sha256(
                root / "test/manual/kernels/bench_flashinfer_objective_values.py"
            ),
            "plan_sha256": a.sha256(batch / "plan.json"),
            "schedule_sha256": a.sha256(batch / "schedule.json"),
            "protocol_errors": [],
            "forbidden_profile_attempts": [],
            "pending_sample": None,
            "initializer_config_restored": True,
            "synthetic_input_references_retained": False,
            "persistence_under_test": False,
            "all_samples_retained": True,
            "queue_query_is_exclusion_criterion": False,
            "profiling_byte_attestation": "admission_full_hashes"
            if mode == "admission"
            else "metadata_only_no_synthetic_byte_attestation",
            "source": source,
            "environment": gpu_environment,
            "actual_layer": actual_layer,
            "fixture": {
                "manifest_sha256": a.sha256(fixture_path),
                "tensor_file_sha256": "c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b",
                "shape": [144, 4096, 1024],
                "quantized_input_sha256": fixture["quantization"][
                    "independent_input_fp8_sha256"
                ],
                "tensors": fixture["tensors"],
            },
            "delay_helper": {
                "wrapper_source_sha256": digest("same Torch sleep"),
                "prime_cycles": 10_000_000,
                "primed_before_startup": True,
            },
            "startup": {
                "started_ns": 10,
                "ended_ns": 20,
                "profile_attempts": 81,
                "profiles": [],
                "candidate_order": {
                    str(m): [-1] + [[m * 100 + i] + [0] * 7 for i in range(1, 9)]
                    for m in (1, 2, 4, 8, 16, 32, 64, 128, 144)
                },
                "initializer_calls": [],
                "incidental_legacy_entries": 9 if objective == "auto" else 0,
                "scope_records": [],
                "cache_sha256": digest(objective + "file"),
                "cache_state_sha256": digest(objective + "memory"),
                "rng_before": {
                    "cpu_sha256": digest("cpu before"),
                    "cuda_sha256": digest("gpu before"),
                },
                "rng_after": {
                    "cpu_sha256": digest("cpu after"),
                    "cuda_sha256": digest("gpu after"),
                },
            },
            "correctness": [],
            "selections": [],
            "graphs": [],
            "rounds": [],
        }
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 144):
            generated = digest("generated" + str(m))
            returned = (
                d["fixture"]["quantized_input_sha256"]
                if m == 144 and values == "captured"
                else generated
            )
            call = {
                "shape": [1, m, 1024],
                "dtype": "torch.float8_e4m3fn",
                "device": "cuda:0",
                "targeted": m == 144,
                "cloned": m == 144,
                "copied": m == 144,
            }
            if mode == "admission":
                call.update(generated_sha256=generated, returned_sha256=returned)
            d["startup"]["initializer_calls"].append(call)
            for tactic in d["startup"]["candidate_order"][str(m)]:
                row = {
                    "index": len(d["startup"]["profiles"]),
                    "m": m,
                    "tactic": tactic,
                    "metadata": inputs(m, 25600),
                    "key": key(m),
                    "settings": settings,
                    "elapsed_ms": 0.005
                    if tactic == -1
                    else 0.007 + (tactic[0] % 100) * 0.00001,
                    "error": None,
                    "fingerprinted": mode == "admission",
                }
                if mode == "admission":
                    row["fingerprints"] = [
                        returned,
                        digest("profile B"),
                        digest("profile scaleA"),
                        digest("profile scaleB"),
                    ]
                d["startup"]["profiles"].append(row)
        stages = ["startup", "natural_lookup", "winner_capture", "default_capture"] + (
            ["admission_explicit"] if mode == "admission" else []
        )
        for stage in stages:
            for event in ("enter", "before_exit", "after_exit"):
                s = copy.deepcopy(bare if event == "after_exit" else state)
                s["is_tuning_mode"] = stage == "startup" and event != "after_exit"
                d["startup"]["scope_records"].append(
                    {"stage": stage, "event": event, **s}
                )
        d["natural"] = {
            "tactic": -1,
            "key": key(144),
            "cache_hit": True,
            "lookup_tier": "memory",
        }
        ids = list(
            a.GRAPHS
            if assignment["capture_order"] == "winner-first"
            else reversed(a.GRAPHS)
        ) + (["admission_explicit"] if mode == "admission" else [])

        def numerical(label, graph=None, zero=False):
            row = {
                "check": label,
                "graph_id": None if graph is None else graph["id"],
                "output_data_ptr": 999
                if graph is None
                else graph["output"]["data_ptr"],
                "passed": True,
                "max_abs_error": 0 if zero else 0.01,
                "exact_zero_required": zero,
            }
            d["correctness"].append(row)
            return label

        numerical("natural_lookup")
        for index, name in enumerate(ids):
            inp = inputs(144, 100000 + index * 10000)
            tactic = tactics[1] if name == "admission_explicit" else -1
            selection = {
                "stage": name + "_capture",
                "tactic": tactic,
                "cache_hit": name != "default",
                "lookup_tier": "miss" if name == "default" else "memory",
                "capturing": True,
                "key": key(144),
                **state,
            }
            graph = {
                "id": name,
                "tactic": tactic,
                "lookup_tier": selection["lookup_tier"],
                "selection": selection,
                "graph_object_id": 100 + index,
                "inputs": inp,
                "linear_input": metadata([144, 1024], "torch.bfloat16", 40000),
                "output": metadata([144, 4096], "torch.bfloat16", inp[4]["data_ptr"]),
                "read_operand_hashes": [
                    d["fixture"]["quantized_input_sha256"],
                    digest("serving B"),
                    digest("serving sA"),
                    digest("serving sB"),
                ],
                "capture_started_ns": 100 + index * 10,
                "capture_ended_ns": 105 + index * 10,
            }
            d["graphs"].append(graph)
            d["selections"].append(selection)
            numerical(name + "_before")
            for suffix in ("poisoned", "changed_input", "restored_input"):
                numerical(name + "_" + suffix, graph, suffix == "changed_input")
        graphs = {g["id"]: g for g in d["graphs"]}
        d["miss_scopes"] = [
            {
                "restored": True,
                "winner_entries_before": 9,
                "file_entries_before": 0,
                "winner_entries_after_removal": 0,
                "file_entries_after_removal": 0,
                "cache_sha256_before": d["startup"]["cache_sha256"],
                "cache_sha256_after": d["startup"]["cache_sha256"],
            }
            for _ in range(2 if mode == "admission" else 1)
        ]
        d["pre_capture_cache_state_sha256"] = d["startup"]["cache_state_sha256"]
        clock = 10000
        if mode == "timing":
            d["initial_warmups"] = [
                {"graph_id": name, "actual_ms": 501, "calls": 81920}
                for name in a.GRAPHS
            ]
            for index, order in enumerate(d["round_schedule"]):
                group = {"index": index, "order": order, "measurements": []}
                d["rounds"].append(group)
                for name in order:
                    graph = graphs[name]
                    row = {
                        "graph_id": name,
                        "graph_object_id": graph["graph_object_id"],
                        "output_data_ptr": graph["output"]["data_ptr"],
                        "replays": 16,
                        "prequeue_cycles": 10_000_000,
                        "rewarm": {"actual_ms": 21, "calls": 5120},
                        "profile_attempts_before": 81,
                        "profile_attempts_after": 81,
                        "start_query_error": None,
                        "wall_started_ns": clock,
                        "enqueue_started_ns": clock + 100,
                        "enqueue_ended_ns": clock + 1000,
                        "ended_ns": clock + 2_000_000,
                        "gpu_event_batch_ms": 0.112,
                        "gpu_event_per_replay_ms": 0.007,
                        "start_event_complete_after_enqueue": index % 2 == 0,
                        "queue_evidence": "start_complete_inconclusive"
                        if index % 2 == 0
                        else "start_pending_after_enqueue",
                        "enqueue_cpu_ms": 0.0009,
                        "enqueue_span_includes_start_query": True,
                        "wall_including_prequeue_ms": 2,
                        "wall_diagnostic_only": True,
                        "correctness_check": numerical(
                            f"round_{index:02d}_{name}", graph
                        ),
                    }
                    group["measurements"].append(row)
                    clock += 3_000_000
            d["selected_over_default_graph_event_median"] = 1

        def acceptance(graph, label):
            nonlocal clock
            clock += 100
            row = {
                "graph_id": graph["id"],
                "graph_object_id": graph["graph_object_id"],
                "output_data_ptr": graph["output"]["data_ptr"],
                "tactic": graph["tactic"],
                "captured_inputs": graph["inputs"],
                "started_ns": clock,
                "ended_ns": clock + 10,
                "used_descriptor": None if graph["tactic"] == -1 else 1,
                "route": "native_default"
                if graph["tactic"] == -1
                else "native_descriptor",
                "descriptor_acceptance": "not_applicable_default_route"
                if graph["tactic"] == -1
                else "accepted",
                "pointers_unchanged": True,
                "original_graph_replayed": True,
                "raw_numerical_check": numerical(label + "_raw", graph),
                "original_graph_check": numerical(label + "_original_graph", graph),
            }
            clock += 20
            return row

        d["post_timing_acceptance"] = acceptance(
            graphs["winner"], "selected_acceptance"
        )
        if mode == "admission":
            d["known_explicit_control"] = {
                "accepted": True,
                "tactic": tactics[1],
                "attempts": [{"tactic": tactics[1], "used_descriptor": 1}],
                "actual_acceptance": acceptance(
                    graphs["admission_explicit"], "admission_explicit"
                ),
            }
        for field, prefix in (("pre_trace_checks", "pre_trace"), ("final", "final")):
            d[field] = {
                **bare,
                "ended_ns": clock + 100
                if field == "pre_trace_checks"
                else clock + 1000,
                "profile_attempts": 81,
                "forbidden_profile_attempts": [],
                "cache_sha256": d["startup"]["cache_sha256"],
                "cache_state_sha256": d["startup"]["cache_state_sha256"],
                "cache_unchanged": True,
                "operands_unchanged": True,
            }
            for name, g in graphs.items():
                numerical(prefix + "_" + name, g)
        trace_name = (
            "admission-" + assignment["cell"]
            if mode == "admission"
            else Path(assignment["raw_name"]).stem
        ) + ".trace.json"
        events = []
        producer = []
        for gi, name in enumerate(a.GRAPHS):
            start = gi * 1000
            events.append(
                {
                    "name": "retained_" + name,
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": start,
                    "dur": 900,
                }
            )
            signatures = []
            sequence = []
            for cycle in range(3):
                correlation = gi * 10 + cycle
                events.append(
                    {
                        "name": "cudaGraphLaunch",
                        "cat": "cuda_runtime",
                        "ts": start + cycle * 100,
                        "dur": 1,
                        "args": {"correlation": correlation},
                    }
                )
                for ki, kernel in enumerate(
                    ("_static_quant_fp8_kernel", "actual_gemm_kernel")
                ):
                    args = {
                        "correlation": correlation,
                        "graph id": gi + 10,
                        "device": 0,
                        "grid": [1, 1, 1],
                        "block": [128, 1, 1],
                        "registers per thread": 32,
                        "shared memory": 0,
                    }
                    event = {
                        "name": kernel,
                        "cat": "kernel",
                        "ts": start + cycle * 100 + 10 + ki * 10,
                        "dur": 5,
                        "args": args,
                    }
                    events.append(event)
                    signature = {
                        "name": kernel,
                        **{
                            key: args[key]
                            for key in (
                                "device",
                                "grid",
                                "block",
                                "registers per thread",
                                "shared memory",
                            )
                        },
                    }
                    if signature not in signatures:
                        signatures.append(signature)
                    sequence.append(signatures.index(signature))
            producer.append(
                {
                    "graph_id": name,
                    "graph_object_id": graphs[name]["graph_object_id"],
                    "output_data_ptr": graphs[name]["output"]["data_ptr"],
                    "graph_launch_events": 3,
                    "kernel_events": 6,
                    "signatures": signatures,
                    "kernel_sequence": sequence,
                }
            )
            numerical("traced_" + name, graphs[name])
        trace_path = raw_root / trace_name
        trace_path.write_text(json.dumps({"traceEvents": events}))
        d["trace"] = {
            "path": str(trace_path),
            "sha256": a.sha256(trace_path),
            "started_ns": clock + 200,
            "ended_ns": clock + 300,
            "calls_per_graph": 3,
            "graphs": producer,
        }
        return d

    args = SimpleNamespace(
        raw_root=None,
        source_root=root,
        plan=batch / "plan.json",
        schedule=batch / "schedule.json",
        fixture_manifest=fixture_path,
        fixture_manifest_data=fixture,
        blocks=1,
        admission=True,
    )
    with tempfile.TemporaryDirectory(prefix="objective-values-analyzer-") as temp:
        raw = Path(temp)
        args.raw_root = raw
        first_assignment = schedule["entries"][0]
        data = payload(first_assignment, "timing", raw)
        outer = {
            "source_sha256": {
                name: a.sha256(root / name)
                for name in [
                    *fixture["source"]["sglang_source_sha256"],
                    "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
                ]
            }
        }
        worker = a.validate_worker(
            data,
            first_assignment,
            SimpleNamespace(**{**vars(args), "admission": False}),
            plan,
            outer,
        )
        check(
            "complete_timing_schema_positive",
            worker["event_ratio"] == 1 and worker["samples_retained"] == 48,
        )
        check(
            "pending_and_inconclusive_query_rows_retained",
            set(worker["query_outcomes_all_retained"].values()) == {24},
        )

        def fault(name, mutate, base=data, admission=False):
            changed = copy.deepcopy(base)
            mutate(changed)
            rejects(
                name,
                lambda: a.validate_worker(
                    changed,
                    first_assignment,
                    SimpleNamespace(**{**vars(args), "admission": admission}),
                    plan,
                    outer,
                ),
            )

        fault(
            "zero_profile_time",
            lambda d: d["startup"]["profiles"][0].update(elapsed_ms=0),
        )
        fault(
            "inline_timing_hash_leak",
            lambda d: d["startup"]["profiles"][0].update(fingerprints=["bad"] * 4),
        )
        fault(
            "wrong_graph_object_sample",
            lambda d: d["rounds"][0]["measurements"][0].update(graph_object_id=-1),
        )
        fault(
            "wrong_event_normalization",
            lambda d: d["rounds"][0]["measurements"][0].update(
                gpu_event_per_replay_ms=0.112 / 5120
            ),
        )
        fault("dropped_query_outcome_round", lambda d: d["rounds"].pop())
        fault(
            "query_outcome_relabel",
            lambda d: d["rounds"][0]["measurements"][0].update(
                queue_evidence="start_pending_after_enqueue"
            ),
        )
        fault(
            "native_default_acceptance_overclaim",
            lambda d: d["post_timing_acceptance"].update(
                descriptor_acceptance="accepted"
            ),
        )
        fault(
            "changed_live_cache_after_trace",
            lambda d: d["final"].update(cache_state_sha256="0" * 64),
        )
        fault("poststartup_profile", lambda d: d["final"].update(profile_attempts=82))
        fault(
            "traced_graph_substitution",
            lambda d: d["trace"]["graphs"][0].update(graph_object_id=-1),
        )
        fault(
            "graph_output_owner_substitution",
            lambda d: d["graphs"][0]["output"].update(data_ptr=111),
        )
        fault(
            "incorrect_zero_control",
            lambda d: next(
                row for row in d["correctness"] if row["exact_zero_required"]
            ).update(max_abs_error=0.1),
        )
        fault(
            "instrumented_fastpath",
            lambda d: d["actual_layer"].update(torch_profiler_enabled=True),
        )
        fault(
            "objective_policy_mismatch",
            lambda d: d["startup"]["profiles"][0]["settings"]["config"].update(
                use_cuda_graph=not d["objective"] == "graph"
            ),
        )
        explicit = copy.deepcopy(data)
        selected = explicit["startup"]["candidate_order"]["144"][1]
        next(
            row
            for row in explicit["startup"]["profiles"]
            if row["m"] == 144 and row["tactic"] == selected
        )["elapsed_ms"] = 0.004
        explicit["natural"]["tactic"] = selected
        selected_graph = next(
            row for row in explicit["graphs"] if row["id"] == "winner"
        )
        selected_graph["tactic"] = selected
        selected_graph["selection"]["tactic"] = selected
        explicit["post_timing_acceptance"].update(
            tactic=selected,
            route="native_descriptor",
            used_descriptor=1,
            descriptor_acceptance="accepted",
        )
        explicit_args = SimpleNamespace(**{**vars(args), "admission": False})
        check(
            "selected_explicit_timing_positive",
            a.validate_worker(explicit, first_assignment, explicit_args, plan, outer)[
                "selected_default"
            ]
            is False,
        )
        rejected = copy.deepcopy(explicit)
        rejected["post_timing_acceptance"]["used_descriptor"] = 0
        rejects(
            "selected_explicit_native_fallback",
            lambda: a.validate_worker(
                rejected, first_assignment, explicit_args, plan, outer
            ),
        )
        admission = [payload(row, "admission", raw) for row in schedule["entries"][:4]]
        for row, d in zip(schedule["entries"][:4], admission):
            a.validate_worker(d, row, args, plan, outer)
        checks.append("all_four_admission_schema_positive")
        check(
            "actual_four_cell_activation_admission",
            a.validate_comparability(admission, admission=True)[
                "only_profile_M144_A_changed"
            ],
        )
        wrong = copy.deepcopy(admission)
        wrong[0]["startup"]["profiles"][0]["fingerprints"][1] = "wrong weight"
        rejects(
            "admission_weight_change",
            lambda: a.validate_comparability(wrong, admission=True),
        )
        wrong = copy.deepcopy(admission)
        wrong[0]["startup"]["rng_after"]["cpu_sha256"] = "wrong RNG"
        rejects(
            "admission_rng_divergence",
            lambda: a.validate_comparability(wrong, admission=True),
        )
        trace = a.load(raw / data["trace"]["path"].split("/")[-1])
        trace["traceEvents"][2]["args"]["correlation"] = -1
        rejects("raw_trace_launch_correlation", lambda: a.parse_retained_trace(trace))
    # Exercise the real full admission/first-block/full-cohort pipeline. Only
    # historical Git lookup is adapted: these new files are not committed yet.
    with tempfile.TemporaryDirectory(prefix="objective-values-main-") as temp:
        raw = Path(temp)
        args.raw_root = raw
        required = [
            a.BATCH + "/analyze.py",
            a.BATCH + "/run_current.py",
            a.BATCH + "/plan.json",
            a.BATCH + "/schedule.json",
            "test/manual/kernels/bench_flashinfer_objective_values.py",
            "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
            *fixture["source"]["sglang_source_sha256"],
        ]
        outer = {
            "schema_version": 1,
            "execution_commit": "f" * 40,
            "plan_sha256": a.sha256(args.plan),
            "schedule_sha256": a.sha256(args.schedule),
            "source_sha256": {name: a.sha256(root / name) for name in required},
            "gpu": {
                "name": "NVIDIA H100 NVL",
                "uuid": plan["environment"]["uuid"],
                "driver_version": "580.159.04",
                "power.limit [W]": "310.00 W",
                "memory.used [MiB]": "0 MiB",
            },
            "image": {
                "RepoDigests": [plan["environment"]["image"]],
                "Architecture": "amd64",
                "Os": "linux",
                "Id": plan["environment"]["image_id"],
            },
            "fixture_manifest_sha256": a.sha256(fixture_path),
            "fixture_tensor_file_sha256": "c30aa761eba827bb0b4eb6de0e99d9b8dda8243dd16d00146f9718d2abb05e8b",
            "candidate_manifest_sha256": plan["qualified_source"][
                "candidate_manifest_sha256"
            ],
            "helper_sha256": plan["qualified_source"]["container_preflight_sha256"],
            "private_source_archive_sha256": "209b2de9e76c358b94407202a4670613055e963200184e716b50eb67aad71c91",
        }
        (raw / "manifest.json").write_text(json.dumps(outer))
        providers = {}
        for name, names in (
            ("gemm", ["bmm_fp8.cu", "group_gemm.cu", "flashinfer_gemm_binding.cu"]),
            (
                "trtllm_utils",
                [
                    "nv_internal/tensorrt_llm/kernels/delayStream.cu",
                    "nv_internal/cpp/common/envUtils.cpp",
                    "nv_internal/cpp/common/logger.cpp",
                    "nv_internal/cpp/common/stringUtils.cpp",
                    "nv_internal/cpp/common/tllmException.cpp",
                ],
            ),
        ):
            sources = {
                "/study/input/current/source/csrc/" + name: digest(name)
                for name in names
            }
            if name == "gemm":
                sources["/study/input/current/source/csrc/bmm_fp8.cu"] = (
                    "ac00c1cdba76e653ead25a77d2b1ca9d07f35a89f65221334fbbab1a695a4709"
                )
                sources[
                    "/study/input/current/source/csrc/flashinfer_gemm_binding.cu"
                ] = "4cfdde4e2d7928ffa442d4a386c8f545de61d68820a6d01e4bf78c1b4a490e8e"
            providers[name] = {
                "library_path": "/study/cache/cached_ops/" + name + ".so",
                "library_sha256": digest(name + "ELF"),
                "ninja_sha256": digest(name + "ninja"),
                "sources": sources,
                "exports_checked": [
                    "bmm_fp8",
                    "bmm_fp8_get_algos",
                    "bmm_fp8_run_with_descriptor",
                ]
                if name == "gemm"
                else ["delay_kernel"],
                "linkage": {"returncode": 0, "stdout": "resolved"},
                "compile_commands": [
                    {
                        "file": p,
                        "directory": "/study/cache",
                        "command": "nvcc -gencode=arch=compute_90a,code=sm_90a " + p,
                    }
                    for p in sources
                ],
            }

        def write_bundle(assignment, mode):
            data = payload(assignment, mode, raw)
            stem = (
                "admission-" + assignment["cell"]
                if mode == "admission"
                else Path(assignment["raw_name"]).stem
            )
            path = raw / (stem + ".json")
            path.write_text(json.dumps(data))
            critical = {
                **a.RUNTIME_SHA256,
                "include/flashinfer/gemm/bmm_fp8.cuh": "5a35da070efbc028e565721110e059c5b143d6a49697995e3c70fdae258eae72",
            }
            imports = {
                module: {
                    "path": "/study/input/current/source/" + p,
                    "sha256": a.RUNTIME_SHA256[p],
                }
                for module, p in (
                    (
                        "flashinfer.autotuner.autotuner",
                        "flashinfer/autotuner/autotuner.py",
                    ),
                    ("flashinfer.autotune_cache", "flashinfer/autotune_cache.py"),
                    ("flashinfer.gemm.gemm_base", "flashinfer/gemm/gemm_base.py"),
                )
            }
            bootstrap = {
                "status": "passed",
                "pid": 1,
                "native_identity_passthrough": True,
                "worker_sha256": data["worker_sha256"],
                "result_sha256": a.sha256(path),
                "helper_sha256": plan["qualified_source"]["container_preflight_sha256"],
                "worker_args": [
                    "--mode",
                    mode,
                    "--block",
                    str(assignment["block"]),
                    "--cell",
                    assignment["cell"],
                    "--capture-order",
                    assignment["capture_order"],
                    "--output",
                    str(path),
                ],
                "source": {
                    "source_commit": source["commit"],
                    "variant": "fp8-runtime-callback",
                    "manifest_sha256": source["candidate_manifest_sha256"],
                    "source_sha256": critical,
                },
                "import": {
                    "flashinfer_runtime_version": "0.0.0+unknown",
                    "flashinfer_runtime_commit": "unknown",
                    "source_modules": imports,
                },
                "native": {"providers": providers},
            }
            path.with_suffix(".bootstrap.json").write_text(json.dumps(bootstrap))
            (raw / (stem + ".container-id")).write_text(digest(stem))
            (raw / (stem + ".exit-code")).write_text("0\n")

        def git_lookup(argv, **kwargs):
            if argv[:4] != ["git", "-C", str(root), "show"] or not argv[4].startswith(
                "f" * 40 + ":"
            ):
                raise AssertionError(
                    "CPU control requested an unexpected Git operation"
                )
            return SimpleNamespace(
                stdout=(root / argv[4].split(":", 1)[1]).read_bytes()
            )

        with patch("subprocess.run", git_lookup):
            for assignment in schedule["entries"][:4]:
                write_bundle(assignment, "admission")
            args.admission = True
            result = a.analyze(args)
            check(
                "actual_nvidia_smi_unit_strings",
                outer["gpu"]["power.limit [W]"] == "310.00 W"
                and outer["gpu"]["memory.used [MiB]"] == "0 MiB",
            )
            unitless = copy.deepcopy(outer)
            unitless["gpu"]["power.limit [W]"] = "310.00"
            (raw / "manifest.json").write_text(json.dumps(unitless))
            a.validate_outer(args, plan)
            checks.append("unitless_gpu_power_metadata_compatible")
            (raw / "manifest.json").write_text(json.dumps(outer))
            check(
                "full_main_four_admission_outputs",
                result["validated_workers"] == 4 and result["independent_blocks"] == 0,
            )
            for assignment in schedule["entries"][:4]:
                write_bundle(assignment, "timing")
            args.admission = False
            args.blocks = 1
            result = a.analyze(args)
            check(
                "full_main_first_block_no_inference",
                result["validated_workers"] == 4
                and result["statistics"]["independent_blocks"] == 1
                and "bootstrap_interval"
                not in result["statistics"]["factor_estimates"]["objective"],
            )
            for assignment in schedule["entries"][4:]:
                write_bundle(assignment, "timing")
            args.blocks = 8
            result = a.analyze(args)
            check(
                "full_main_all32_and200000_bootstraps",
                result["validated_workers"] == 32
                and result["statistics"]["factor_estimates"]["objective"][
                    "bootstrap_interval"
                ]["resamples"]
                == a.BOOTSTRAP_RESAMPLES,
            )
            check(
                "full_main_json_serialization",
                json.loads(json.dumps(result, allow_nan=False))["status"] == "passed",
            )
            for field, value in (
                ("plan_sha256", "wrong"),
                ("helper_sha256", "wrong"),
                ("fixture_tensor_file_sha256", "wrong"),
            ):
                bad = copy.deepcopy(outer)
                bad[field] = value
                (raw / "manifest.json").write_text(json.dumps(bad))
                rejects("full_main_outer_" + field, lambda: a.analyze(args))
            (raw / "manifest.json").write_text(json.dumps(outer))
            path = raw / (
                "admission-" + schedule["entries"][0]["cell"] + ".bootstrap.json"
            )
            saved = path.read_text()
            bad = json.loads(saved)
            bad["native"]["providers"]["gemm"]["sources"][
                "/study/input/current/source/csrc/bmm_fp8.cu"
            ] = "wrong"
            path.write_text(json.dumps(bad))
            rejects("full_main_wrong_actual_native_source", lambda: a.analyze(args))
            path.write_text(saved)
            path = raw / (
                Path(schedule["entries"][4]["raw_name"]).stem + ".container-id"
            )
            saved = path.read_text()
            path.write_text(
                (
                    raw
                    / (Path(schedule["entries"][0]["raw_name"]).stem + ".container-id")
                ).read_text()
            )
            rejects("full_main_reused_container_process", lambda: a.analyze(args))
            path.write_text(saved)
    return {
        "status": "passed",
        "checks": len(checks),
        "controls": checks,
        "scope": "CPU schema/statistics positive and fault controls only; no actual GPU behavior qualified.",
        "controlled_boundary": "Full-main Git blob lookup adapted to current uncommitted files only; actual source file hashes and all outer/bootstrap/raw/trace validations still execute.",
    }
