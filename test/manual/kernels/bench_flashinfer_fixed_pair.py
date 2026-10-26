"""Fixed descriptor/default diagnostic after the original startup search.

Use the qualified external bootstrap and frozen plan. This manual study does
not change production selection. ``winner`` names the fixed explicit control,
not necessarily the natural startup winner. Late profiler measurements follow
the graph measurements and are not the original candidate-search history.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import inspect
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch

from bench_flashinfer_objective_values import (
    GRAPH_IDS,
)
from bench_flashinfer_objective_values import Experiment as OriginalExperiment
from bench_flashinfer_objective_values import (
    require,
    sha,
    tactic_json,
)

FIXED_DESCRIPTOR = (
    77309411394,
    4294967332,
    0,
    1407374883553283,
    558779540176897,
    120259084316,
    292058693646,
    0,
)
PARENT_SHA = "979e28aeb3248b27cb27df21b98cf274442c0e6b92c4df3444c48f7214ce90ba"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("admission", "timing"), required=True)
    parser.add_argument("--trial", type=int, choices=range(8), required=True)
    parser.add_argument(
        "--capture-order", choices=("winner-first", "default-first"), required=True
    )
    parser.add_argument(
        "--profile-order", choices=("auto-first", "graph-first"), required=True
    )
    for name in (
        "plan",
        "schedule",
        "historical-contract",
        "fixture",
        "fixture-manifest",
        "cache",
        "output",
        "trace",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.block, args.cell = args.trial, "auto_synthetic"
    return args


def paired_orders(document, block, objective):
    orders = document["profiler_pair_schedules"][str(block)][objective]
    require(
        len(orders) == 12
        and all(sorted(order) == sorted(GRAPH_IDS) for order in orders)
        and sum(order[0] == "winner" for order in orders) == 6,
        "profiler pair schedule is not balanced twelve-round schedule",
    )
    return orders


class Experiment(OriginalExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.historical = json.loads(args.historical_contract.read_text())
        require(
            sha(args.historical_contract) == self.plan["historical_contract_sha256"]
            and self.historical["fixed_descriptor"] == list(FIXED_DESCRIPTOR)
            and self.plan["protocol"]["fixed_descriptor"] == list(FIXED_DESCRIPTOR),
            "fixed historical descriptor/contract changed",
        )
        parent = Path(inspect.getfile(OriginalExperiment))
        require(sha(parent) == PARENT_SHA, "original worker source changed")
        self.objective_order = self.assignment["profiler_objective_order"]
        require(
            self.objective_order
            == (
                ["auto", "graph"]
                if args.profile_order == "auto-first"
                else ["graph", "auto"]
            )
            and sorted(self.objective_order) == ["auto", "graph"],
            "profiler objective order differs from frozen schedule",
        )
        self.pair_orders = {
            objective: paired_orders(self.schedule, args.block, objective)
            for objective in self.objective_order
        }
        self.retained = None
        self.diagnostic_attempts, self.allow_diagnostic = 0, False
        self.result.update(
            worker_sha256=sha(__file__),
            parent_worker_sha256=sha(parent),
            historical_contract_sha256=sha(args.historical_contract),
            fixed_descriptor=list(FIXED_DESCRIPTOR),
            graph_role_alias={"winner": "fixed_explicit", "default": "untuned_default"},
            profiler_objective_order=self.objective_order,
            paired_profiles=[],
            diagnostic_profile_attempts=[],
            profile_correctness=[],
            cache_partition_restorations=[],
            limits=[
                "Retaining final-profile operands extends allocation lifetime after index80; this is not byte-identical historical execution.",
                "Late pair timings follow graph timing, reference preparation and repeated runner preparation; they are not the original search-order measurements.",
                "Serving graph traces observe the retained serving graphs, not each timed synthetic profiling launch.",
                "The fixed graph descriptor is independent of the natural startup winner. No serving-performance claim.",
            ],
        )

    def profile(self, runner, inputs, tactic, config, *more, **kwargs):
        elapsed = super().profile(runner, inputs, tactic, config, *more, **kwargs)
        index = len(self.result["startup"]["profiles"]) - 1
        if index == 80:
            require(self.retained is None, "final profile operands already retained")
            self.retained = {
                "inputs": tuple(inputs),
                "runner": runner,
                "config": config,
                "more": more,
                "kwargs": dict(kwargs),
            }
            self.result["retained_profile"] = {
                "retained_after_index": index,
                "metadata": [self.metadata(t) for t in inputs],
                "tensor_object_ids": [id(t) for t in inputs],
                "retained_after_ns": time.perf_counter_ns(),
                "startup_input_batches_present": kwargs.get("input_tensor_batches")
                is not None,
            }
        return elapsed

    def startup(self):
        super().startup()
        rows = self.result["startup"]["profiles"]
        require(
            self.profile_attempts == 81
            and self.retained is not None
            and self.result["startup"]["candidate_order"]
            == self.historical["candidate_order"]
            and [row["key"] for row in rows] == self.historical["profile_keys"]
            and all(
                row["settings"] == self.historical["original_settings"] for row in rows
            )
            and rows[72]["tactic"] == -1
            and rows[74]["tactic"] == list(FIXED_DESCRIPTOR),
            "original search candidate order, keys, settings or fixed ordinal changed",
        )
        expected = self.result["retained_profile"]["metadata"]
        require(
            all(row["metadata"] == expected for row in rows[72:81])
            and [self.metadata(t) for t in self.retained["inputs"]] == expected
            and not self.retained["more"]
            and set(self.retained["kwargs"]) <= {"input_tensor_batches"},
            "runtime candidates did not share the same exact six operands",
        )
        self.result["original_history"] = {
            "default_index": 72,
            "explicit_index": 74,
            "default_ms": rows[72]["elapsed_ms"],
            "explicit_ms": rows[74]["elapsed_ms"],
            "explicit_over_default": rows[74]["elapsed_ms"] / rows[72]["elapsed_ms"],
            "actual_m": 144,
            "canonical_cache_m": 256,
            "all_81_candidate_order_keys_settings_match": True,
        }

    def capture_controls(self):
        self.result["pre_capture_cache_state_sha256"] = self.cache_state_sha()
        self.result["pre_capture_cache_state_field_sha256"] = dict(
            self.cache_state_field_sha256
        )
        require(self.cache_state_sha() == self.memory_sha, "startup cache changed")
        with self.policy_scope("replay", "natural_lookup"):
            saved = {}
            with patch.object(
                self.tuner,
                "choose_one",
                self.choose_recorder(None, "memory", "natural_lookup", saved),
            ):
                out = self.operation()
            self.check(out, "natural_lookup")
            self.natural_winner = saved["tactic"]
            runtime = self.result["startup"]["profiles"][72:81]
            require(
                tactic_json(self.natural_winner)
                == min(runtime, key=lambda row: row["elapsed_ms"])["tactic"],
                "natural winner is not the first profiling minimum",
            )
            self.result["natural"] = {
                "tactic": tactic_json(self.natural_winner),
                "key": saved["key"].file_key,
                "cache_hit": True,
                "lookup_tier": "memory",
                "consumed_by_fixed_graph": False,
            }
        self.winner = FIXED_DESCRIPTOR
        order = (
            GRAPH_IDS
            if self.args.capture_order == "winner-first"
            else tuple(reversed(GRAPH_IDS))
        )
        for graph_id in order:
            with self.policy_scope("replay", graph_id + "_capture"), self.miss_scope():
                if graph_id == "winner":
                    self.tuner._cache_profile_result(
                        saved["key"], None, saved["config"], FIXED_DESCRIPTOR
                    )
                self.capture(
                    graph_id,
                    FIXED_DESCRIPTOR if graph_id == "winner" else -1,
                    "memory" if graph_id == "winner" else "miss",
                )
        require(
            len({saved["output"].data_ptr() for saved in self.graphs.values()}) == 2,
            "fixed graphs share output allocation",
        )
        self.assert_bare()

    def prepare_reference(self):
        self.stage = "late_profile_reference"
        inputs = self.retained["inputs"]
        self.result["retained_profile"]["reference_preparation_started_ns"] = (
            time.perf_counter_ns()
        )
        self.profile_read_hashes = [self.tensor_hash(t) for t in inputs[:4]]
        a, b, sa, sb, _, _ = inputs
        self.profile_reference = (
            self.torch.bmm(a.float(), b.float()) * sa.float() * sb.float()
        )
        self.torch.cuda.synchronize()
        self.result["retained_profile"].update(
            read_operand_hashes=self.profile_read_hashes,
            reference_preparation_ended_ns=time.perf_counter_ns(),
            reference="FP32 bmm of exact retained quantized operands multiplied by exact retained scales",
        )

    def check_profile_output(self, label, zero=False):
        inputs = self.retained["inputs"]
        require(
            [id(t) for t in inputs]
            == self.result["retained_profile"]["tensor_object_ids"]
            and [self.metadata(t) for t in inputs]
            == self.result["retained_profile"]["metadata"],
            "retained profiling operand ownership changed",
        )
        self.torch.cuda.synchronize()
        actual = inputs[4].float()
        target = (
            self.torch.zeros_like(self.profile_reference)
            if zero
            else self.profile_reference
        )
        require(
            bool(self.torch.isfinite(actual).all()),
            "nonfinite retained profiling output",
        )
        require(
            bool(self.torch.count_nonzero(actual) == 0)
            if zero
            else bool(self.torch.allclose(actual, target, rtol=0.02, atol=0.125)),
            "retained profiling numerical check failed: " + label,
        )
        self.result["profile_correctness"].append(
            {
                "check": label,
                "passed": True,
                "output_data_ptr": inputs[4].data_ptr(),
                "max_abs_error": float((actual - target).abs().max()),
                "exact_zero_required": zero,
                "checked_ns": time.perf_counter_ns(),
            }
        )
        return label

    def diagnostic_profile(self, runner, inputs, tactic, config, *more, **kwargs):
        self.diagnostic_attempts += 1
        row = {
            "index": self.diagnostic_attempts - 1,
            "stage": self.stage,
            "objective": self.late_objective,
            "tactic": tactic_json(tactic),
            "settings": self.settings(config),
            "metadata": [self.metadata(t) for t in inputs],
            "tensor_object_ids": [id(t) for t in inputs],
            "started_ns": time.perf_counter_ns(),
            "elapsed_ms": None,
            "error": None,
        }
        self.result["diagnostic_profile_attempts"].append(row)
        try:
            require(
                self.allow_diagnostic
                and not self.allow_profiling
                and runner is self.retained["runner"]
                and not more
                and tactic in (-1, FIXED_DESCRIPTOR)
                and len(inputs) == 6
                and all(
                    t is original
                    for t, original in zip(inputs, self.retained["inputs"], strict=True)
                )
                and len(kwargs.get("input_tensor_batches", [])) == 1
                and all(
                    t is original
                    for t, original in zip(
                        kwargs["input_tensor_batches"][0], inputs, strict=True
                    )
                )
                and config.use_cuda_graph == (self.late_objective == "graph")
                and not config.use_cold_l2_cache
                and all(
                    row["settings"][name] == self.historical["original_settings"][name]
                    for name in (
                        "warmup",
                        "repeat",
                        "stream_delay_micro_secs",
                        "cuda_graph_delay_micro_secs",
                        "use_global_timer",
                    )
                )
                and row["settings"]["measurement_policy"]
                == (
                    dataclasses.asdict(self.policy) if self.policy is not None else None
                )
                and self.tuner._in_v2_context
                and self.tuner.is_tuning_mode
                and self.torch.cuda.current_stream()
                == self.model_runner.forward_stream,
                "undeclared profiler call, operands, schedule or objective",
            )
            elapsed = self.original_profile(runner, inputs, tactic, config, **kwargs)
            require(
                isinstance(elapsed, (int, float))
                and math.isfinite(elapsed)
                and elapsed > 0,
                "invalid actual profiler duration",
            )
            row.update(elapsed_ms=float(elapsed), ended_ns=time.perf_counter_ns())
            return elapsed
        except Exception as error:
            row.update(
                error=f"{type(error).__name__}: {error}",
                ended_ns=time.perf_counter_ns(),
            )
            raise

    @contextlib.contextmanager
    def restore_policy_partitions(self, objective):
        before = self.cache_state_sha()
        names = tuple(self.cache_state_field_sha256)
        snapshot = {name: copy.deepcopy(getattr(self.tuner, name)) for name in names}
        row = {"objective": objective, "before_sha256": before}
        self.result["cache_partition_restorations"].append(row)
        try:
            yield
        finally:
            row["before_restore_sha256"] = self.cache_state_sha()
            row["changed_fields"] = [
                name for name in names if getattr(self.tuner, name) != snapshot[name]
            ]
            # Scope entry can materialize an empty identity partition. Direct
            # profiling has no authority to add/remove any winner or file key.
            for name in row["changed_fields"]:
                require(
                    name in {"_winner_partitions", "_protocol_record_partitions"},
                    "direct profiling mutated non-partition cache state: " + name,
                )
                current, old = getattr(self.tuner, name), snapshot[name]
                require(
                    all(
                        key in current and current[key] == value
                        for key, value in old.items()
                    )
                    and all(
                        not value for key, value in current.items() if key not in old
                    ),
                    "direct profiling changed a nonempty partition",
                )
            for name, old in snapshot.items():
                current = getattr(self.tuner, name)
                if isinstance(current, (dict, set)):
                    current.clear()
                    current.update(old)
                else:
                    setattr(self.tuner, name, old)
            row.update(after_sha256=self.cache_state_sha(), restored=True)
            require(
                row["after_sha256"] == before, "policy partitions not fully restored"
            )

    def paired_profiler(self):
        self.prepare_reference()
        inputs, runner = list(self.retained["inputs"]), self.retained["runner"]
        for objective in self.objective_order:
            self.late_objective = objective
            self.policy = (
                self.flashinfer.MeasurementPolicy(execution_mode="cuda_graph")
                if objective == "graph"
                else None
            )
            try:
                with self.restore_policy_partitions(objective):
                    with (
                        self.torch.cuda.stream(self.model_runner.forward_stream),
                        self.policy_scope("tune", "late_" + objective),
                    ):
                        config, batches = self.tuner.prepare_tactic_profile(
                            inputs, self.retained["config"]
                        )
                        require(
                            len(batches) == 1
                            and all(
                                t is original
                                for t, original in zip(batches[0], inputs, strict=True)
                            ),
                            "public preparation replaced retained operands",
                        )
                        expected = dataclasses.replace(
                            self.retained["config"], use_cuda_graph=objective == "graph"
                        )
                        require(
                            config == expected,
                            "late objective changed more than graph execution",
                        )
                        group = {
                            "objective": objective,
                            "started_ns": time.perf_counter_ns(),
                            "settings": self.settings(config),
                            "rounds": [],
                        }
                        self.result["paired_profiles"].append(group)
                        handled = runner.precompile_tactics(
                            inputs, [-1, FIXED_DESCRIPTOR], None
                        )
                        if (
                            not handled
                            and "do_preparation"
                            in inspect.signature(runner.forward).parameters
                        ):
                            runner(inputs, tactic=-1, do_preparation=True)
                        self.torch.cuda.synchronize()
                        group["preparation"] = {
                            "precompile_handled": bool(handled),
                            "ended_ns": time.perf_counter_ns(),
                            "public_schedule_uses_exact_retained_objects": True,
                        }
                        for index, order in enumerate(self.pair_orders[objective]):
                            pair = {"index": index, "order": order, "measurements": []}
                            group["rounds"].append(pair)
                            for role in order:
                                self.stage = f"late_{objective}_{index:02d}_{role}"
                                inputs[4].fill_(float("nan"))
                                self.torch.cuda.synchronize()
                                self.allow_diagnostic = True
                                try:
                                    with patch.object(
                                        self.tuner,
                                        "_profile_single_kernel",
                                        self.diagnostic_profile,
                                    ):
                                        elapsed = self.tuner.profile_tactic(
                                            runner,
                                            inputs,
                                            FIXED_DESCRIPTOR
                                            if role == "winner"
                                            else -1,
                                            config,
                                            batches,
                                        )
                                finally:
                                    self.allow_diagnostic = False
                                check = self.check_profile_output(self.stage)
                                pair["measurements"].append(
                                    {
                                        "role": role,
                                        "tactic": list(FIXED_DESCRIPTOR)
                                        if role == "winner"
                                        else -1,
                                        "elapsed_ms": elapsed,
                                        "attempt_index": self.diagnostic_attempts - 1,
                                        "correctness_check": check,
                                    }
                                )
                        ratios = []
                        for pair in group["rounds"]:
                            measured = {
                                row["role"]: row["elapsed_ms"]
                                for row in pair["measurements"]
                            }
                            ratios.append(measured["winner"] / measured["default"])
                        group.update(
                            ended_ns=time.perf_counter_ns(),
                            explicit_over_default_median=statistics.median(ratios),
                        )
            finally:
                self.policy = None
            self.assert_bare()
        require(self.diagnostic_attempts == 48, "late profiler call budget changed")
        require(
            [self.tensor_hash(t) for t in inputs[:4]] == self.profile_read_hashes,
            "late profiler mutated read operands",
        )
        self.result["retained_profile"]["after_pairs_metadata"] = [
            self.metadata(t) for t in inputs
        ]

    def accept_profile_inputs(self):
        a, b, sa, sb, out, workspace = self.retained["inputs"]
        original = a.clone()
        rows = []
        for role in GRAPH_IDS:
            tactic = FIXED_DESCRIPTOR if role == "winner" else -1
            row = {
                "role": role,
                "tactic": tactic_json(tactic),
                "started_ns": time.perf_counter_ns(),
                "inputs": [self.metadata(t) for t in self.retained["inputs"]],
                "checks": [],
            }
            rows.append(row)

            def invoke():
                out.fill_(float("nan"))
                self.torch.cuda.synchronize()
                if tactic == -1:
                    self.native.bmm_fp8(a, b, out, sa, sb, workspace)
                    row.update(
                        used_descriptor=None,
                        descriptor_acceptance="not_applicable_default_route",
                    )
                else:
                    used = int(
                        self.native.bmm_fp8_run_with_descriptor(
                            a,
                            b,
                            out,
                            sa,
                            sb,
                            workspace,
                            self.gb._cublaslt_algo_arg(tactic),
                        )
                    )
                    row.update(
                        used_descriptor=used,
                        descriptor_acceptance="accepted"
                        if used == 1
                        else "rejected_fallback",
                    )
                    require(
                        used == 1,
                        "fixed descriptor fell back on exact retained synthetic problem",
                    )

            invoke()
            row["checks"].append(self.check_profile_output("profile_accept_" + role))
            a.zero_()
            invoke()
            row["checks"].append(
                self.check_profile_output(
                    "profile_accept_" + role + "_changed_input", zero=True
                )
            )
            a.copy_(original)
            invoke()
            row["checks"].append(
                self.check_profile_output("profile_accept_" + role + "_restored_input")
            )
            row["ended_ns"] = time.perf_counter_ns()
        require(
            [self.tensor_hash(t) for t in self.retained["inputs"][:4]]
            == self.profile_read_hashes,
            "synthetic native guards did not restore input bytes",
        )
        return rows

    def run(self):
        self.setup()
        try:
            with (
                patch.object(self.gb, "_FP8_GEMM_SM100_TUNING_CONFIG", self.config),
                patch.object(self.tuner, "_profile_single_kernel", self.profile),
            ):
                self.startup()
                self.capture_controls()
                if self.args.mode == "timing":
                    self.time_graphs()
                    self.result["fixed_explicit_over_default_graph_event_median"] = (
                        self.result["selected_over_default_graph_event_median"]
                    )
                self.paired_profiler()
                self.result["post_timing_acceptance"] = self.accept_actual(
                    self.graphs["winner"], "fixed_acceptance"
                )
                self.result["default_acceptance"] = self.accept_actual(
                    self.graphs["default"], "default_acceptance"
                )
                self.result["synthetic_acceptance"] = self.accept_profile_inputs()
                self.result["pre_trace_checks"] = self.final_checks("pre_trace")
                self.result["trace"] = self.trace()
                self.result["final"] = self.final_checks("final")
        finally:
            require(
                self.gb._FP8_GEMM_SM100_TUNING_CONFIG is self.original_config,
                "temporary initializer config leaked",
            )
        self.result.update(
            status="success",
            initializer_config_restored=True,
            synthetic_input_references_retained=True,
            profiling_byte_attestation="admission_full_hashes"
            if self.args.mode == "admission"
            else "metadata_only_startup_postgraph_read_hashes",
            persistence_under_test=False,
            all_samples_retained=True,
            queue_query_is_exclusion_criterion=False,
            declared_late_profile_attempts=self.diagnostic_attempts,
            total_actual_profiler_calls=self.profile_attempts
            + self.diagnostic_attempts,
            cache_guard_scope="Winner, file, protocol and bookkeeping fields inherited from the original worker; the measurement-config memoization cache is not included.",
        )
        return self.result


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), "refusing to overwrite worker result")
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = Experiment(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = (
            experiment.result
            if experiment is not None
            else {
                "mode": args.mode,
                "block": args.block,
                "trial": args.trial,
                "pid": os.getpid(),
            }
        )
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
