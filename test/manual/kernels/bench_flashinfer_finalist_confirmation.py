"""Shadow confirmation of the actual best explicit startup finalist.

The original startup search remains unchanged. Balanced confirmation uses its
retained synthetic operands before either serving graph is captured. The
decision is hypothetical: both choices are evaluated using the same two later
graphs, and the original cache winner is preserved. Research only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch

from bench_flashinfer_fixed_pair import Experiment as FixedPairExperiment
from bench_flashinfer_objective_values import GRAPH_IDS
from bench_flashinfer_objective_values import Experiment as OriginalExperiment
from bench_flashinfer_objective_values import require, sha, tactic_json

FIXED_PARENT_SHA = "7c8c743fe08b898035bbdf693123ea7b0c8a2d49a29db801f0db3c81178abf42"
ORIGINAL_PARENT_SHA = "979e28aeb3248b27cb27df21b98cf274442c0e6b92c4df3444c48f7214ce90ba"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("admission", "timing"), required=True)
    parser.add_argument("--trial", type=int, choices=range(8), required=True)
    parser.add_argument(
        "--capture-order", choices=("winner-first", "default-first"), required=True
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


def best_explicit(profiles):
    candidates = [
        row
        for row in profiles
        if row["m"] == 144
        and row["tactic"] != -1
        and isinstance(row["elapsed_ms"], (int, float))
        and math.isfinite(row["elapsed_ms"])
        and row["elapsed_ms"] > 0
        and row.get("error") is None
    ]
    require(candidates, "no finite explicit runtime finalist")
    return min(candidates, key=lambda row: row["elapsed_ms"])


def confirmation_orders(schedule, block):
    orders = schedule["confirmation_pair_schedules"][str(block)]
    require(
        len(orders) == 12
        and all(sorted(order) == sorted(GRAPH_IDS) for order in orders)
        and sum(order[0] == "winner" for order in orders) == 6,
        "confirmation schedule is not twelve balanced pairs",
    )
    return orders


def decision_payload(rounds, explicit, decided_ns):
    require(len(rounds) == 12, "confirmation decision requires all twelve pairs")
    ratios = []
    for pair in rounds:
        rows = {row["role"]: row for row in pair["measurements"]}
        require(
            len(pair["measurements"]) == 2
            and set(rows) == set(GRAPH_IDS)
            and all(
                isinstance(row["elapsed_ms"], (int, float))
                and math.isfinite(row["elapsed_ms"])
                and row["elapsed_ms"] > 0
                for row in rows.values()
            ),
            "decision received an incomplete or invalid profiler pair",
        )
        ratios.append(
            math.log(rows["winner"]["elapsed_ms"] / rows["default"]["elapsed_ms"])
        )
    median = statistics.median(ratios)
    role = "winner" if median < 0 else "default"
    return {
        "explicit_tactic": tactic_json(explicit),
        "default_tactic": -1,
        "pair_orders": [pair["order"] for pair in rounds],
        "log_ratios": ratios,
        "median_log_ratio": median,
        "decision_role": role,
        "decision_tactic": tactic_json(explicit) if role == "winner" else -1,
        "decided_ns": decided_ns,
    }


def decision_digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class Experiment(FixedPairExperiment):
    def __init__(self, args):
        # Initialize the common public-facade harness without the old fixed
        # descriptor or its two-objective protocol.
        OriginalExperiment.__init__(self, args)
        self.historical = json.loads(args.historical_contract.read_text())
        require(
            sha(args.historical_contract) == self.plan["historical_contract_sha256"],
            "original search contract changed",
        )
        fixed_parent, original_parent = (
            Path(inspect.getfile(FixedPairExperiment)),
            Path(inspect.getfile(OriginalExperiment)),
        )
        require(
            sha(fixed_parent) == FIXED_PARENT_SHA
            and sha(original_parent) == ORIGINAL_PARENT_SHA,
            "common research helper source changed",
        )
        self.confirmation_orders = confirmation_orders(self.schedule, args.block)
        self.retained = None
        self.diagnostic_attempts, self.allow_diagnostic = 0, False
        self.result.update(
            study="best_explicit_confirmation",
            worker_sha256=sha(__file__),
            parent_worker_sha256=sha(original_parent),
            fixed_pair_helper_sha256=sha(fixed_parent),
            historical_contract_sha256=sha(args.historical_contract),
            graph_role_alias={"winner": "best_explicit", "default": "untuned_default"},
            diagnostic_profile_attempts=[],
            profile_correctness=[],
            cache_partition_restorations=[],
            startup_choices=[],
            limits=[
                "Retaining final-profile operands extends allocation lifetime after index80; this is not byte-identical historical execution.",
                "Synthetic FP32 reference/hash preparation and numerical guards precede shadow confirmation and alter its history; original81 startup measurements remain untouched.",
                "Both natural and hypothetical choices are evaluated on the SAME two post-confirmation graphs, not untouched startup graphs versus changed-history graphs.",
                "Serving graphs use captured activations plus quantization; confirmation uses synthetic profiling activations. Trace evidence observes serving graphs, not every profiling launch.",
                "Confirmation is a hypothetical decision; it never publishes a new winner or changes serving configuration.",
            ],
        )

    def startup(self):
        def record_choice(op, runners, config, inputs, **kwargs):
            require(
                op == "fp8_gemm"
                and len(runners) == 1
                and type(runners[0]).__name__ == "CublasFp8GemmRunner"
                and not kwargs,
                "startup public runner contract changed",
            )
            returned = self.original_choose(op, runners, config, inputs, **kwargs)
            require(
                len(returned) == 2 and returned[0] is runners[0],
                "startup chooser returned another runner",
            )
            key = self.tuner._get_cache_key(
                op,
                returned[0],
                tuple(tuple(t.shape) for t in inputs),
                config,
                returned[0].get_cache_key_extras(inputs),
            )
            self.result["startup_choices"].append(
                {
                    "op": op,
                    "tactic": tactic_json(returned[1]),
                    "key": key.file_key,
                    "profile_attempts": self.profile_attempts,
                    "requested_runner_identity": id(runners[0]),
                    "returned_runner_identity": id(returned[0]),
                    "returned_ns": time.perf_counter_ns(),
                }
            )
            return returned

        with patch.object(self.tuner, "choose_one", record_choice):
            OriginalExperiment.startup(self)
        rows = self.result["startup"]["profiles"]
        choices = self.result["startup_choices"]
        require(
            self.profile_attempts == 81
            and self.retained is not None
            and len(choices) == 1
            and choices[0]["profile_attempts"] == 81
            and self.result["startup"]["candidate_order"]
            == self.historical["candidate_order"]
            and [row["key"] for row in rows] == self.historical["profile_keys"]
            and all(
                row["settings"] == self.historical["original_settings"] for row in rows
            )
            and rows[72]["tactic"] == -1,
            "original startup candidate order, keys, settings or public selection changed",
        )
        metadata = self.result["retained_profile"]["metadata"]
        require(
            all(row["metadata"] == metadata for row in rows[72:81])
            and [self.metadata(t) for t in self.retained["inputs"]] == metadata
            and not self.retained["more"]
            and set(self.retained["kwargs"]) <= {"input_tensor_batches"},
            "runtime candidates did not share the same retained operands",
        )
        explicit = best_explicit(rows)
        self.explicit = tuple(explicit["tactic"])
        self.result["finalist"] = {
            "tactic": tactic_json(self.explicit),
            "profile_index": explicit["index"],
            "key": explicit["key"],
            "elapsed_ms": explicit["elapsed_ms"],
            "selection_rule": "first finite minimum in observed runtime explicit candidate order",
        }
        natural = min(rows[72:81], key=lambda row: row["elapsed_ms"])
        require(
            choices[0]["op"] == "fp8_gemm"
            and choices[0]["tactic"] == natural["tactic"]
            and natural["tactic"] in (-1, tactic_json(self.explicit)),
            "actual startup return differs from the first runtime minimum",
        )
        self.expected_natural = (
            -1 if natural["tactic"] == -1 else tuple(natural["tactic"])
        )
        self.result["natural"] = {
            "tactic": natural["tactic"],
            "key": choices[0]["key"],
            "first_pass_profile_index": natural["index"],
            "source": "actual original startup choose_one return",
            "frozen_ns": time.perf_counter_ns(),
            "cache_sha256": self.cache_sha,
            "cache_state_sha256": self.memory_sha,
        }
        self.result["original_history"] = {
            "default_index": 72,
            "explicit_index": explicit["index"],
            "default_ms": rows[72]["elapsed_ms"],
            "explicit_ms": explicit["elapsed_ms"],
            "explicit_over_default": explicit["elapsed_ms"] / rows[72]["elapsed_ms"],
            "actual_m": 144,
            "canonical_cache_m": 256,
            "all_81_candidate_order_keys_settings_match": True,
        }

    def diagnostic_profile(self, runner, inputs, tactic, config, *more, **kwargs):
        self.diagnostic_attempts += 1
        row = {
            "index": self.diagnostic_attempts - 1,
            "stage": self.stage,
            "objective": "auto",
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
                and tactic in (-1, self.explicit)
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
                and row["settings"] == self.historical["original_settings"]
                and self.policy is None
                and self.tuner._in_v2_context
                and self.tuner.is_tuning_mode
                and self.torch.cuda.current_stream()
                == self.model_runner.forward_stream,
                "undeclared confirmation profiler call, operands or settings",
            )
            elapsed = self.original_profile(runner, inputs, tactic, config, **kwargs)
            require(
                isinstance(elapsed, (int, float))
                and math.isfinite(elapsed)
                and elapsed > 0,
                "invalid confirmation duration",
            )
            row.update(elapsed_ms=float(elapsed), ended_ns=time.perf_counter_ns())
            return elapsed
        except Exception as error:
            row.update(
                error=f"{type(error).__name__}: {error}",
                ended_ns=time.perf_counter_ns(),
            )
            raise

    def confirm(self):
        self.prepare_reference()
        inputs, runner = list(self.retained["inputs"]), self.retained["runner"]
        self.policy = None
        with self.restore_policy_partitions("auto"):
            with (
                self.torch.cuda.stream(self.model_runner.forward_stream),
                self.policy_scope("tune", "confirmation"),
            ):
                config, batches = self.tuner.prepare_tactic_profile(
                    inputs, self.retained["config"]
                )
                require(
                    config == self.retained["config"]
                    and len(batches) == 1
                    and all(
                        t is original
                        for t, original in zip(batches[0], inputs, strict=True)
                    ),
                    "confirmation preparation replaced operands or auto configuration",
                )
                group = {
                    "objective": "auto",
                    "started_ns": time.perf_counter_ns(),
                    "settings": self.settings(config),
                    "rounds": [],
                }
                self.result["confirmation"] = group
                handled = runner.precompile_tactics(inputs, [-1, self.explicit], None)
                if (
                    not handled
                    and "do_preparation" in inspect.signature(runner.forward).parameters
                ):
                    runner(inputs, tactic=-1, do_preparation=True)
                self.torch.cuda.synchronize()
                group["preparation"] = {
                    "precompile_handled": bool(handled),
                    "ended_ns": time.perf_counter_ns(),
                    "public_schedule_uses_exact_retained_objects": True,
                }
                for index, order in enumerate(self.confirmation_orders):
                    pair = {"index": index, "order": order, "measurements": []}
                    group["rounds"].append(pair)
                    for role in order:
                        self.stage = f"confirmation_{index:02d}_{role}"
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
                                    self.explicit if role == "winner" else -1,
                                    config,
                                    batches,
                                )
                        finally:
                            self.allow_diagnostic = False
                        check = self.check_profile_output(self.stage)
                        pair["measurements"].append(
                            {
                                "role": role,
                                "tactic": tactic_json(self.explicit)
                                if role == "winner"
                                else -1,
                                "elapsed_ms": elapsed,
                                "attempt_index": self.diagnostic_attempts - 1,
                                "correctness_check": check,
                            }
                        )
                group["ended_ns"] = time.perf_counter_ns()
        self.assert_bare()
        require(
            self.diagnostic_attempts == 24
            and self.cache_state_sha() == self.memory_sha
            and sha(self.args.cache) == self.cache_sha
            and [self.tensor_hash(t) for t in inputs[:4]] == self.profile_read_hashes,
            "confirmation changed call budget, startup cache or read operands",
        )
        payload = decision_payload(
            group["rounds"], self.explicit, time.perf_counter_ns()
        )
        group.update(payload)
        group["decision_payload"] = json.loads(json.dumps(payload))
        group["decision_sha256"] = decision_digest(payload)
        self.initial_decision_sha256 = group["decision_sha256"]
        group["initial_decision_sha256"] = self.initial_decision_sha256
        self.result["retained_profile"]["after_confirmation_metadata"] = [
            self.metadata(t) for t in inputs
        ]
        self.verify_decision()

    def verify_decision(self):
        group = self.result["confirmation"]
        expected = decision_payload(group["rounds"], self.explicit, group["decided_ns"])
        require(
            group["decision_payload"] == expected
            and all(group[key] == value for key, value in expected.items())
            and decision_digest(expected) == group["decision_sha256"],
            "shadow decision changed after it was finalized",
        )
        require(
            group["decision_sha256"]
            == group["initial_decision_sha256"]
            == self.initial_decision_sha256,
            "shadow decision no longer matches its pre-capture digest",
        )
        return group["decision_sha256"]

    def capture_controls(self):
        self.verify_decision()
        require(
            self.cache_state_sha() == self.memory_sha
            and sha(self.args.cache) == self.cache_sha,
            "startup cache changed before post-confirmation natural lookup",
        )
        self.result["pre_capture_cache_state_sha256"] = self.cache_state_sha()
        self.result["pre_capture_cache_state_field_sha256"] = dict(
            self.cache_state_field_sha256
        )
        with self.policy_scope("replay", "natural_lookup"):
            saved = {}
            with patch.object(
                self.tuner,
                "choose_one",
                self.choose_recorder(
                    self.expected_natural, "memory", "natural_lookup", saved
                ),
            ):
                out = self.operation()
            self.check(out, "natural_lookup")
        require(
            saved["key"].file_key == self.result["natural"]["key"]
            and self.cache_state_sha() == self.memory_sha
            and sha(self.args.cache) == self.cache_sha,
            "actual natural hit changed startup selection or cache",
        )
        self.result["natural"].update(
            cache_hit=True,
            lookup_tier="memory",
            confirmed_after_ns=time.perf_counter_ns(),
            cache_preserved_after_confirmation=True,
        )
        order = (
            GRAPH_IDS
            if self.args.capture_order == "winner-first"
            else tuple(reversed(GRAPH_IDS))
        )
        for graph_id in order:
            with self.policy_scope("replay", graph_id + "_capture"), self.miss_scope():
                if graph_id == "winner":
                    self.tuner._cache_profile_result(
                        saved["key"], None, saved["config"], self.explicit
                    )
                self.capture(
                    graph_id,
                    self.explicit if graph_id == "winner" else -1,
                    "memory" if graph_id == "winner" else "miss",
                )
                self.result["graphs"][-1]["confirmation_decision_sha256"] = (
                    self.verify_decision()
                )
        require(
            len({saved["output"].data_ptr() for saved in self.graphs.values()}) == 2,
            "graphs share an output allocation",
        )
        self.result["graph_choices"] = {
            "natural_role": "default" if self.expected_natural == -1 else "winner",
            "confirmation_role": self.result["confirmation"]["decision_role"],
            "both_evaluated_on_same_retained_graphs": True,
        }
        self.assert_bare()

    def accept_profile_inputs(self):
        a, b, sa, sb, out, workspace = self.retained["inputs"]
        original = a.clone()
        rows = []
        for role in GRAPH_IDS:
            tactic = self.explicit if role == "winner" else -1
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
                        "best explicit fell back on retained profiling problem",
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
            "synthetic guard did not restore input bytes",
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
                self.confirm()
                self.capture_controls()
                if self.args.mode == "timing":
                    self.time_graphs()
                    self.result["best_explicit_over_default_graph_event_median"] = (
                        self.result["selected_over_default_graph_event_median"]
                    )
                self.result["post_timing_acceptance"] = self.accept_actual(
                    self.graphs["winner"], "explicit_acceptance"
                )
                self.result["default_acceptance"] = self.accept_actual(
                    self.graphs["default"], "default_acceptance"
                )
                self.result["synthetic_acceptance"] = self.accept_profile_inputs()
                self.result["pre_trace_checks"] = self.final_checks("pre_trace")
                self.result["trace"] = self.trace()
                self.result["final"] = self.final_checks("final")
                self.result["final"]["decision_sha256"] = self.verify_decision()
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
            else "metadata_only_startup_poststartup_read_hashes",
            persistence_under_test=False,
            all_samples_retained=True,
            queue_query_is_exclusion_criterion=False,
            declared_late_profile_attempts=self.diagnostic_attempts,
            total_actual_profiler_calls=self.profile_attempts
            + self.diagnostic_attempts,
            cache_guard_scope="Inherited winner, file, protocol and bookkeeping fields; measurement-config memoization excluded.",
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
