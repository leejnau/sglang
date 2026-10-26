"""Focused CPU controls for the shadow-finalist manual worker."""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args()
worker = a.repo / "test/manual/kernels/bench_flashinfer_finalist_confirmation.py"
sys.path.insert(0, str(worker.parent))
spec = importlib.util.spec_from_file_location("finalist_worker", worker)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
checks = []


def good(label, fn):
    fn()
    checks.append({"control": label, "passed": True})


def bad(label, fn):
    try:
        fn()
    except (RuntimeError, ValueError, KeyError, TypeError):
        checks.append({"control": label, "passed": True, "expected_rejection": True})
    else:
        raise AssertionError(label + " accepted")


def profile(index, tactic, elapsed, m=144, error=None):
    return {
        "index": index,
        "m": m,
        "tactic": tactic,
        "elapsed_ms": elapsed,
        "error": error,
    }


explicit_a = [1] * 8
explicit_b = [2] * 8
rows = [
    profile(72, -1, 0.001),
    profile(73, explicit_a, 0.003),
    profile(74, explicit_b, 0.002),
]
good(
    "actual_best_explicit_not_default_or_fixed_ordinal",
    lambda: w.require(w.best_explicit(rows)["tactic"] == explicit_b, "wrong finalist"),
)
rows.append(profile(75, [3] * 8, 0.002))
good(
    "first_observed_exact_tie_retained",
    lambda: w.require(w.best_explicit(rows)["index"] == 74, "unstable tie"),
)
for elapsed in (None, float("inf"), float("nan"), 0, -1):
    test = rows + [profile(76, [4] * 8, elapsed)]
    good(
        "invalid_elapsed_excluded_" + repr(elapsed),
        lambda test=test: w.require(
            w.best_explicit(test)["index"] == 74, "invalid finalist"
        ),
    )
good(
    "failed_measurement_excluded",
    lambda: w.require(
        w.best_explicit(rows + [profile(76, [4] * 8, 0.0001, error="failed")])["index"]
        == 74,
        "failed finalist",
    ),
)
good(
    "other_endpoint_excluded",
    lambda: w.require(
        w.best_explicit(rows + [profile(76, [4] * 8, 0.0001, m=128)])["index"] == 74,
        "wrong endpoint",
    ),
)
bad("no_explicit_candidate_rejected", lambda: w.best_explicit([profile(72, -1, 0.001)]))
bad(
    "all_explicit_failed_rejected",
    lambda: w.best_explicit([profile(73, explicit_a, float("inf"))]),
)


def rounds(ratios):
    return [
        {
            "index": i,
            "order": ["winner", "default"] if i % 2 == 0 else ["default", "winner"],
            "measurements": [
                {"role": "winner", "elapsed_ms": ratio},
                {"role": "default", "elapsed_ms": 1.0},
            ],
        }
        for i, ratio in enumerate(ratios)
    ]


for label, ratios, expected in (
    ("faster", [0.99] * 12, "winner"),
    ("slower", [1.01] * 12, "default"),
    ("exact_tie", [1.0] * 12, "default"),
    ("even_log_median_not_arithmetic_ratio", [0.5] * 6 + [1.8] * 6, "winner"),
):
    good(
        "decision_" + label,
        lambda ratios=ratios, expected=expected: w.require(
            w.decision_payload(rounds(ratios), tuple(explicit_b), 123)["decision_role"]
            == expected,
            "wrong log-median decision",
        ),
    )
for label, broken in (
    ("incomplete", rounds([1.0] * 11)),
    ("nonfinite", rounds([1.0] * 11 + [float("nan")])),
    ("zero", rounds([1.0] * 11 + [0.0])),
):
    bad(
        "decision_" + label,
        lambda broken=broken: w.decision_payload(broken, tuple(explicit_b), 123),
    )

# Use the real digest verification method, and reject changes to either raw
# timing evidence or any finalized decision field.
e = w.Experiment.__new__(w.Experiment)
e.explicit = tuple(explicit_b)
pairs = rounds([0.99] * 12)
payload = w.decision_payload(pairs, e.explicit, 123)
e.initial_decision_sha256 = w.decision_digest(payload)
e.result = {
    "confirmation": {
        **copy.deepcopy(payload),
        "rounds": pairs,
        "decision_payload": copy.deepcopy(payload),
        "decision_sha256": e.initial_decision_sha256,
        "initial_decision_sha256": e.initial_decision_sha256,
    }
}
good("decision_digest_positive", lambda: e.verify_decision())
baseline = copy.deepcopy(e.result)
for label, mutate in (
    (
        "raw_measurement",
        lambda g: g["rounds"][0]["measurements"][0].update(elapsed_ms=1.2),
    ),
    ("timestamp", lambda g: g.update(decided_ns=124)),
    ("choice", lambda g: g.update(decision_role="default")),
    ("initial_digest", lambda g: g.update(initial_decision_sha256="0" * 64)),
    ("payload", lambda g: g["decision_payload"].update(default_tactic=0)),
    ("digest", lambda g: g.update(decision_sha256="0" * 64)),
):
    e.result = copy.deepcopy(baseline)
    mutate(e.result["confirmation"])
    bad("final_decision_mutation_" + label, lambda: e.verify_decision())

e.result = copy.deepcopy(baseline)
changed_pairs = rounds([1.01] * 12)
changed_payload = w.decision_payload(changed_pairs, e.explicit, 123)
e.result["confirmation"].update(copy.deepcopy(changed_payload))
e.result["confirmation"].update(
    rounds=changed_pairs,
    decision_payload=copy.deepcopy(changed_payload),
    decision_sha256=w.decision_digest(changed_payload),
    initial_decision_sha256=w.decision_digest(changed_payload),
)
bad("coordinated_postdecision_evidence_rewrite_rejected", lambda: e.verify_decision())

batch = (
    worker.parent
    / "results/flashinfer_autotune_selection/20261026T230000Z-h100-finalist-confirmation"
)
if (batch / "plan.json").exists() and (batch / "schedule.json").exists():
    schedule = json.loads((batch / "schedule.json").read_text())
    for trial in range(8):
        row = next(row for row in schedule["entries"] if row["block"] == trial)
        args = NS(
            mode="timing",
            trial=trial,
            block=trial,
            cell="auto_synthetic",
            capture_order=row["capture_order"],
            plan=batch / "plan.json",
            schedule=batch / "schedule.json",
            historical_contract=batch / "historical-contract.json",
        )
        good(
            "actual_frozen_assignment_" + str(trial),
            lambda args=args: w.Experiment(args),
        )
    broken = copy.deepcopy(schedule)
    broken["confirmation_pair_schedules"]["0"] = [["winner", "default"]] * 12
    bad(
        "unbalanced_confirmation_schedule_rejected",
        lambda: w.confirmation_orders(broken, 0),
    )
    broken = copy.deepcopy(schedule)
    broken["confirmation_pair_schedules"]["0"][0] = ["winner", "winner"]
    bad(
        "duplicate_confirmation_role_rejected", lambda: w.confirmation_orders(broken, 0)
    )
else:
    raise RuntimeError("actual plan/schedule required for complete CPU controls")


# Exercise actual run dispatch, proving confirmation always precedes captures,
# both graphs are retained regardless hypothetical choice, and exceptions stop
# before serving graph capture. This substitutes operations, not CUDA evidence.
def run_experiment(mode="timing", decision="default", failure=False):
    e = w.Experiment.__new__(w.Experiment)
    log = []
    e.args = NS(mode=mode)
    e.result = {}
    e.graphs = {}
    e.profile_attempts = 81
    e.diagnostic_attempts = 24
    e.original_config = object()
    e.config = object()
    e.gb = NS(_FP8_GEMM_SM100_TUNING_CONFIG=e.original_config)
    e.tuner = NS(_profile_single_kernel=lambda: None)
    e.setup = lambda: log.append("setup")
    e.startup = lambda: log.append("startup81")

    def confirm():
        log.append("confirmation24")
        if failure:
            raise RuntimeError("deliberate confirmation failure")
        e.result["confirmation"] = {"decision_role": decision}

    e.confirm = confirm

    def capture():
        w.require(log[-1] == "confirmation24", "capture preceded confirmation")
        e.graphs = {"winner": {"id": "winner"}, "default": {"id": "default"}}
        log.append("capture_both")

    e.capture_controls = capture

    def timing():
        w.require(set(e.graphs) == {"winner", "default"}, "missing graph")
        log.append("time_both")
        e.result["selected_over_default_graph_event_median"] = 1.02

    e.time_graphs = timing
    e.accept_actual = lambda saved, label: log.append("native_" + saved["id"]) or {}
    e.accept_profile_inputs = lambda: log.append("synthetic_native") or {}
    e.final_checks = lambda prefix: log.append(prefix + "_guard") or {}
    e.trace = lambda: log.append("trace_after_all_timing") or {}
    e.verify_decision = lambda: "frozen_digest"
    return e, log


for decision in ("winner", "default"):
    e, log = run_experiment(decision=decision)

    def run_check(e=e, log=log):
        result = e.run()
        w.require(
            log
            == [
                "setup",
                "startup81",
                "confirmation24",
                "capture_both",
                "time_both",
                "native_winner",
                "native_default",
                "synthetic_native",
                "pre_trace_guard",
                "trace_after_all_timing",
                "final_guard",
            ],
            "wrong sequence",
        )
        w.require(
            result["total_actual_profiler_calls"] == 105
            and e.gb._FP8_GEMM_SM100_TUNING_CONFIG is e.original_config,
            "budget or config leakage",
        )

    good("actual_run_sequence_both_graphs_decision_" + decision, run_check)
e, log = run_experiment(mode="admission")
good(
    "admission_confirms_without_latency_cohort",
    lambda: w.require(
        e.run()["status"] == "success"
        and "confirmation24" in log
        and "time_both" not in log,
        "wrong admission path",
    ),
)
e, log = run_experiment(failure=True)


def failure_check():
    try:
        e.run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("failure swallowed")
    w.require(
        log == ["setup", "startup81", "confirmation24"]
        and e.gb._FP8_GEMM_SM100_TUNING_CONFIG is e.original_config,
        "failed confirmation captured graph or leaked config",
    )


good("confirmation_failure_stops_and_restores", failure_check)


# Real late-profile guard: dynamic finalist is accepted, another descriptor is
# rejected, and denial/invalid durations remain recorded in the call budget.
def diagnostic():
    e = w.Experiment.__new__(w.Experiment)
    e.diagnostic_attempts = 0
    e.result = {"diagnostic_profile_attempts": []}
    e.stage = "confirmation_00_winner"
    tensors = [object() for _ in range(6)]
    runner = object()
    e.explicit = tuple(explicit_b)
    e.retained = {"runner": runner, "inputs": tuple(tensors)}
    e.metadata = lambda t: {"object": id(t)}
    e.historical = {"original_settings": {"contract": "original_auto"}}
    e.settings = lambda config: {"contract": "original_auto"}
    e.policy = None
    e.allow_diagnostic = True
    e.allow_profiling = False
    e.tuner = NS(_in_v2_context=True, is_tuning_mode=True)
    e.model_runner = NS(forward_stream="forward")
    e.torch = NS(cuda=NS(current_stream=lambda: "forward"))
    e.original_profile = lambda *a, **kw: 0.01
    return e, tensors, runner


e, tensors, runner = diagnostic()
good(
    "dynamic_finalist_real_profiler_delegation",
    lambda: w.require(
        e.diagnostic_profile(
            runner, tensors, e.explicit, object(), input_tensor_batches=[tensors]
        )
        == 0.01,
        "not delegated",
    ),
)
for label, change in (
    ("undeclared", lambda e: setattr(e, "allow_diagnostic", False)),
    ("startup_overlap", lambda e: setattr(e, "allow_profiling", True)),
    ("explicit_policy", lambda e: setattr(e, "policy", object())),
    (
        "timer_changed",
        lambda e: setattr(e, "settings", lambda config: {"contract": "different"}),
    ),
    ("wrong_stream", lambda e: setattr(e.model_runner, "forward_stream", "other")),
):
    e, tensors, runner = diagnostic()
    change(e)
    bad(
        "diagnostic_" + label,
        lambda e=e, tensors=tensors, runner=runner: e.diagnostic_profile(
            runner, tensors, e.explicit, object(), input_tensor_batches=[tensors]
        ),
    )
    good(
        "denial_recorded_" + label,
        lambda e=e: w.require(
            e.diagnostic_attempts == 1
            and e.result["diagnostic_profile_attempts"][0]["error"],
            "denial lost",
        ),
    )
e, tensors, runner = diagnostic()
bad(
    "other_descriptor_rejected",
    lambda: e.diagnostic_profile(
        runner, tensors, tuple(explicit_a), object(), input_tensor_batches=[tensors]
    ),
)
e, tensors, runner = diagnostic()
replaced = list(tensors)
replaced[0] = object()
bad(
    "replaced_profiling_operand_rejected",
    lambda: e.diagnostic_profile(
        runner, replaced, e.explicit, object(), input_tensor_batches=[replaced]
    ),
)
e, tensors, runner = diagnostic()
e.original_profile = lambda *a, **kw: float("nan")
bad(
    "nonfinite_profiler_result_retained",
    lambda: e.diagnostic_profile(
        runner, tensors, e.explicit, object(), input_tensor_batches=[tensors]
    ),
)

report = {
    "status": "success",
    "count": len(checks),
    "worker_sha256": w.sha(worker),
    "controls": checks,
    "scope": "CPU-only actual new worker branches; inherited CUDA/numerical/native behavior remains for the disposable admission gate.",
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": "success", "count": len(checks), "output": str(a.output)}))
