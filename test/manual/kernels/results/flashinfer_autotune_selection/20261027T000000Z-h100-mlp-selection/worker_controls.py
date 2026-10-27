"""Focused CPU controls for the actual MLP selection study worker."""

import argparse
import contextlib
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args()
worker = a.repo / "test/manual/kernels/bench_flashinfer_mlp_selection.py"
sys.path.insert(0, str(worker.parent))
spec = importlib.util.spec_from_file_location("mlp_worker", worker)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
checks = []


def good(label, fn):
    fn()
    checks.append({"control": label, "passed": True})


def bad(label, fn):
    try:
        fn()
    except (RuntimeError, ValueError, TypeError, KeyError):
        checks.append({"control": label, "passed": True, "expected_rejection": True})
    else:
        raise AssertionError(label + " accepted")


batch = (
    worker.parent
    / "results/flashinfer_autotune_selection/20261027T000000Z-h100-mlp-selection"
)
schedule = json.loads((batch / "schedule.json").read_text())
for row in schedule["entries"]:
    args = NS(**row)
    good(
        "actual_assignment_" + row["id"],
        lambda args=args, row=row: w.require(
            w.assignment(schedule, args)[0] == row, "wrong assignment"
        ),
    )
for field, value in (
    ("capture_order", "invalid"),
    ("mode_order", "invalid"),
    ("arm", "invalid"),
):
    args = NS(**schedule["entries"][0])
    setattr(args, field, value)
    bad("wrong_assignment_" + field, lambda args=args: w.assignment(schedule, args))
for mode in w.MODES:
    broken = copy.deepcopy(schedule)
    broken["round_schedules"]["0"][mode] = [["natural", "default"]] * 12
    bad(
        "unbalanced_" + mode,
        lambda broken=broken: w.assignment(broken, NS(**schedule["entries"][0])),
    )


# Real event/wall sampler dispatch with CPU stand-ins: operation calls appear
# only between primed start/end events. No lookup/native hook is run there.
def batch_case(event_ms=24.0):
    log = []

    class Event:
        def __init__(self, enable_timing):
            self.index = len([x for x in log if x == "new"])
            log.append("new")

        def record(self):
            log.append("record" + str(self.index))

        def synchronize(self):
            log.append("sync" + str(self.index))

        def elapsed_time(self, other):
            return event_ms

    torch = NS(cuda=NS(Event=Event))
    with patch.object(w.time, "perf_counter_ns", side_effect=[1_000_000, 25_000_000]):
        result = w.pure_batch(torch, lambda: log.append("operation"), 4)
    return result, log


result, log = batch_case()
good(
    "actual_sampler_calls_and_priming_order",
    lambda: w.require(
        log
        == [
            "new",
            "new",
            "record0",
            "record1",
            "sync1",
            "record0",
            "operation",
            "operation",
            "operation",
            "operation",
            "record1",
            "sync1",
        ],
        "wrong event or operation order",
    ),
)
good(
    "actual_sampler_normalization",
    lambda: w.require(
        result["iterations"] == 4
        and result["wall_batch_ms"] == 24
        and result["gpu_event_batch_ms"] == 24
        and result["wall_per_call_ms"] == result["gpu_event_per_call_ms"] == 6,
        "wrong normalization",
    ),
)
for duration in (0, -1, float("nan"), float("inf")):
    bad(
        "invalid_event_duration_" + str(duration),
        lambda duration=duration: batch_case(duration),
    )
for iterations in (0, -1, w.MAX_ITERATIONS + 1, True):
    bad(
        "invalid_iteration_count_" + str(iterations),
        lambda iterations=iterations: w.pure_batch(NS(), lambda: None, iterations),
    )

# Real inherited cache-miss scope: restore both memory and file winners even
# when a default measurement fails; never save a control winner to disk.
with tempfile.TemporaryDirectory() as temporary:
    e = w.Experiment.__new__(w.Experiment)
    e.args = NS(cache=Path(temporary) / "cache.json")
    e.args.cache.write_text('{"natural":"unchanged"}')
    fields = (
        "profiling_cache",
        "_winner_partitions",
        "_profiling_cache_policies",
        "_file_configs",
        "_file_config_policies",
        "_invalidated_file_keys",
        "_excluded_file_keys",
        "_logged_file_hits",
        "_logged_cache_miss_oor",
        "_logged_invalid_tactic",
    )
    e.tuner = NS(**{name: {} for name in fields}, _dirty=False, _dirty_seq=0)
    e.tuner._winner_cache = lambda: e.tuner.profiling_cache
    e.tuner.profiling_cache["winner"] = ("descriptor", None)
    e.tuner._file_configs["winner"] = ("runner", "descriptor")
    e.result = {"miss_scopes": []}
    original = {name: copy.deepcopy(getattr(e.tuner, name)) for name in fields}
    before = w.sha(e.args.cache)

    def restored():
        try:
            with e.role_scope("default"):
                w.require(
                    not e.tuner._winner_cache() and not e.tuner._file_configs,
                    "not a genuine miss",
                )
                e.tuner._logged_file_hits["temporary"] = True
                raise LookupError("deliberate sample failure")
        except LookupError:
            pass
        w.require(
            all(getattr(e.tuner, name) == value for name, value in original.items())
            and w.sha(e.args.cache) == before,
            "cache not restored",
        )

    good("default_measurement_failure_restores_real_cache_fields", restored)

    def natural_scope():
        with e.role_scope("natural"):
            w.require(
                e.tuner._winner_cache() == original["profiling_cache"]
                and e.tuner._file_configs == original["_file_configs"],
                "natural scope erased winner",
            )

    good("natural_scope_preserves_actual_lookup_state", natural_scope)

# Warm/default profiling is denied before original CUDA profiler execution.
e = w.Experiment.__new__(w.Experiment)
e.profile_attempts = 0
e.allow_profiling = False
e.stage = "startup"
e.profiles = []
e.forbidden_profile_attempts = []
e.original_profile = lambda *a, **kw: (_ for _ in ()).throw(
    AssertionError("original profiler was called")
)
bad("warm_unexpected_profiling_rejected", lambda: e.profile_hook(None, [], -1, None))
good(
    "warm_forbidden_attempt_counted_before_throw",
    lambda: w.require(
        e.profile_attempts == 1 and e.forbidden_profile_attempts == ["startup"],
        "lost swallowed-attempt evidence",
    ),
)

# Actual new startup observer checks context flags without reconstructing
# or substituting the parent chooser's returned tuple.
for arm in ("default", "v1", "v2"):
    e = w.Experiment.__new__(w.Experiment)
    e.args = NS(arm=arm)
    e.stage = "startup"
    e.selections = []
    e.tuner = NS(
        _effective_measure_policy=None,
        _in_v2_context=arm == "v2",
        is_tuning_mode=arm != "default",
        _active_tuning_contexts=1,
        _active_managed_store=None,
    )
    returned = (object(), -1)

    def parent_choose(self, *args, **kwargs):
        self.selections.append({})
        return returned

    with patch.object(w.LinearLayerExperiment, "choose_hook", parent_choose):
        good(
            "actual_arm_context_" + arm,
            lambda e=e: w.require(
                e.choose_hook(None, None, None, None) is returned,
                "chooser result changed",
            ),
        )
        e.tuner._in_v2_context = not e.tuner._in_v2_context
        bad(
            "wrong_actual_arm_context_" + arm,
            lambda e=e: e.choose_hook(None, None, None, None),
        )

# Qualified native identity must survive inherited setup, which resets its
# cached native_module reference. Replacing both reset references still fails.
e = w.Experiment.__new__(w.Experiment)
native = object()
replacement = object()
e.qualified_native_module = native
e.native_module = native
e.native_cell = NS(cell_contents=native)
e.tuner = NS(_profile_single_kernel=lambda: None)


def replace_provider():
    e.native_module = replacement
    e.native_cell.cell_contents = replacement


e.setup = replace_provider
bad("setup_cannot_redefine_qualified_provider_identity", lambda: e.run())


# The owner guard binds every graph check to the retained native output view.
class Tensor:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


for natural_tactic in (-1, (1,) * 8):
    e = w.Experiment.__new__(w.Experiment)
    e.natural_tactic = natural_tactic
    e.args = NS(arm="v2", phase="warm")
    good(
        "natural_tactic_preserved_" + str(natural_tactic),
        lambda e=e: w.require(
            e.role_tactic("natural") == e.natural_tactic
            and e.role_tactic("default") == -1,
            "tactic substitution",
        ),
    )
    good(
        "warm_natural_file_default_miss_" + str(natural_tactic),
        lambda e=e: w.require(
            e.role_tier("natural") == "file" and e.role_tier("default") == "miss",
            "wrong lookup tiers",
        ),
    )
e = w.Experiment.__new__(w.Experiment)
e.out = Tensor(123)
e.graphs = {
    "natural": {"binding": NS(output=Tensor(123)), "inputs": [None] * 4 + [Tensor(123)]}
}
good("actual_retained_owner_positive", lambda: e.check_owner("natural"))
e.out = Tensor(456)
bad("different_graph_output_rejected", lambda: e.check_owner("natural"))
e.out = Tensor(123)
e.graphs["natural"]["inputs"][4] = Tensor(456)
bad("wrong_native_output_view_rejected", lambda: e.check_owner("natural"))

# Actual calibration/pair dispatch uses a common iteration count, chosen from
# BOTH candidates. Admission executes exactly the same measurement method.
e = w.Experiment.__new__(w.Experiment)
e.args = NS(mode_order="eager-first", iterations=20, min_sample_ms=20)
e.orders = schedule["round_schedules"]["0"]
e.result = {"measurements": {}}
e.role_scope = lambda role: contextlib.nullcontext()
e.callable = lambda mode, role: lambda: None
e.warm = lambda call, ms: {"actual_ms": ms, "calls": 256}


def sample(mode, role, iterations, label):
    per = 0.5 if role == "natural" else 0.25
    return {
        "role": role,
        "iterations": iterations,
        "wall_batch_ms": iterations * per,
        "gpu_event_batch_ms": iterations * per,
    }


e.sample = sample
e.measure_modes()
for mode in w.MODES:
    group = e.result["measurements"][mode]
    expected = 16 if mode == "cuda_graph" else 80
    good(
        "common_iterations_cover_both_" + mode,
        lambda group=group, expected=expected: w.require(
            group["common_iterations"] == expected
            and len(group["pairs"]) == 12
            and all(
                m["iterations"] == expected
                for p in group["pairs"]
                for m in p["measurements"]
            ),
            "role-specific or incomplete sampling",
        ),
    )
    good(
        "actual_balanced_pair_order_" + mode,
        lambda group=group, mode=mode: w.require(
            [p["order"] for p in group["pairs"]] == e.orders[mode]
            and all(
                [m["role"] for m in p["measurements"]] == p["order"]
                for p in group["pairs"]
            ),
            "wrong order",
        ),
    )


# New graph primary: delay before start event, sixteen actual replays, query
# immediately after enqueue and before sync. Both bool outcomes remain valid.
def queued_case(outcome=False, query_error=False):
    log = []

    class Event:
        def __init__(self, enable_timing):
            self.index = len([x for x in log if x == "new"])
            log.append("new")

        def record(self):
            log.append("record" + str(self.index))

        def synchronize(self):
            log.append("sync" + str(self.index))

        def elapsed_time(self, other):
            return 0.096

        def query(self):
            log.append("query" + str(self.index))
            if query_error:
                raise RuntimeError("deliberate query failure")
            return outcome

    torch = NS(
        cuda=NS(Event=Event, _sleep=lambda cycles: log.append("sleep" + str(cycles)))
    )
    progress = {}
    try:
        with patch.object(
            w.time, "perf_counter_ns", side_effect=[1000, 2000, 3000, 1000000]
        ):
            result = w.queued_graph_batch(torch, lambda: log.append("replay"), progress)
    except RuntimeError:
        return None, log, progress
    return result, log, progress


for outcome in (False, True):
    result, log, progress = queued_case(outcome)
    good(
        "graph_query_all_outcomes_retained_" + str(outcome),
        lambda result=result, outcome=outcome: w.require(
            result["start_event_complete_after_enqueue"] is outcome
            and result["queue_query_is_exclusion_criterion"] is False
            and result["iterations"] == 16
            and result["prequeue_cycles"] == 10000000,
            "outcome filtered",
        ),
    )
    good(
        "graph_delay_and_query_order_" + str(outcome),
        lambda log=log: w.require(
            log.index("sleep10000000")
            < log.index("record0", log.index("sleep10000000"))
            and log.count("replay") == 16
            and log[-3:] == ["record1", "query0", "sync1"],
            "wrong queued timing order",
        ),
    )
    good(
        "graph_event_excludes_queue_wall_diagnostic_" + str(outcome),
        lambda result=result: w.require(
            result["gpu_event_per_call_ms"] == 0.096 / 16
            and result["wall_includes_prequeue"]
            and result["wall_diagnostic_only"],
            "wrong graph normalization or wall interpretation",
        ),
    )
result, log, progress = queued_case(query_error=True)
good(
    "graph_query_error_partial_preserved",
    lambda: w.require(
        result is None
        and progress["start_query_error"]
        and progress["query_error_observed_ns"] > 0
        and log[-1] == "query0",
        "query failure lost",
    ),
)

report = {
    "status": "success",
    "count": len(checks),
    "worker_sha256": w.sha(worker),
    "controls": checks,
    "scope": "CPU-only actual schedule, sampler, calibration, profile-denial, owner and inherited-cache branches. Actual arm/persistence/native/numerical behavior requires disposable CUDA admission.",
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": "success", "count": len(checks), "output": str(a.output)}))
