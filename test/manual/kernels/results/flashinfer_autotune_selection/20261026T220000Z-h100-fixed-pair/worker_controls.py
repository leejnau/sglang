"""Focused CPU controls for the fixed-pair worker; no CUDA execution."""

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--repo", type=Path, default=Path("/private/tmp/sglang-fixed-pair-study")
)
parser.add_argument(
    "--output", type=Path, default=Path(__file__).with_name("cpu-controls.json")
)
arguments = parser.parse_args()
ROOT = arguments.repo.resolve()
WORKER = ROOT / "test/manual/kernels/bench_flashinfer_fixed_pair.py"
BATCH = (
    ROOT
    / "test/manual/kernels/results/flashinfer_autotune_selection/20261026T220000Z-h100-fixed-pair"
)
sys.path.insert(0, str(WORKER.parent))
spec = importlib.util.spec_from_file_location("fixed_pair_worker", WORKER)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
controls = []


def good(name, callback):
    callback()
    controls.append({"control": name, "passed": True})


def bad(name, callback):
    try:
        callback()
    except (RuntimeError, ValueError, KeyError, TypeError):
        controls.append({"control": name, "passed": True, "expected_rejection": True})
    else:
        raise AssertionError(name + " did not reject")


def args(trial=0):
    schedule = json.loads((BATCH / "schedule.json").read_text())
    row = schedule["entries"][trial]
    return NS(
        mode="timing",
        trial=trial,
        block=trial,
        cell="auto_synthetic",
        capture_order=row["capture_order"],
        profile_order=row["profiler_objective_order"][0] + "-first",
        plan=BATCH / "plan.json",
        schedule=BATCH / "schedule.json",
        historical_contract=BATCH / "historical-contract.json",
    )


for trial in range(8):
    good(
        "frozen_schedule_trial_" + str(trial),
        lambda trial=trial: w.Experiment(args(trial)),
    )
for which in ("capture_order", "profile_order"):
    value = args()
    setattr(
        value,
        which,
        {"capture_order": "default-first", "profile_order": "graph-first"}[which],
    )
    bad("wrong_" + which, lambda value=value: w.Experiment(value))
orders = json.loads((BATCH / "schedule.json").read_text())
for objective in ("auto", "graph"):
    broken = copy.deepcopy(orders)
    broken["profiler_pair_schedules"]["0"][objective] = [["winner", "default"]] * 12
    bad(
        "unbalanced_" + objective,
        lambda broken=broken, objective=objective: w.paired_orders(
            broken, 0, objective
        ),
    )

# Retention executes only after successful final baseline call, index80.
e = w.Experiment(args())
e.metadata = lambda t: {"id": t}
e.result["startup"]["profiles"] = [{}] * 79
inputs = tuple(range(6))


def original_profile(self, runner, inputs, tactic, config, *more, **kwargs):
    self.result["startup"]["profiles"].append({})
    return 0.01


with patch.object(w.OriginalExperiment, "profile", original_profile):
    e.profile("runner", inputs, -1, "config")
    good(
        "no_retention_after_index79",
        lambda: w.require(e.retained is None, "premature retain"),
    )
    e.profile("runner", inputs, -1, "config", input_tensor_batches=[list(inputs)])
    good(
        "retention_after_index80",
        lambda: w.require(
            e.retained["inputs"] == inputs
            and e.result["retained_profile"]["retained_after_index"] == 80,
            "wrong retain",
        ),
    )


# The actual new cache partition guard allows empty scope bookkeeping only.
def state_experiment():
    e = w.Experiment(args())
    e.tuner = NS(
        profiling_cache={"key": (-1, None)},
        _winner_partitions={},
        _protocol_record_partitions={},
        _dirty=False,
    )

    def digest():
        fields = {
            k: hashlib.sha256(repr(v).encode()).hexdigest()
            for k, v in vars(e.tuner).items()
        }
        e.cache_state_field_sha256 = fields
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    e.cache_state_sha = digest
    return e


for partition in ("_winner_partitions", "_protocol_record_partitions"):
    e = state_experiment()

    def empty(e=e, partition=partition):
        with e.restore_policy_partitions("graph"):
            getattr(e.tuner, partition)["graph"] = {}
        w.require(
            getattr(e.tuner, partition) == {}
            and e.result["cache_partition_restorations"][0]["restored"],
            "not restored",
        )

    good("empty_partition_restored_" + partition, empty)
    e = state_experiment()

    def nonempty(e=e, partition=partition):
        with e.restore_policy_partitions("graph"):
            getattr(e.tuner, partition)["graph"] = {"winner": 1}

    bad("nonempty_partition_rejected_" + partition, nonempty)
e = state_experiment()


def mutation():
    with e.restore_policy_partitions("graph"):
        e.tuner.profiling_cache["key"] = (1, None)


bad("real_winner_mutation_rejected", mutation)
e = state_experiment()


def exception_restore():
    try:
        with e.restore_policy_partitions("graph"):
            e.tuner._winner_partitions["graph"] = {}
            raise LookupError("deliberate")
    except LookupError:
        pass
    w.require(e.tuner._winner_partitions == {}, "exception leaked partition")


good("empty_partition_restored_on_exception", exception_restore)

# Exercise the inherited real miss scope under an exception: fixed insertion
# must not replace the natural winner or mutate incidental file bytes.
e = w.Experiment(args())
with tempfile.TemporaryDirectory() as temporary:
    e.args.cache = Path(temporary) / "cache.json"
    e.args.cache.write_text("{}")
    names = (
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
    e.tuner = NS(**{name: {} for name in names}, _dirty=False, _dirty_seq=0)
    e.tuner.profiling_cache["natural"] = (-1, None)
    e.tuner._winner_cache = lambda: e.tuner.profiling_cache
    before = {name: copy.deepcopy(getattr(e.tuner, name)) for name in names}

    def miss_exception():
        try:
            with e.miss_scope():
                e.tuner.profiling_cache["fixed"] = (w.FIXED_DESCRIPTOR, None)
                raise LookupError("deliberate fixed capture failure")
        except LookupError:
            pass
        w.require(
            all(getattr(e.tuner, k) == v for k, v in before.items()),
            "miss restoration failed",
        )

    good("fixed_cache_insertion_exception_restores_natural", miss_exception)


# Exercise actual diagnostic guard and actual delegation without CUDA mocks.
@dataclasses.dataclass
class Config:
    use_cuda_graph: bool = False
    use_cold_l2_cache: bool = False


def diagnostic():
    e = w.Experiment(args())
    tensors = [object() for _ in range(6)]
    runner = object()
    e.retained = {"inputs": tuple(tensors), "runner": runner}
    e.metadata = lambda t: {"object": id(t)}
    e.policy = None
    e.settings = lambda config: {
        **e.historical["original_settings"],
        "config": dataclasses.asdict(config),
    }
    e.late_objective = "auto"
    e.stage = "late_auto_00_default"
    e.allow_diagnostic = True
    e.allow_profiling = False
    e.model_runner = NS(forward_stream="forward")
    e.torch = NS(cuda=NS(current_stream=lambda: "forward"))
    e.tuner = NS(_in_v2_context=True, is_tuning_mode=True)
    e.original_profile = lambda *a, **kw: 0.01
    return e, tensors, runner


e, tensors, runner = diagnostic()
good(
    "actual_profiler_delegation_positive",
    lambda: w.require(
        e.diagnostic_profile(
            runner, tensors, -1, Config(), input_tensor_batches=[tensors]
        )
        == 0.01,
        "no delegation",
    ),
)
for name, change in (
    ("disabled", lambda e: setattr(e, "allow_diagnostic", False)),
    ("startup_overlap", lambda e: setattr(e, "allow_profiling", True)),
    ("no_v2", lambda e: setattr(e.tuner, "_in_v2_context", False)),
    ("not_tuning", lambda e: setattr(e.tuner, "is_tuning_mode", False)),
    ("wrong_stream", lambda e: setattr(e.model_runner, "forward_stream", "other")),
):
    e, tensors, runner = diagnostic()
    change(e)
    bad(
        "guard_" + name,
        lambda e=e, tensors=tensors, runner=runner: e.diagnostic_profile(
            runner, tensors, -1, Config(), input_tensor_batches=[tensors]
        ),
    )
    good(
        "rejected_attempt_counted_" + name,
        lambda e=e: w.require(
            e.diagnostic_attempts == 1
            and e.result["diagnostic_profile_attempts"][0]["error"],
            "swallowed denial",
        ),
    )
for name, tactic, config in (
    ("wrong_tactic", tuple([0] * 8), Config()),
    ("wrong_objective", -1, Config(True)),
    ("cold_l2", -1, Config(False, True)),
):
    e, tensors, runner = diagnostic()
    bad(
        name,
        lambda e=e, tensors=tensors, runner=runner, tactic=tactic, config=config: (
            e.diagnostic_profile(
                runner, tensors, tactic, config, input_tensor_batches=[tensors]
            )
        ),
    )
e, tensors, runner = diagnostic()
wrong = list(tensors)
wrong[0] = object()
bad(
    "replaced_input_object",
    lambda: e.diagnostic_profile(
        runner, wrong, -1, Config(), input_tensor_batches=[wrong]
    ),
)
e, tensors, runner = diagnostic()
bad(
    "extra_input_batch",
    lambda: e.diagnostic_profile(
        runner, tensors, -1, Config(), input_tensor_batches=[tensors, tensors]
    ),
)
e, tensors, runner = diagnostic()
e.original_profile = lambda *a, **k: float("nan")
bad(
    "nonfinite_real_profiler_result",
    lambda: e.diagnostic_profile(
        runner, tensors, -1, Config(), input_tensor_batches=[tensors]
    ),
)
e, tensors, runner = diagnostic()
e.original_profile = lambda *a, **k: 0
bad(
    "zero_real_profiler_result",
    lambda: e.diagnostic_profile(
        runner, tensors, -1, Config(), input_tensor_batches=[tensors]
    ),
)

# Real CPU tensors exercise no-write/stale-output and ownership guard behavior.
import torch


def numerical():
    e = w.Experiment(args())
    inputs = [torch.ones(1, 2, 2) for _ in range(6)]
    e.retained = {"inputs": tuple(inputs)}
    e.torch = NS(
        cuda=NS(synchronize=lambda: None),
        isfinite=torch.isfinite,
        zeros_like=torch.zeros_like,
        count_nonzero=torch.count_nonzero,
        allclose=torch.allclose,
    )
    e.profile_reference = torch.full((1, 2, 2), 2.0)
    e.result["retained_profile"] = {
        "tensor_object_ids": [id(t) for t in inputs],
        "metadata": [e.metadata(t) for t in inputs],
    }
    return e, inputs


e, tensors = numerical()
tensors[4].fill_(2)
good("real_tensor_numerical_positive", lambda: e.check_profile_output("positive"))
e, tensors = numerical()
tensors[4].fill_(float("nan"))
bad("real_tensor_no_write_nan_rejected", lambda: e.check_profile_output("no_write"))
e, tensors = numerical()
tensors[4].fill_(3)
bad("real_tensor_wrong_output_rejected", lambda: e.check_profile_output("wrong"))
e, tensors = numerical()
tensors[4].zero_()
good(
    "real_tensor_changed_input_exact_zero",
    lambda: e.check_profile_output("zero", zero=True),
)
e, tensors = numerical()
tensors[4].fill_(0.00001)
bad(
    "real_tensor_zero_tolerance_not_relaxed",
    lambda: e.check_profile_output("bad_zero", zero=True),
)
e, tensors = numerical()
e.retained["inputs"] = tuple(list(tensors[:4]) + [tensors[4].clone(), tensors[5]])
bad(
    "real_tensor_replaced_output_rejected",
    lambda: e.check_profile_output("wrong_owner"),
)

report = {
    "status": "success",
    "worker_sha256": w.sha(WORKER),
    "controls": controls,
    "count": len(controls),
    "scope": "CPU controls exercise new worker guards and inherited miss restoration; no CUDA or performance qualification.",
}
output = arguments.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": "success", "count": len(controls), "report": str(output)}))
