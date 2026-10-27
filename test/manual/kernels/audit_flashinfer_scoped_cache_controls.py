"""Focused CPU controls for the manual scoped-capture orchestration only.

These stand-ins check forwarding and cleanup; actual model/cache qualification
uses the unchanged public API on the CUDA host.
"""

import argparse
import contextlib
import copy
import dataclasses
import importlib
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import audit_flashinfer_scoped_cache as worker


@dataclasses.dataclass
class Report:
    max_records: int = 8192
    hits: int = 0
    misses: int = 0
    skipped: int = 0
    dropped_records: int = 0
    records: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Shape:
    batch_size: int = 1


class Runner:
    model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False)
    )


class Controls(unittest.TestCase):
    def setUp(self):
        self.root, self.mode, self.report = None, None, None
        self.capturing = False
        self.calls, self.hooks = [], 0
        self.files = {}
        self.legacy = {"winner": "legacy-explicit"}
        self.failure_stage = None
        self.negative_hit = False
        self.observer = worker.Observation.__new__(worker.Observation)
        o = self.observer
        o.args = SimpleNamespace(
            phase="cold",
            cache_dir=Path("/legacy"),
            managed_root=Path("/managed"),
            negative_root=Path("/negative"),
        )
        o.result = {"capture_records": []}
        o.tuner = object()
        o.stage, o.observing, o.capture_count = "startup", False, 0
        o.profiles, o.protocol_calls = [], []
        o.replays = 0
        o.policy, o.coverage = "graph-policy", Report
        o.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: None,
                is_current_stream_capturing=lambda: self.capturing,
            )
        )

        @contextlib.contextmanager
        def scope(**kwargs):
            saved = self.root, self.mode, self.report
            self.root = kwargs.get("cache_root")
            self.mode, self.report = kwargs["mode"], kwargs.get("replay_report")
            self.assertEqual(kwargs["cache_scope"], "context")
            if self.root is not None:
                self.assertEqual(kwargs["measurement_policy"], "graph-policy")
            try:
                yield self.report
            finally:
                self.root, self.mode, self.report = saved

        o.autotune_v2 = scope

    def state(self, _):
        return dict(
            is_tuning=self.mode == "tune",
            active_tuning_contexts=int(self.mode == "tune"),
            managed_store=self.root is not None,
            ambient_store=False,
            winner_identity=(str(self.root), "graph-env") if self.root else None,
        )

    def hook(self):
        self.hooks += 1

    def forward(self):
        o = self.observer
        self.calls.append((o.stage, self.capturing, self.root))
        if self.root == o.args.negative_root and not self.negative_hit:
            self.report.misses += 1
            self.report.records.append(dict(status="miss"))
            raise RuntimeError("autotune strict replay miss for control")
        if self.mode == "tune":
            if not self.files.get(self.root):
                o.profiles.append(dict(stage=o.stage))
                self.files[self.root] = {"entry.json": {"tactic": "graph-explicit"}}
            return
        self.report.hits += 1
        self.report.records.append(
            dict(
                status="hit",
                op="control",
                input_shapes=[[1, 8]],
                runner="Runner",
                selected_tactic="graph-explicit" if self.root else "legacy-explicit",
                cache_identity=self.state(None)["winner_identity"],
            )
        )
        if o.stage == self.failure_stage:
            raise LookupError("intentional forward failure")

    def original_capture(
        self, backend, key, forward, *, capture_inputs, post_warmup_hook
    ):
        self.assertEqual(capture_inputs, "preserved")
        for _ in range(2):
            forward()
            post_warmup_hook()
        self.capturing = True
        try:
            forward()
        finally:
            self.capturing = False

    def invoke(self):
        module = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
        with (
            patch.dict(
                sys.modules, {module: SimpleNamespace(DecodeCudaGraphRunner=Runner)}
            ),
            patch.object(worker, "scoped_state", side_effect=self.state),
            patch.object(
                worker,
                "legacy_snapshot",
                side_effect=lambda *_: copy.deepcopy(self.legacy),
            ),
            patch.object(
                worker,
                "json_files",
                side_effect=lambda root: copy.deepcopy(self.files.get(root, {})),
            ),
        ):
            self.observer.capture(
                self.original_capture,
                SimpleNamespace(_cuda_graph_runner=Runner()),
                Shape(),
                self.forward,
                capture_inputs="preserved",
                post_warmup_hook=self.hook,
            )

    def test_exact_preparation_original_capture_and_cleanup(self):
        self.invoke()
        self.assertEqual(
            [row[0] for row in self.calls],
            [
                "legacy_before",
                "strict_negative",
                "scoped_tune",
                "scoped_warmup",
                "scoped_warmup",
                "scoped_capture",
                "legacy_after",
            ],
        )
        self.assertEqual(self.hooks, 6)
        self.assertEqual(len(self.observer.profiles), 1)
        self.assertIsNone(self.root)
        self.assertFalse(self.observer.observing)
        self.assertTrue(self.observer.result["scoped_lifecycle"]["legacy_restored"])

    def test_warm_preparation_has_no_profile_or_publication(self):
        self.observer.args.phase = "warm"
        self.files[Path("/managed")] = {"entry.json": {"tactic": "graph-explicit"}}
        before = copy.deepcopy(self.files)
        self.invoke()
        self.assertEqual(self.files, before)
        self.assertEqual(self.observer.profiles, [])

    def test_strict_negative_must_reject_legacy_hit(self):
        self.negative_hit = True
        with self.assertRaisesRegex(RuntimeError, "unexpectedly reused legacy"):
            self.invoke()
        self.assertIsNone(self.root)
        self.assertEqual(self.hooks, 2)

    def test_forward_exception_restores_scope_and_observer(self):
        self.failure_stage = "scoped_warmup"
        with self.assertRaisesRegex(LookupError, "intentional"):
            self.invoke()
        self.assertIsNone(self.root)
        self.assertFalse(self.observer.observing)
        self.assertEqual(self.observer.stage, "startup")

    def test_legacy_mutation_is_not_hidden_by_scope_cleanup(self):
        original = self.forward

        def mutate():
            result = original()
            if self.mode == "tune":
                self.legacy["winner"] = "changed"
            return result

        self.forward = mutate
        with self.assertRaisesRegex(RuntimeError, "changed legacy winners"):
            self.invoke()
        self.assertIsNone(self.root)

    def test_report_delta_retains_cached_default_and_rejects_miss(self):
        report = Report()
        before = dataclasses.asdict(report)
        report.hits = 1
        report.records.append(dict(status="hit", selected_tactic="-1"))
        worker.require_hits(worker.coverage_delta(report, before))
        report.misses = 1
        with self.assertRaises(RuntimeError):
            worker.require_hits(worker.coverage_delta(report, before))

    def numerical(self, wrong_graph_lookup=False, wrong_eager_tactic=False):
        route = "eager"
        o = self.observer
        expected = dict(
            status="hit",
            op="control",
            input_shapes=[[1, 8]],
            runner="Runner",
            selected_tactic="legacy-explicit",
            cache_identity=None,
        )
        o.result["scoped_lifecycle"] = {"legacy_after": {"records": [expected]}}

        def decode():
            if route == "graph":
                o.replays += 1
                if wrong_graph_lookup:
                    self.forward()
            else:
                self.forward()
                if wrong_eager_tactic:
                    self.report.records[-1]["selected_tactic"] = "-1"
            return 17

        bench = SimpleNamespace(decode=decode)

        def original(bench, tokenizer, observer):
            nonlocal route
            self.assertEqual(bench.decode(), 17)
            route = "graph"
            self.assertEqual(bench.decode(), 17)
            return "preserved numerical result"

        with (
            patch.object(worker, "scoped_state", side_effect=self.state),
            patch.object(
                worker,
                "legacy_snapshot",
                side_effect=lambda *_: copy.deepcopy(self.legacy),
            ),
        ):
            return worker.numerical_checks(original, bench, None, o)

    def test_actual_numerical_calls_keep_eager_hit_and_graph_zero_lookups(self):
        self.assertEqual(self.numerical(), "preserved numerical result")
        self.assertEqual(
            [row["arm"] for row in self.observer.result["numerical_reports"]],
            ["eager", "graph"],
        )
        self.assertIsNone(self.root)

    def test_numerical_graph_lookup_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unexpectedly selected Python"):
            self.numerical(wrong_graph_lookup=True)

    def test_numerical_eager_different_winner_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "did not reuse legacy"):
            self.numerical(wrong_eager_tactic=True)


def real_tuner_control(source):
    """Real pinned CPU lookup/persistence; GPU-only imports/metadata are isolated."""
    import torch

    def package(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    def unavailable(*args, **kwargs):
        raise AssertionError("CPU lookup control reached a GPU-only boundary")

    package("flashinfer", source / "flashinfer")
    package("flashinfer.jit", source / "flashinfer/jit")
    core = types.ModuleType("flashinfer.jit.core")
    core.logger = logging.getLogger("scoped-cache-cpu-control")
    sys.modules[core.__name__] = core
    utils = types.ModuleType("flashinfer.utils")
    utils.next_positive_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    utils.is_confidential_compute = lambda: False
    utils.get_globaltimer_kernel = unavailable
    sys.modules[utils.__name__] = utils
    delay = types.ModuleType("flashinfer.tllm_utils")
    delay.delay_kernel = delay.get_trtllm_utils_module = unavailable
    sys.modules[delay.__name__] = delay
    api = importlib.import_module("flashinfer.autotuner")
    cache = importlib.import_module("flashinfer.autotune_cache")

    class CpuRunner(api.TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [-1]

        def forward(self, *args, **kwargs):
            raise AssertionError("Lookup control executed a kernel")

    checks = []

    def check(value, label):
        if not value:
            raise AssertionError(label)
        checks.append(label)

    runner, config = CpuRunner(), api.TuningConfig()
    inputs = [torch.ones(2)]
    policy = cache.MeasurementPolicy(execution_mode="cuda_graph")
    with tempfile.TemporaryDirectory() as directory:
        root, legacy = Path(directory) / "managed", Path(directory) / "legacy"
        tuner = api.AutoTuner()
        key = tuner._get_cache_key("scoped_cpu_control", runner, ((2,),), config, ())
        tuner._cache_profile_result(key, None, config, -1)
        before_state = worker.scoped_state(tuner)
        before_cache = worker.legacy_snapshot(tuner, legacy)
        with (
            patch.object(api.AutoTuner, "get", return_value=tuner),
            patch.object(
                api, "_collect_metadata", return_value={"cpu_control": "scoped"}
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(tuner, "_profile_single_kernel", side_effect=unavailable),
        ):
            report = cache.ReplayCoverage()
            try:
                with cache.autotune_v2(
                    mode="replay",
                    cache_scope="context",
                    cache_root=root,
                    measurement_policy=policy,
                    strict_replay=True,
                    replay_report=report,
                ):
                    tuner.choose_one("scoped_cpu_control", [runner], config, inputs)
            except RuntimeError as error:
                check("strict replay miss" in str(error), "real strict miss error")
            else:
                raise AssertionError(
                    "Legacy cached default leaked into managed identity"
                )
            check(
                report.misses == 1 and report.hits == 0,
                "cached legacy -1 is a managed miss",
            )
            check(not worker.json_files(root), "strict miss publishes no files")
            check(
                worker.scoped_state(tuner) == before_state,
                "strict exception restores controls",
            )
            with cache.autotune_v2(
                mode="tune",
                cache_scope="context",
                cache_root=root,
                measurement_policy=policy,
            ):
                measured = tuner._apply_measure_policy(config, policy)
                tuner._cache_profile_result(key, None, measured, -1)
                identity = tuner._winner_cache_identity()
                check(
                    worker.scoped_state(tuner)["isolate_legacy"],
                    "identified scope isolates legacy",
                )
            check(
                worker.legacy_snapshot(tuner, legacy) == before_cache,
                "managed seed preserves real legacy maps",
            )
            report = cache.ReplayCoverage()
            with cache.autotune_v2(
                mode="replay",
                cache_scope="context",
                cache_root=root,
                measurement_policy=policy,
                strict_replay=True,
                replay_report=report,
            ):
                selected, tactic = tuner.choose_one(
                    "scoped_cpu_control", [runner], config, inputs
                )
            check(
                selected is runner and tactic == -1 and report.hits == 1,
                "real cached managed -1 hit",
            )
            check(
                worker.scoped_state(tuner) == before_state,
                "managed hit scope restores controls",
            )
        fresh = api.AutoTuner()
        with (
            patch.object(api.AutoTuner, "get", return_value=fresh),
            patch.object(
                api, "_collect_metadata", return_value={"cpu_control": "scoped"}
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            report = cache.ReplayCoverage()
            with cache.autotune_v2(
                mode="replay",
                cache_scope="context",
                cache_root=root,
                measurement_policy=policy,
                strict_replay=True,
                replay_report=report,
            ):
                fresh.choose_one("scoped_cpu_control", [runner], config, inputs)
                check(
                    fresh._winner_cache_identity() == identity,
                    "fresh instance retains managed identity",
                )
            check(
                report.hits == 1 and report.records[0]["source"] == "managed",
                "fresh instance uses real persisted winner",
            )
    return dict(
        checks=checks,
        source_sha256={
            name: worker.base.digest((source / name).read_bytes())
            for name in (
                "flashinfer/autotune_cache.py",
                "flashinfer/autotuner/autotuner.py",
            )
        },
        limits="CPU lookup and persistence only; explicit GPU import boundaries and metadata shim, no kernels",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors)
    )
    if args.source and result.wasSuccessful():
        report["real_tuner_control"] = real_tuner_control(args.source)
    if args.output:
        worker.base.write_json(args.output, report)
    print(json.dumps(report))
    sys.exit(not result.wasSuccessful())
