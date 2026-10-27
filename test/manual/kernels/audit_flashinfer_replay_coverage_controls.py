"""CPU controls for the manual observer's forwarding and restoration boundaries."""

import contextlib
import dataclasses
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "coverage_worker", Path(__file__).with_name("audit_flashinfer_replay_coverage.py")
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@dataclasses.dataclass
class Shape:
    size: int = 1


@dataclasses.dataclass
class Report:
    max_records: int = 8192
    hits: int = 0
    misses: int = 0
    skipped: int = 0
    dropped_records: int = 0
    records: list = dataclasses.field(default_factory=list)


class Runner:
    model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False)
    )


class Tuner:
    _in_v2_context = False

    def _profile_single_kernel(self, *args, **kwargs):
        return 1

    def choose_with_protocol(self, *args, **kwargs):
        return 7

    def load_configs(self, *args, **kwargs):
        pass

    def save_configs(self, *args, **kwargs):
        pass


class ObserverControls(unittest.TestCase):
    def setUp(self):
        self.state = {
            "is_tuning": False,
            "active_tuning_contexts": 0,
            "managed_store": False,
            "ambient_store": False,
            "winner_identity": None,
            "skip_ops": ["inherited"],
        }
        self.capturing = False
        self.calls = 0
        self.post_warmups = 0
        test = self

        class Backend:
            _cuda_graph_runner = Runner()

            def capture_one(
                self, shape, forward_fn, *, capture_inputs, post_warmup_hook
            ):
                test.assertEqual(capture_inputs, "preserved")
                for _ in range(2):
                    forward_fn()
                    post_warmup_hook()
                test.capturing = True
                try:
                    forward_fn()
                finally:
                    test.capturing = False

            def replay(self):
                return 9

        self.backend_class = Backend
        self.backend = Backend()
        self.observer = worker.Observation.__new__(worker.Observation)
        self.observer.args = SimpleNamespace(cache_dir=Path("/tmp"))
        self.observer.result = {"capture_records": []}
        self.observer.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)
        )
        self.observer.tuner_class, self.observer.tuner = Tuner, Tuner()
        self.observer.coverage = Report
        self.observer.stage, self.observer.observing = "startup", False
        (
            self.observer.profiles,
            self.observer.protocol_calls,
            self.observer.load_save,
        ) = [], [], []
        self.observer.replays = 0

        @contextlib.contextmanager
        def scope(**kwargs):
            self.assertEqual(
                set(kwargs),
                {"mode", "persistent_cache", "cache_scope", "replay_report"},
            )
            self.assertEqual(
                (kwargs["mode"], kwargs["persistent_cache"], kwargs["cache_scope"]),
                ("replay", False, "context"),
            )
            self.observer.tuner._in_v2_context = True
            self.report = kwargs["replay_report"]
            try:
                yield
            finally:
                self.observer.tuner._in_v2_context = False

        self.observer.autotune_v2 = scope
        self.modules = {
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner": SimpleNamespace(
                DecodeCudaGraphRunner=Runner
            ),
            "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend": SimpleNamespace(
                FullCudaGraphBackend=Backend
            ),
        }

    @contextlib.contextmanager
    def installed(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch.object(worker, "tuner_state", side_effect=lambda _: dict(self.state)),
            patch.object(worker, "cache_inventory", return_value={}),
            self.observer.install(),
        ):
            yield

    def forward(self):
        self.calls += 1
        self.report.hits += 1
        return "original result"

    def post_warmup(self):
        self.post_warmups += 1

    def capture(self, forward=None):
        self.backend.capture_one(
            Shape(),
            forward or self.forward,
            capture_inputs="preserved",
            post_warmup_hook=self.post_warmup,
        )

    def test_exact_existing_forward_and_hook_counts(self):
        original = self.backend_class.capture_one
        with self.installed():
            self.capture()
        self.assertEqual((self.calls, self.post_warmups), (3, 2))
        self.assertEqual(
            [row["capturing"] for row in self.observer.result["capture_records"]],
            [False, False, True],
        )
        self.assertIs(self.backend_class.capture_one, original)
        self.assertFalse(self.observer.tuner._in_v2_context)

    def test_forward_exception_restores_observer(self):
        def fail():
            raise LookupError("forward failed")

        with self.installed(), self.assertRaises(LookupError):
            self.capture(fail)
        self.assertFalse(self.observer.observing)
        self.assertFalse(self.observer.tuner._in_v2_context)
        self.assertEqual(self.observer.stage, "startup")
        self.assertTrue(self.observer.result["capture_records"][0]["context_restored"])

    def test_active_tuning_rejected_before_forward(self):
        self.state["is_tuning"] = True
        with self.installed(), self.assertRaises(RuntimeError):
            self.capture()
        self.assertEqual(self.calls, 0)

    def test_ambient_store_rejected(self):
        self.state["ambient_store"] = True
        with self.installed(), self.assertRaises(RuntimeError):
            self.capture()

    def test_changed_lookup_state_rejected(self):
        def change():
            self.state["skip_ops"] = ["unexpected"]

        with self.installed(), self.assertRaises(RuntimeError):
            self.capture(change)
        self.assertFalse(self.observer.tuner._in_v2_context)

    def test_protocol_call_preserved_outside_observer(self):
        with self.installed():
            self.assertEqual(self.observer.tuner.choose_with_protocol(), 7)
        self.assertEqual(
            self.observer.protocol_calls, [{"stage": "startup", "observed": False}]
        )

    def test_protocol_call_rejected_inside_observer(self):
        with self.installed(), self.assertRaises(RuntimeError):
            self.capture(self.observer.tuner.choose_with_protocol)
        self.assertTrue(self.observer.protocol_calls[0]["observed"])

    def test_replay_only_counts_existing_invocation(self):
        with self.installed():
            self.assertEqual(self.backend.replay(), 9)
        self.assertEqual(self.observer.replays, 1)
        self.assertEqual(self.observer.result["capture_records"], [])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObserverControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"passed": result.wasSuccessful(), "controls": result.testsRun}))
    sys.exit(not result.wasSuccessful())
