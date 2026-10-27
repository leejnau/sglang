"""CPU qualification of the research scoped-cache control handoff.

Run in an isolated process. Real SGLang configuration/helpers and FlashInfer
selection/persistence are exercised with CPU tensors and controlled profiling
times. This does not execute kernels, CUDA graphs, or a production adapter.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

from audit_flashinfer_scoped_handoff import ScopedGraphRegion
from audit_flashinfer_scoped_handoff_bootstrap import bootstrap


def inventory(root):
    return {
        str(path.relative_to(root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contents": json.loads(path.read_text()),
        }
        for path in sorted(root.rglob("*.json"))
    }


@contextlib.contextmanager
def override_config(runtime, **fields):
    context = runtime.get_context()
    previous = {name: context.config_leaf(name) for name in fields}
    context.override("scoped-handoff-control", **fields)
    try:
        yield
    finally:
        context.override("scoped-handoff-control-restore", **previous)


class Fixture:
    def __init__(self, modules, root, result):
        self.b, self.root, self.result = modules, root, result
        self.tuner = modules.api.AutoTuner()
        self.model_runner = modules.model_runner()
        self.config = modules.api.TuningConfig()
        self.inputs = [modules.torch.ones(2)]
        self.winner = 2
        self.operation = None
        self.profiles = []
        self.checks = result["checks"] = []
        self.events = []

        class CpuRunner(modules.api.TunableRunner):
            def get_valid_tactics(self, inputs, profile):
                return [1, 2]

            def forward(self, inputs, tactic=-1):
                raise AssertionError("CPU control attempted kernel execution")

        self.runner = CpuRunner()

    def check(self, condition, label):
        if not condition:
            raise AssertionError(label)
        self.checks.append(label)

    def profile(self, runner, inputs, tactic, config, **kwargs):
        elapsed = 1.0 if tactic == self.winner else 10.0
        self.profiles.append(
            dict(
                op=self.operation,
                tactic=tactic,
                controlled_time=elapsed,
                identity=self.tuner._winner_cache_identity(),
            )
        )
        return elapsed

    @contextlib.contextmanager
    def active(self):
        try:
            with (
                patch.object(self.b.api.AutoTuner, "get", return_value=self.tuner),
                patch.object(
                    self.tuner, "_profile_single_kernel", side_effect=self.profile
                ),
                self.b.environ.SGLANG_CACHE_DIR.override(str(self.root / "legacy")),
                self.b.environ.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
            ):
                yield self
        finally:
            self.result["profiles"] = self.profiles
            self.result["persisted"] = inventory(self.root)

    def choose(self, op):
        previous, self.operation = self.operation, op
        try:
            runner, tactic = self.tuner.choose_one(
                op, [self.runner], self.config, self.inputs
            )
            self.check(runner is self.runner, f"{op}: actual runner selected")
            return tactic
        finally:
            self.operation = previous

    def seed_legacy(self, op, tactic):
        self.winner = tactic
        with self.b.cache.autotune_v2(mode="tune", persistent_cache=False):
            self.check(self.choose(op) == tactic, f"{op}: legacy seed winner")

    def seed_managed(self, region, op, tactic):
        self.winner = tactic
        with region.context("tune"):
            self.check(self.choose(op) == tactic, f"{op}: managed seed winner")

    def region(self, name="graph"):
        region = ScopedGraphRegion.create(self.model_runner, self.root / name)
        self.check(region is not None, "eligible graph region exists")
        return region

    def legacy_choice(self, op):
        with self.b.cache.autotune_v2(
            mode="replay", persistent_cache=False, cache_scope="context"
        ):
            return self.choose(op)

    def state(self):
        return (
            self.tuner._effective_skip_ops,
            self.tuner._winner_cache_identity(),
            self.tuner._active_managed_store,
            self.tuner._effective_measure_policy,
            self.tuner.is_tuning_mode,
            self.tuner._v2_local.active,
            self.tuner._v2_local.cache_scope,
            self.tuner._v2_local.strict_replay,
            self.tuner._v2_local.replay_report,
        )


def gate_and_sequence(f):
    for field in (
        "disable_flashinfer_autotune",
        "enable_deterministic_inference",
        "disable_cuda_graph",
    ):
        with override_config(f.b.runtime, **{field: True}):
            region = ScopedGraphRegion.create(f.model_runner, f.root / field)
            f.check(region is None, f"{field}: graph preparation is ineligible")
            f.check(not f.profiles, f"{field}: no profiling")
            f.check(not inventory(f.root), f"{field}: no cache publication")
            f.check(not f.root.exists(), f"{field}: no cache directory created")

    f.model_runner._kernel_warmed_up = True
    resource = object()
    f.model_runner.handoff_resource_sentinel = resource
    region = f.region()
    f.winner = 2

    def forbidden():
        raise AssertionError("Active capture reached preparation or callbacks")

    try:
        region.run(forbidden, forbidden, forbidden, is_capturing=lambda: True)
    except RuntimeError as error:
        f.check(
            "precede CUDA capture" in str(error), "active capture rejects preparation"
        )
    else:
        raise AssertionError("Preparation began inside active capture")
    f.check(
        not f.profiles and not inventory(f.root),
        "active capture has no profiling or publication",
    )
    f.check(not region.complete, "active capture rejection leaves region incomplete")

    def prepare():
        f.events.append("prepare")
        f.check(f.choose("sequence") == 2, "preparation chooses explicit winner")

    def capture():
        for stage in ("warmup-1", "warmup-2", "capture"):
            f.events.append(stage)
            f.check(f.choose("sequence") == 2, f"{stage}: cached explicit winner")
        return "retained-capture-result"

    returned = region.run(
        prepare, capture, lambda: f.events.append("reset"), is_capturing=lambda: False
    )
    f.check(
        f.events == ["prepare", "reset", "warmup-1", "warmup-2", "capture"],
        "preparation and reset precede original warmups/capture callback",
    )
    f.check(returned == "retained-capture-result", "capture result preserved")
    f.check(region.complete, "completion follows successful replay")
    f.check(region.report.hits == 3, "all callback selections are real hits")
    f.check(len(f.profiles) == 3, "only three preparation candidates profiled")
    f.check(
        {row["tactic"] for row in f.profiles} == {-1, 1, 2},
        "default participates alongside explicit candidates",
    )
    f.check(f.model_runner._kernel_warmed_up, "legacy warmup flag retained")
    f.check(
        f.model_runner.handoff_resource_sentinel is resource,
        "unrelated caller resource retained",
    )
    f.result["events"] = f.events
    f.result["coverage"] = dataclasses.asdict(region.report)


def skips_after_startup(f):
    # Seed without skips, then configure them after real selection/persistence.
    region = f.region()
    f.seed_managed(region, "skipped", 1)
    f.seed_managed(region, "companion", 2)
    before = inventory(region.spec.root)
    f.result["seeded_graph"] = before
    with override_config(f.b.runtime, flashinfer_autotune_skip_ops=["skipped"]):
        with f.b.sg.flashinfer_autotune_context(f.model_runner, run_lm_head=False):
            f.check(f.choose("skipped") == -1, "startup applies configured skip")
        f.check(
            "skipped" not in f.tuner._effective_skip_ops,
            "startup exit removes its skip stack entry",
        )
        region = f.region()
        count = len(f.profiles)

        def forward():
            f.check(f.choose("skipped") == -1, "handoff skip overrides cached explicit")
            f.check(f.choose("companion") == 2, "unskipped companion keeps explicit")

        region.run(forward, forward, lambda: None, is_capturing=lambda: False)
        f.check(len(f.profiles) == count, "skip plus warm companion never profiles")
        f.check(
            inventory(region.spec.root) == before, "skip handoff preserves store bytes"
        )
        f.check(
            (region.report.hits, region.report.skipped, region.report.misses)
            == (1, 1, 0),
            "skip is accounted separately from a real hit",
        )
        report = f.b.cache.ReplayCoverage()
        try:
            with region.context("replay", report=report):
                f.choose("uncovered")
        except RuntimeError as error:
            f.check("strict replay miss" in str(error), "uncovered call fails strictly")
        else:
            raise AssertionError("Uncovered unskipped operation was accepted")
        f.check(report.misses == 1, "strict miss recorded")
        f.check(len(f.profiles) == count, "strict miss never profiles")
        f.result["coverage"] = dataclasses.asdict(region.report)
        f.result["negative_coverage"] = dataclasses.asdict(report)


def reuse_off(f):
    stable = f.b.sg.flashinfer_autotune_cache_path(f.model_runner)
    f.seed_legacy("conflict", 1)
    f.seed_legacy("disk_only", 1)
    f.tuner.save_configs(str(stable))
    stable_before = stable.read_bytes()
    f.result["stable_legacy_before"] = dict(
        sha256=hashlib.sha256(stable_before).hexdigest(),
        contents=json.loads(stable_before),
    )
    f.tuner.clear_cache()
    f.check(f.tuner.load_configs(str(stable)), "stable legacy file loads normally")
    f.check(f.choose("disk_only") == 1, "loaded disk-only explicit tactic is usable")
    # Create memory first, then load the conflicting disk entry through the
    # real parser, preserving production cache lookup precedence.
    f.tuner.clear_cache()
    f.seed_legacy("conflict", 2)
    f.check(f.tuner.load_configs(str(stable)), "conflicting disk and memory coexist")
    f.check(f.choose("conflict") == 2, "memory winner precedes conflicting disk")
    managed = f.region()
    f.seed_managed(managed, "graph_only", 1)
    managed_before = inventory(managed.spec.root)
    f.result["stable_graph_before"] = managed_before
    with f.b.environ.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False):
        with f.b.sg.flashinfer_autotune_context(f.model_runner, run_lm_head=False):
            f.check(not f.tuner._file_configs, "reuse-off clears loaded-file entries")
            count = len(f.profiles)
            f.check(f.choose("conflict") == 2, "reuse-off retains eligible memory")
            f.check(len(f.profiles) == count, "memory reuse avoids profiling")
            f.winner = -1
            f.check(f.choose("new_legacy") == -1, "fresh legacy default can win")
        f.check(stable.read_bytes() == stable_before, "stable legacy bytes unchanged")
        runs = list((stable.parent / "runs").glob("*.json"))
        f.check(len(runs) == 1, "reuse-off writes one isolated legacy snapshot")
        saved = json.loads(runs[0].read_text())
        f.check(
            any("new_legacy" in key for key in saved),
            "isolated snapshot includes fresh winner",
        )
        f.check(
            not any("disk_only" in key for key in saved),
            "isolated snapshot excludes dropped file entry",
        )
        graph = f.region()
        f.check(
            graph.spec.root != managed.spec.root,
            "reuse-off chooses distinct graph identity",
        )
        chosen_root = graph.spec.root
        f.winner = 2
        before_profiles = len(f.profiles)

        def forward():
            f.check(
                f.choose("graph_only") == 2,
                "isolated graph ignores stable conflicting winner",
            )

        graph.run(forward, forward, lambda: None, is_capturing=lambda: False)
        f.check(graph.spec.root == chosen_root, "one root retained through tune/replay")
        f.check(len(f.profiles) == before_profiles + 3, "isolated graph profiles once")
        warm_bytes = inventory(chosen_root)
        f.result["isolated_graph_after_cold"] = warm_bytes
        graph.run(forward, forward, lambda: None, is_capturing=lambda: False)
        f.check(
            len(f.profiles) == before_profiles + 3,
            "warm region performs no additional profiling",
        )
        f.check(
            inventory(chosen_root) == warm_bytes, "warm region preserves stored bytes"
        )
        f.check(
            inventory(managed.spec.root) == managed_before,
            "stable graph store bytes unchanged",
        )
        snapshot = f.root / "after_graph.json"
        f.tuner.save_configs(str(snapshot))
        f.check(
            not any("graph_only" in key for key in json.loads(snapshot.read_text())),
            "legacy save excludes identified graph winners",
        )
        fresh = f.b.api.AutoTuner()
        f.check(
            fresh.load_configs(str(runs[0])), "isolated snapshot loads in fresh tuner"
        )
        loaded_report = f.b.cache.ReplayCoverage()
        with (
            patch.object(f.b.api.AutoTuner, "get", return_value=fresh),
            f.b.cache.autotune_v2(
                mode="replay",
                persistent_cache=False,
                cache_scope="context",
                replay_report=loaded_report,
            ),
        ):
            selected, tactic = fresh.choose_one(
                "new_legacy", [f.runner], f.config, f.inputs
            )
            f.check(
                selected is f.runner and tactic == -1,
                "saved default winner survives load",
            )
        f.check(
            loaded_report.hits == 1
            and loaded_report.misses == 0
            and loaded_report.records[0]["source"] == "legacy",
            "reloaded default is a persisted hit, not fallback",
        )
        f.result["coverage"] = dataclasses.asdict(graph.report)
        f.result["reloaded_coverage"] = dataclasses.asdict(loaded_report)
        f.result["stable_legacy_after"] = dict(
            sha256=hashlib.sha256(stable.read_bytes()).hexdigest(),
            contents=json.loads(stable.read_bytes()),
        )
        f.result["stable_graph_after"] = inventory(managed.spec.root)
        f.result["isolated_graph_after_warm"] = inventory(chosen_root)


def error_restoration(f):
    f.seed_legacy("error_case", 1)
    with f.b.cache.autotune_v2(
        mode="replay", cache_root=f.root / "ambient", persistent_cache=True
    ):
        pass
    with (
        override_config(f.b.runtime, flashinfer_autotune_skip_ops=["configured"]),
        f.b.api.autotune(False, skip_ops={"enclosing"}),
    ):
        before = f.state()
        region = f.region()
        f.winner = 2
        for failure in ("after_prepare", "capture"):
            reset_events = []

            def forward():
                f.check(
                    f.choose("error_case") == 2, "graph retains its explicit winner"
                )
                f.check(
                    f.choose("configured") == -1, "configured skip remains effective"
                )
                f.check(f.choose("enclosing") == -1, "enclosing skip remains effective")

            def reset():
                reset_events.append("reset")

            def prepare():
                forward()
                if failure == "after_prepare":
                    raise LookupError("intentional post-preparation failure")

            def capture():
                forward()
                raise LookupError("intentional capture callback failure")

            try:
                region.run(prepare, capture, reset, is_capturing=lambda: False)
            except LookupError as error:
                f.check(
                    "intentional" in str(error), f"{failure}: injected error propagated"
                )
            else:
                raise AssertionError("Injected callback failure was swallowed")
            f.check(not region.complete, f"{failure}: failure is not completed")
            f.check(reset_events == ["reset"], f"{failure}: preparation reset executed")
            f.check(
                f.state() == before, f"{failure}: ambient identity and skips restored"
            )
            f.check(
                f.legacy_choice("error_case") == 1, f"{failure}: legacy winner intact"
            )
            persisted = inventory(region.spec.root)
            count = len(f.profiles)
            region.run(forward, forward, lambda: None, is_capturing=lambda: False)
            f.check(
                region.complete, f"{failure}: valid prepared entries remain replayable"
            )
            f.check(len(f.profiles) == count, f"{failure}: recovery needs no profiling")
            f.check(
                inventory(region.spec.root) == persisted,
                f"{failure}: recovery preserves valid files",
            )
            f.check(
                f.state() == before,
                f"{failure}: successful retry restores ambient state",
            )
        f.result["coverage"] = dataclasses.asdict(region.report)


@contextlib.contextmanager
def mutation(name):
    """Fault only the research handoff; retain the same selection oracles."""
    if name is None:
        yield
        return
    original = ScopedGraphRegion.context

    def faulty(region, mode, *, report=None):
        previous = region.spec
        if name == "omit-skips":
            region.spec = dataclasses.replace(previous, skip_ops=frozenset())
        elif mode == "replay":
            region.spec = dataclasses.replace(
                previous, root=previous.root / "wrong-replay-root"
            )
        try:
            return original(region, mode, report=report)
        finally:
            region.spec = previous

    with patch.object(ScopedGraphRegion, "context", faulty):
        yield


def run(source, output, *, mutation_name=None):
    sglang_root = Path(__file__).resolve().parents[3]
    report = {
        "status": "running",
        "mutation": mutation_name,
        "families": {},
        "limits": "CPU control flow and persistence only; synthetic profiling times, no kernel or CUDA capture execution; research adapter has no production caller",
        "flashinfer_source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "flashinfer/autotuner/autotuner.py",
                "flashinfer/autotune_cache.py",
            )
        },
        "sglang_source_sha256": {
            name: hashlib.sha256((sglang_root / name).read_bytes()).hexdigest()
            for name in (
                "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
                "python/sglang/srt/runtime_context.py",
                "test/manual/kernels/audit_flashinfer_scoped_handoff.py",
                "test/manual/kernels/audit_flashinfer_scoped_handoff_bootstrap.py",
                "test/manual/kernels/audit_flashinfer_scoped_handoff_controls.py",
            )
        },
    }
    try:
        with (
            tempfile.TemporaryDirectory(prefix="scoped-handoff-") as temporary,
            bootstrap(sglang_root, source) as modules,
            mutation(mutation_name),
        ):
            report["boundaries"] = modules.boundaries
            report["controls"] = modules.controls
            for case in (
                gate_and_sequence,
                skips_after_startup,
                reuse_off,
                error_restoration,
            ):
                result = report["families"][case.__name__] = {}
                fixture = Fixture(modules, Path(temporary) / case.__name__, result)
                with fixture.active():
                    case(fixture)
                result["status"] = "passed"
            report["boundary_calls"] = modules.boundary_calls
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error))
        report["traceback"] = traceback.format_exc()
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mutation", choices=("omit-skips", "replay-new-root"))
    args = parser.parse_args()
    result = run(args.flashinfer_source, args.output, mutation_name=args.mutation)
    print(json.dumps({key: result[key] for key in ("status", "mutation")}))
    if result["status"] != "passed":
        print(result["traceback"])
    raise SystemExit(result["status"] != "passed")
