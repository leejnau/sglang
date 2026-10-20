"""CPU qualification of a proposed upstream context-scoped managed cache API.

Uses the frozen policy-lifecycle proof's complete-module loader and controlled
CPU runner. This is an unpublished API proposal, not SGLang production routing,
CUDA capture validation or a performance benchmark.
"""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

from flashinfer_autotune_policy_lifecycle import Proof, sha, write


def graph_scope(p, out, mode="replay", **kwargs):
    options = {
        "mode": mode,
        "persistent_cache": True,
        "cache_root": out / "graph-store",
        "measurement_policy": p.graph,
        "cache_scope": "context",
    }
    options.update(kwargs)
    return p.cache.autotune_v2(**options)


def clean(p, ambient):
    t = p.tuner
    p.check("ambient object preserved", t._managed_cache is ambient)
    p.check("scope inactive after exit", not t._v2_local.active)
    p.check("policy reset after exit", t._effective_measure_policy is None)
    p.check("scope reset after exit", t._v2_local.cache_scope == "process")
    p.check("all tuning contexts exited", t._active_tuning_contexts == 0)
    p.check("tuning mode reset", not t.is_tuning_mode)


def load_legacy(p, out):
    p.check("real v1 load succeeds", p.tuner.load_configs(str(out / "legacy.json")))
    p.choose(0)
    return sha(out / "legacy.json")


def seed(p, out):
    with p.scope("tune", policy=None):
        p.choose(0)
    p.tuner.save_configs(str(out / "legacy.json"))
    stamp = load_legacy(p, out)
    maps = p.tuner._file_configs, p.tuner._file_config_policies
    with graph_scope(p, out, "tune"):
        p.check(
            "cold graph scope never attaches ambient", p.tuner._managed_cache is None
        )
        p.choose(1)
        p.check(
            "graph profiles use effective graph objective",
            all(x["use_cuda_graph"] for x in p.profiles[-3:]),
        )
    clean(p, None)
    p.choose(0)
    p.check(
        "legacy file unchanged by graph publication", sha(out / "legacy.json") == stamp
    )
    p.check(
        "legacy file maps never swapped",
        p.tuner._file_configs is maps[0] and p.tuner._file_config_policies is maps[1],
    )
    entries = list((out / "graph-store").glob("v2/*/entries/*.json"))
    p.check("real managed entry published", len(entries) == 1)
    p.check(
        "real managed tactic1 persisted",
        json.loads(entries[0].read_text())["tactic"] == 1,
    )


def warm(p, out):
    stamp = load_legacy(p, out)
    maps = p.tuner._file_configs, p.tuner._file_config_policies
    for failing in (False, True):
        try:
            with graph_scope(p, out):
                p.check(
                    "warm graph scope never attaches ambient",
                    p.tuner._managed_cache is None,
                )
                p.choose(1)
                if failing:
                    raise OSError("injected scoped-forward failure")
        except OSError:
            if not failing:
                raise
        clean(p, None)
        p.choose(0)
    for overrides in [
        {"cache_root": out / "absent-store"},
        {"measurement_policy": p.cache.MeasurementPolicy(execution_mode="eager")},
    ]:
        with graph_scope(p, out, **overrides):
            p.choose(-1, hit=False)
    clean(p, None)
    p.check("fresh managed reuse performs zero profiling", not p.profiles)
    p.check("legacy bytes unchanged", sha(out / "legacy.json") == stamp)
    p.check(
        "original file maps untouched",
        p.tuner._file_configs is maps[0] and p.tuner._file_config_policies is maps[1],
    )


def process_default(p, out):
    with p.scope("tune", policy=None):
        p.choose(0)
    with p.cache.autotune_v2(
        mode="tune",
        persistent_cache=True,
        cache_root=out / "process-store",
        measurement_policy=p.graph,
    ):
        p.choose(1)
        ambient = p.tuner._managed_cache
        p.check("default process context attaches", ambient is not None)
    clean(p, ambient)
    p.choose(1)
    before = len(p.profiles)
    # Exact existing SGLang memory-only/default-policy context remains legacy.
    with p.scope(policy=None):
        p.choose(0)
    p.check(
        "existing memory-only legacy reuse has no profiling", len(p.profiles) == before
    )
    clean(p, ambient)


def memory(p, out):
    stamp = load_legacy(p, out)
    forbidden = out / "memory-context-must-not-create"
    with graph_scope(p, out, "tune", persistent_cache=False, cache_root=forbidden):
        p.choose(1)
    before = len(p.profiles)
    with graph_scope(p, out, persistent_cache=False, cache_root=forbidden):
        p.choose(1)
    with graph_scope(
        p, out, persistent_cache=False, cache_root=forbidden, measurement_policy=None
    ):
        p.choose(0)
    with graph_scope(
        p,
        out,
        persistent_cache=False,
        cache_root=forbidden,
        measurement_policy=p.cache.MeasurementPolicy(),
    ):
        p.choose(0)
    clean(p, None)
    p.check("memory replay does not profile", len(p.profiles) == before)
    p.check("memory scopes touch no requested cache root", not forbidden.exists())
    p.check("memory scopes never resolve managed store", not p.tuner._managed_stores)
    p.check(
        "memory scopes leave legacy bytes unchanged", sha(out / "legacy.json") == stamp
    )


def overlap(p, out):
    load_legacy(p, out)
    t = p.tuner
    for failing in (False, True):
        with p.cache.autotune_v2(
            mode="replay", cache_root=out / "graph-store", measurement_policy=p.graph
        ):
            ambient_a = t._managed_cache
        entered, changed = threading.Event(), threading.Event()
        errors = []

        def scoped_reader():
            try:
                try:
                    with graph_scope(p, out):
                        p.check(
                            "scoped thread sees old ambient without changing it",
                            t._managed_cache is ambient_a,
                        )
                        local_store = t._active_managed_store
                        entered.set()
                        if not changed.wait(10):
                            raise RuntimeError("ambient-change rendezvous timed out")
                        p.check(
                            "scoped target survives another thread attachment",
                            t._active_managed_store is local_store,
                        )
                        p.choose(1)
                        if failing:
                            raise OSError("injected thread-scope exit")
                except OSError:
                    if not failing:
                        raise
                p.check("thread-local context cleaned", not t._v2_local.active)
            except BaseException as error:
                errors.append(repr(error))
                entered.set()

        worker = threading.Thread(target=scoped_reader, daemon=True)
        worker.start()
        try:
            if not entered.wait(10):
                raise RuntimeError("scope-entry rendezvous timed out")
            if errors:
                raise RuntimeError(str(errors))
            with p.cache.autotune_v2(
                mode="replay", cache_root=out / "new-ambient-store"
            ):
                ambient_b = t._managed_cache
                p.check("other thread can replace ambient", ambient_b is not ambient_a)
        finally:
            changed.set()
            worker.join(10)
        p.check("reader thread exits", not worker.is_alive())
        p.check("reader thread passes", not errors, errors)
        clean(p, ambient_b)
    p.check("overlap exercises replay only, zero profiling", not p.profiles)


def controls(p, out):
    load_legacy(p, out)
    original_maps = p.tuner._file_configs, p.tuner._file_config_policies
    for value in ("invalid", None, False):
        try:
            with graph_scope(p, out, cache_scope=value):
                raise AssertionError("invalid scope entered")
        except ValueError:
            pass
        else:
            raise RuntimeError("invalid scope accepted")
    clean(p, None)
    with graph_scope(p, out):
        try:
            with graph_scope(p, out):
                raise AssertionError("nested v2 entered")
        except RuntimeError as error:
            p.check("nested v2 rejected", "nested autotune_v2" in str(error))
        else:
            raise RuntimeError("nesting guard missing")
        p.choose(1)
        try:
            p.cache.autotune_v2_reload()
        except RuntimeError as error:
            p.check(
                "scoped reload fails before mutating caches",
                "reload inside" in str(error),
            )
        else:
            raise RuntimeError("unsupported scoped reload accepted")
        p.choose(1)
    legacy_winners = dict(p.tuner.profiling_cache)
    with graph_scope(p, out, "tune"):
        try:
            p.tuner.rank_tactics(p.op, [p.runner], p.config, p.inputs, k=2)
        except RuntimeError as error:
            p.check(
                "unsafe shortlist tuning rejected", "rank_tactics(k>1)" in str(error)
            )
        else:
            raise RuntimeError("unsafe shortlist path accepted")
        p.check(
            "single-winner rank delegates safely",
            p.tuner.rank_tactics(p.op, [p.runner], p.config, p.inputs, k=1) == [1],
        )
    p.check(
        "shortlist guard preserves legacy winner bank",
        p.tuner.profiling_cache == legacy_winners,
    )
    with p.api.autotune(True):
        with graph_scope(p, out):
            p.check("outer tuning counter behavior unchanged", p.tuner.is_tuning_mode)
            p.choose(1)
    clean(p, None)
    p.check(
        "validation and nesting never mutate file maps",
        p.tuner._file_configs is original_maps[0]
        and p.tuner._file_config_policies is original_maps[1],
    )
    p.check("controls require no profiling", not p.profiles)


def bundled(p, out):
    import os

    original = p.engine.load_from_file
    calls = []

    def bundled_hit(key):
        calls.append(key)
        return True, 0, 0, None

    p.engine.load_from_file = bundled_hit
    os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "1"
    try:
        p.choose(0)
        before = len(calls)
        with graph_scope(p, out, cache_root=out / "bundled-miss"):
            p.choose(-1, hit=False)
        p.check("identified scope does not query bundled source", len(calls) == before)
        p.choose(0)
        p.check("unscoped bundled lookup remains enabled", len(calls) > before)
    finally:
        p.engine.load_from_file = original
        os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
    clean(p, None)
    p.check("bundled lookup control performs no profiling", not p.profiles)


def invalidation(p, out):
    stamp = load_legacy(p, out)
    old_files = dict(p.tuner._file_configs)
    old_policies = dict(p.tuner._file_config_policies)
    with graph_scope(p, out, "tune", cache_root=out / "invalidation-store"):
        p.choose(1)
        config = p.tuner._apply_measure_policy(p.config, p.graph)
        p.tuner._discard_profile_cache(
            p.op, [p.runner], tuple(tuple(x.shape) for x in p.inputs), config, p.inputs
        )
        p.check(
            "scoped invalidation preserves legacy entries",
            p.tuner._file_configs == old_files,
        )
        p.check(
            "scoped invalidation preserves legacy provenance",
            p.tuner._file_config_policies == old_policies,
        )
        p.choose(1)
    clean(p, None)
    p.choose(0)
    p.check(
        "scoped invalidation leaves legacy disk bytes unchanged",
        sha(out / "legacy.json") == stamp,
    )
    p.check("actual invalidation forces scoped reprofiling", len(p.profiles) == 6)


STAGES = (
    "seed",
    "warm",
    "process_default",
    "memory",
    "overlap",
    "controls",
    "bundled",
    "invalidation",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("controller", *STAGES), default="controller"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "controller":
        if any(args.output_dir.iterdir()):
            raise FileExistsError("use a fresh output directory")
        reports = []
        for stage in STAGES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--flashinfer-source",
                str(args.flashinfer_source.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--output-dir",
                str(args.output_dir.resolve()),
                "--stage",
                stage,
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            (args.output_dir / (stage + ".log")).write_text(
                process.stdout + process.stderr
            )
            if process.returncode:
                raise RuntimeError(stage + " failed; see retained log")
            reports.append(
                json.loads((args.output_dir / (stage + ".json")).read_text())
            )
        if len({r["pid"] for r in reports}) != len(reports):
            raise RuntimeError("fresh process identities are not distinct")
        write(
            args.output_dir / "result.json",
            {
                "status": "success",
                "worker_sha256": sha(Path(__file__)),
                "base_worker_sha256": sha(
                    Path(__file__).with_name("flashinfer_autotune_policy_lifecycle.py")
                ),
                "manifest_sha256": sha(args.manifest),
                "fresh_processes": len(reports),
                "assertions_passed": sum(len(r["checks"]) for r in reports),
                "reports": reports,
                "limits": [
                    "Research-only upstream API proposal.",
                    "Uses scoped managed graph persistence as peer to unchanged v1 legacy persistence, not graph-winner v1 serialization.",
                    "Synthetic CPU runner, controlled profiling and metadata; no CUDA or performance evidence.",
                    "Overlap checks exercise replay-only attachment; concurrent tuning and serving remain unqualified.",
                ],
            },
        )
        print(
            json.dumps(
                {
                    "status": "success",
                    "processes": len(reports),
                    "checks": sum(len(r["checks"]) for r in reports),
                }
            )
        )
        return
    import os

    p = Proof(args.flashinfer_source.resolve(), json.loads(args.manifest.read_text()))
    # Managed entry points import the package export of this same CPU fixture.
    p.api._collect_metadata = p.engine._collect_metadata
    globals()[args.stage](p, args.output_dir)
    write(
        args.output_dir / (args.stage + ".json"),
        {
            "status": "success",
            "stage": args.stage,
            "pid": os.getpid(),
            "checks": p.checks,
            "profiles": p.profiles,
        },
    )


if __name__ == "__main__":
    main()
