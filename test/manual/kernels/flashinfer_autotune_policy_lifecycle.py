"""CPU-only policy lifecycle qualification using complete FlashInfer modules.

Profiling times and GPU-only import boundaries are controlled, not measured.
The phase-bank adapter is research machinery using private tuner fields. It is
not a production integration or a CUDA capture/performance test.
"""

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_engine(source, manifest):
    """Load whole source files; bypass only eager GPU package initialization."""
    for name, expected in manifest["flashinfer_source_sha256"].items():
        if sha(source / name) != expected:
            raise RuntimeError("source mismatch: " + name)
    if any(
        name == "flashinfer" or name.startswith("flashinfer.") for name in sys.modules
    ):
        raise RuntimeError("run in a fresh interpreter without FlashInfer imported")
    os.environ["FLASHINFER_AUTOTUNE_TIMER"] = "cuda_event"
    os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
    os.environ.pop("FLASHINFER_TACTICS_BLOCKLIST", None)

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    def unavailable(*args, **kwargs):
        raise RuntimeError("GPU-only boundary unexpectedly reached")

    package("flashinfer", source / "flashinfer")
    package("flashinfer.jit", source / "flashinfer/jit")
    core = types.ModuleType("flashinfer.jit.core")
    core.logger = logging.getLogger("policy-lifecycle")
    sys.modules[core.__name__] = core
    tllm = types.ModuleType("flashinfer.tllm_utils")
    tllm.delay_kernel = unavailable
    sys.modules[tllm.__name__] = tllm
    utils = types.ModuleType("flashinfer.utils")
    utils.next_positive_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    utils.is_confidential_compute = lambda: False
    utils.get_globaltimer_kernel = unavailable
    sys.modules[utils.__name__] = utils
    api = importlib.import_module("flashinfer.autotuner")
    engine = importlib.import_module("flashinfer.autotuner.autotuner")
    cache = importlib.import_module("flashinfer.autotune_cache")
    import torch

    if torch.__version__.split("+")[0] != "2.14.1" or torch.cuda.is_available():
        raise RuntimeError("this frozen CPU proof requires Torch2.14.1 without CUDA")
    # The real collector expects a CUDA deployment. Pin its fixture stamp only;
    # real JSON generation, metadata comparison, merge and loading remain intact.
    engine._collect_metadata = lambda: dict(manifest["cpu_metadata_fixture"])
    torch.cuda.is_current_stream_capturing = lambda: False
    return api, engine, cache, torch


class Proof:
    def __init__(self, source, manifest):
        self.manifest = manifest
        self.api, self.engine, self.cache, self.torch = load_engine(source, manifest)
        self.tuner = self.api.AutoTuner.get()
        self.config = self.api.TuningConfig(use_cuda_graph=False)
        self.graph = self.cache.MeasurementPolicy(execution_mode="cuda_graph")
        self.inputs = [self.torch.full((2, 3), 11.0)]
        self.checks = []
        self.profiles = []
        self.op = "manual::policy_lifecycle"
        self.runner = self.make_runner()
        self.install_profile()

    def make_runner(self):
        class PolicyRunner(self.api.TunableRunner):
            def get_valid_tactics(self, inputs, profile):
                return [0, 1]

            def forward(self, inputs, tactic=-1):
                return inputs[0] + tactic

        return PolicyRunner()

    def install_profile(self):
        def profile(tuner, runner, inputs, tactic, tuning_config, **kwargs):
            wanted = 1 if tuning_config.use_cuda_graph else 0
            value = 1.0 if tactic == wanted else 4.0 if tactic == -1 else 2.0
            self.profiles.append(
                {
                    "tactic": tactic,
                    "time_fixture": value,
                    "use_cuda_graph": tuning_config.use_cuda_graph,
                    "use_cold_l2_cache": tuning_config.use_cold_l2_cache,
                    "identity": repr(tuner._winner_cache_identity()),
                }
            )
            return value

        self.tuner._profile_single_kernel = types.MethodType(profile, self.tuner)

    def check(self, name, condition, detail=None):
        if not condition:
            raise RuntimeError("failed: " + name + ": " + repr(detail))
        self.checks.append({"name": name, "passed": True, "detail": detail})

    def scope(self, mode="replay", policy="graph", **kwargs):
        return self.cache.autotune_v2(
            mode=mode,
            persistent_cache=False,
            measurement_policy=self.graph if policy == "graph" else policy,
            **kwargs,
        )

    def choose(self, expected, hit=True):
        runner, tactic = self.tuner.choose_one(
            self.op, [self.runner], self.config, self.inputs
        )
        config = self.config
        policy = self.tuner._effective_measure_policy
        if policy is not None:
            config = self.tuner._apply_measure_policy(config, policy)
        found, _, cached, _ = self.tuner.search_cache(
            self.op,
            [self.runner],
            tuple(tuple(x.shape) for x in self.inputs),
            config,
            inputs=self.inputs,
        )
        self.check("actual choose_one tactic", tactic == expected, tactic)
        if hit is not None:
            self.check("actual search_cache hit", found == hit, found)
        if hit:
            self.check("actual cached tactic", cached == expected, cached)
        output = runner(self.inputs, tactic=tactic)
        self.check(
            "CPU runner executes selected tactic",
            self.torch.equal(output, self.inputs[0] + expected),
        )
        return tactic

    def snapshot(self):
        t = self.tuner
        return {
            "legacy": dict(t.profiling_cache),
            "partitions": {k: dict(v) for k, v in t._winner_partitions.items()},
            "provenance": dict(t._profiling_cache_policies),
            "files": dict(t._file_configs),
            "file_policies": dict(t._file_config_policies),
            "records": copy.deepcopy(t._namespaced_records),
            "dirty_namespaces": set(t._dirty_namespaces),
            "dirty": t._dirty,
            "dirty_seq": t._dirty_seq,
            "generations": dict(t._observed_cache_generations),
        }

    def context_clean(self):
        t = self.tuner
        self.check("all tuning contexts exited", t._active_tuning_contexts == 0)
        self.check("tuning mode restored", not t.is_tuning_mode)
        self.check("v2 state restored", not t._v2_local.active)
        self.check("measurement policy restored", t._effective_measure_policy is None)
        self.check("no managed store attached", t._managed_cache is None)


def bank_paths(root, policy):
    fields = policy.manifest_fields()
    key = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return root / (key + ".json"), root / (key + ".manifest.json")


def export_graph_bank(proof, root, fail=False):
    """Manual-only bridge: project into a detached serializer, not live maps."""
    t = proof.tuner
    proof.check(
        "export requires graph scope", t._effective_measure_policy == proof.graph
    )
    path, meta = bank_paths(root, proof.graph)
    if not fail and (path.exists() or meta.exists()):
        raise FileExistsError("fresh export must not merge an existing phase bank")
    writer = proof.api.AutoTuner()
    identity = t._winner_cache_identity()
    writer.profiling_cache.update(t._winner_cache())
    for key in writer.profiling_cache:
        writer._profiling_cache_policies[(None, key)] = t._profiling_cache_policies[
            (identity, key)
        ]
    if fail:
        raise OSError("injected detached-export failure")
    writer.save_configs(str(path))
    write(
        meta,
        {
            "policy": proof.graph.manifest_fields(),
            "source_sha256": proof.manifest["flashinfer_source_sha256"],
            "cache_sha256": sha(path),
        },
    )
    return path, meta


def read_graph_bank(proof, root, requested=None):
    path, meta = bank_paths(root, proof.graph)
    stamp = json.loads(meta.read_text())
    requested = proof.graph if requested is None else requested
    if (
        stamp["policy"] != requested.manifest_fields()
        or stamp["source_sha256"] != proof.manifest["flashinfer_source_sha256"]
        or stamp["cache_sha256"] != sha(path)
    ):
        raise ValueError("phase bank policy/content mismatch")
    reader = proof.api.AutoTuner()
    if not reader.load_configs(str(path)):
        raise RuntimeError("real v1 bank load rejected")
    return reader


@contextlib.contextmanager
def activate_graph_bank(proof, reader):
    """Manual single-thread adapter; private file maps restored even on errors."""
    t = proof.tuner
    with t._lock:
        old_files, old_policies = t._file_configs, t._file_config_policies
        old_logged_hits = set(t._logged_file_hits)
        old_partition_ids = set(t._winner_partitions)
        with proof.scope():
            try:
                if t._winner_cache():
                    raise RuntimeError(
                        "a live memory winner would shadow this file bank"
                    )
                t._file_configs = dict(reader._file_configs)
                t._file_config_policies = dict(reader._file_config_policies)
                yield
            finally:
                t._file_configs, t._file_config_policies = old_files, old_policies
                t._logged_file_hits = old_logged_hits
                for identity in set(t._winner_partitions) - old_partition_ids:
                    if not t._winner_partitions[identity]:
                        del t._winner_partitions[identity]


def seed(p, out):
    t = p.tuner
    with p.scope("tune", policy=None):
        p.choose(0)
    p.context_clean()
    p.choose(0)
    t.save_configs(str(out / "legacy.json"))
    baseline = (out / "legacy.json").read_bytes()
    with p.scope("tune"):
        p.choose(1)
        p.check(
            "graph profiling overrides actual config", p.profiles[-1]["use_cuda_graph"]
        )
        before = p.snapshot()
        export_graph_bank(p, out)
        p.check("detached export preserves all live banks", p.snapshot() == before)
        try:
            export_graph_bank(p, out)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("existing bank was merged without validation")
        p.check(
            "existing export rejection preserves live banks", p.snapshot() == before
        )
        try:
            export_graph_bank(p, out, fail=True)
        except OSError:
            pass
        else:
            raise RuntimeError("injected export failure missing")
        p.check("failed export preserves all live banks", p.snapshot() == before)
        p.choose(1)
    p.context_clean()
    p.choose(0)
    t.save_configs(str(out / "plain-save-after-graph.json"))
    rows = json.loads((out / "plain-save-after-graph.json").read_text())
    entries = [v for k, v in rows.items() if k.startswith("(")]
    p.check(
        "plain v1 save omits graph winner", len(entries) == 1 and entries[0][1] == 0
    )
    p.check(
        "graph-bank export leaves legacy disk bytes unchanged",
        (out / "legacy.json").read_bytes() == baseline,
    )


def plain_reload(p, out):
    p.check("real legacy load succeeds", p.tuner.load_configs(str(out / "legacy.json")))
    p.choose(0)
    with p.scope():
        p.choose(0)
        p.check("replay does not enable tuning", not p.tuner.is_tuning_mode)
    p.check("wrong graph replay caused no profiling", len(p.profiles) == 0)
    with p.scope("tune"):
        p.choose(1)
    p.check("tuning rejects mismatched file provenance", len(p.profiles) == 3)
    p.context_clean()
    p.choose(0)


def bridge_reload(p, out):
    t = p.tuner
    p.check("real legacy load succeeds", t.load_configs(str(out / "legacy.json")))
    t._namespaced_records["manual-unrelated"] = {"target": "keep", "draft": 7}
    before = p.snapshot()
    original_maps = (t._file_configs, t._file_config_policies)
    reader = read_graph_bank(p, out)
    p.check("isolated load preserves live state", p.snapshot() == before)
    for policy in [
        p.cache.MeasurementPolicy(execution_mode="eager"),
        p.cache.MeasurementPolicy(execution_mode="cuda_graph", _timer="cupti"),
    ]:
        try:
            read_graph_bank(p, out, requested=policy)
        except ValueError:
            pass
        else:
            raise RuntimeError("wrong full policy accepted")
    p.choose(0)
    bank_file, _ = bank_paths(out, p.graph)
    original_bytes = bank_file.read_bytes()
    bank_file.write_bytes(original_bytes + b" ")
    try:
        read_graph_bank(p, out)
    except ValueError:
        pass
    else:
        raise RuntimeError("modified cache bytes accepted")
    finally:
        bank_file.write_bytes(original_bytes)
    p.check(
        "rejected tampered bank leaves live state unchanged", p.snapshot() == before
    )
    for failing in (False, True):
        try:
            with activate_graph_bank(p, reader):
                p.choose(1)  # capture-time lookup surrogate; no CUDA graph executed.
                p.check(
                    "loaded graph winner remains file-sourced", not t._winner_cache()
                )
                if failing:
                    raise OSError("injected capture-forward failure")
        except OSError:
            if not failing:
                raise
        p.context_clean()
        p.check(
            "original file-map identities restored",
            t._file_configs is original_maps[0]
            and t._file_config_policies is original_maps[1],
        )
        p.check("all live banks restored", p.snapshot() == before)
        p.choose(0)
    p.check("fresh-process bridge performs zero profiling", not p.profiles)
    # A matching file must not silently lose to a previously populated bank.
    with p.scope():
        identity = t._winner_cache_identity()
        config = t._apply_measure_policy(p.config, p.graph)
        key = t._get_cache_key(
            p.op, p.runner, tuple(tuple(x.shape) for x in p.inputs), config
        )
        t._winner_cache()[key] = (0, None)
    shadowed = p.snapshot()
    try:
        with activate_graph_bank(p, reader):
            raise AssertionError("shadowing memory winner was accepted")
    except RuntimeError as error:
        p.check("shadowing memory winner rejected", "shadow" in str(error))
    else:
        raise RuntimeError("shadowing guard missing")
    p.check("shadow rejection preserves existing banks", p.snapshot() == shadowed)
    del t._winner_partitions[identity]
    p.check("negative fixture removed exactly", p.snapshot() == before)


def file_order(p, out):
    t = p.tuner
    graph_file, _ = bank_paths(out, p.graph)
    p.check("first legacy load succeeds", t.load_configs(str(out / "legacy.json")))
    p.choose(0)
    p.check("second graph load succeeds", t.load_configs(str(graph_file)))
    p.choose(1)
    p.check("last legacy reload succeeds", t.load_configs(str(out / "legacy.json")))
    p.choose(0)
    p.check("file collisions reproduce with zero profiling", not p.profiles)


def controls(p, out):
    t = p.tuner
    # Missing explicit-policy winners fall back after scope exit and do not save.
    with p.scope("tune"):
        p.choose(1)
    p.choose(-1, hit=False)
    t.save_configs(str(out / "graph-only-plain-save.json"))
    p.check(
        "graph-only plain save contains no tactic",
        not any(
            k.startswith("(")
            for k in json.loads((out / "graph-only-plain-save.json").read_text())
        ),
    )
    before_profiles = len(p.profiles)
    with p.api.autotune(True):
        with p.scope():
            p.check("inner replay retains outer v1 tuning", t.is_tuning_mode)
            p.choose(1)
    p.check(
        "warm nested scope performs no profiling", len(p.profiles) == before_profiles
    )
    try:
        with p.scope():
            with p.scope():
                raise AssertionError("nested v2 unexpectedly entered")
    except RuntimeError as error:
        p.check("nested v2 is explicitly rejected", "nested autotune_v2" in str(error))
    else:
        raise RuntimeError("nested-v2 guard missing")
    p.context_clean()
    with p.scope("tune", skip_ops=[p.op]):
        p.choose(-1, hit=None)
    p.check("skip prevents profiling", len(p.profiles) == before_profiles)
    with p.api.autotune(True, cache=str(out / "fresh-run.json")):
        with p.scope("tune"):
            p.choose(1)
    p.check(
        "unique output path does not force existing policy winner to retune",
        len(p.profiles) == before_profiles,
    )
    p.context_clean()
    # Empty policy fields share legacy memory and therefore its v1 serialization.
    with p.scope("tune", policy=p.cache.MeasurementPolicy()):
        p.choose(0)
    p.choose(0)
    t.save_configs(str(out / "empty-policy.json"))
    eager = p.cache.MeasurementPolicy(execution_mode="eager")
    eager_events = p.cache.MeasurementPolicy(execution_mode="eager", _timer="events")
    p.check(
        "timer manifests differ",
        eager.manifest_fields() != eager_events.manifest_fields(),
    )
    p.check(
        "v1 provenance does not encode complete policy",
        t._profiling_policy(t._apply_measure_policy(p.config, eager))
        == t._profiling_policy(t._apply_measure_policy(p.config, eager_events)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "controller",
            "seed",
            "plain_reload",
            "bridge_reload",
            "file_order",
            "controls",
        ],
        default="controller",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "controller":
        if any(args.output_dir.iterdir()):
            raise FileExistsError("controller requires a fresh empty output directory")
        reports = []
        for stage in (
            "seed",
            "plain_reload",
            "bridge_reload",
            "file_order",
            "controls",
        ):
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
        if len({report["pid"] for report in reports}) != len(reports):
            raise RuntimeError("fresh process identities were not distinct")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
        write(
            args.output_dir / "result.json",
            {
                "status": "success",
                "sglang_execution_commit": commit,
                "worker_sha256": sha(Path(__file__)),
                "manifest_sha256": sha(args.manifest),
                "fresh_processes": len(reports),
                "assertions_passed": sum(len(r["checks"]) for r in reports),
                "reports": reports,
                "limits": json.loads(args.manifest.read_text())["limits"],
            },
        )
        print(
            json.dumps(
                {
                    "status": "success",
                    "fresh_processes": len(reports),
                    "assertions_passed": sum(len(r["checks"]) for r in reports),
                }
            )
        )
        return
    manifest = json.loads(args.manifest.read_text())
    p = Proof(args.flashinfer_source.resolve(), manifest)
    globals()[args.stage](p, args.output_dir)
    p.context_clean()
    write(
        args.output_dir / (args.stage + ".json"),
        {
            "status": "success",
            "stage": args.stage,
            "pid": os.getpid(),
            "python": sys.version,
            "torch": p.torch.__version__,
            "cuda_available": p.torch.cuda.is_available(),
            "source_commit": manifest["flashinfer_release_commit"],
            "source_sha256": manifest["flashinfer_source_sha256"],
            "checks": p.checks,
            "profiles": p.profiles,
            "seams": manifest["controlled_boundaries"],
        },
    )


if __name__ == "__main__":
    main()
