"""CPU guards for the MLP-only adapter; inherited protocol has separate controls."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    kernels = args.repo / "test/manual/kernels"
    sys.path.insert(0, str(kernels))
    path = kernels / "bench_flashinfer_mlp_confirmation.py"
    spec = importlib.util.spec_from_file_location("mlp_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    checks = []

    def checked(label, action):
        action()
        checks.append(label)

    def rejected(label, action):
        try:
            action()
        except (RuntimeError, AssertionError):
            checks.append(label)
        else:
            raise AssertionError("accepted invalid adapter state: " + label)

    class Tensor:
        def __init__(self, shape, dtype="fp8", pointer=1, strides=None):
            self.shape, self.dtype, self.device = shape, dtype, "cuda:0"
            self.pointer, self.strides = pointer, strides

        def data_ptr(self):
            return self.pointer

        def stride(self):
            return self.strides

        def is_contiguous(self):
            return True

        def clone(self):
            raise AssertionError("initializer cloned a synthetic tensor")

        def copy_(self, other):
            raise AssertionError("initializer copied a synthetic tensor")

    draws, calls = [], []

    def initialize(shape, dtype, device):
        value = Tensor(shape, dtype)
        draws.append(value)
        return value

    q = Tensor((1, 144, 3072))
    draw = worker.original_initializer(initialize, q, "synthetic", calls)
    for m in worker.original.PROFILE_MS:
        value = draw((1, m, 3072), "fp8", "cuda:0")
        assert value is draws[-1]
        assert not calls[-1]["cloned"] and not calls[-1]["copied"]
        assert "generated_sha256" not in calls[-1]
    checks.append(
        "all nine original draws return the identical tensor without copy or hash"
    )
    assert [r["shape"][1] for r in calls] == list(worker.original.PROFILE_MS)
    assert sum(r["targeted"] for r in calls) == 1
    checks.append("original profile endpoints and exact runtime target are preserved")
    admissions, hashes = [], []

    def tensor_hash(value):
        hashes.append(value)
        return "a" * 64

    draw = worker.original_initializer(
        initialize, q, "synthetic", admissions, tensor_hash
    )
    value = draw((1, 144, 3072), "fp8", "cuda:0")
    assert (
        hashes == [value]
        and admissions[0]["generated_sha256"] == admissions[0]["returned_sha256"]
    )
    checks.append("admission attests the original tensor once without replacement")
    rejected(
        "captured-value initializer substitution rejected",
        lambda: worker.original_initializer(initialize, q, "captured", []),
    )

    original_globals = (
        worker.original.FIXTURE_SHA,
        worker.original.FIXTURE_MANIFEST_SHA,
        worker.original.make_initializer,
    )
    experiment = worker.Experiment.__new__(worker.Experiment)
    experiment.plan = {
        "fixture_sha256": "fixture",
        "fixture_manifest_sha256": "manifest",
    }

    def fail_setup(self):
        assert worker.original.FIXTURE_SHA == "fixture"
        assert worker.original.FIXTURE_MANIFEST_SHA == "manifest"
        assert worker.original.make_initializer is worker.original_initializer
        raise RuntimeError("deliberate setup failure")

    with patch.object(worker.ConfirmationExperiment, "setup", fail_setup):
        rejected(
            "scoped MLP setup executes actual adapter substitutions", experiment.setup
        )
    assert original_globals == (
        worker.original.FIXTURE_SHA,
        worker.original.FIXTURE_MANIFEST_SHA,
        worker.original.make_initializer,
    )
    checks.append(
        "all fixture and initializer helper globals restore after setup failure"
    )
    original_startup = worker.original.Experiment.startup

    def fail_startup(self):
        assert worker.original.Experiment.startup is worker.Experiment._original_startup
        raise RuntimeError("deliberate startup failure")

    with patch.object(worker.ConfirmationExperiment, "startup", fail_startup):
        rejected(
            "parent validation receives the MLP startup bridge", experiment.startup
        )
    assert worker.original.Experiment.startup is original_startup
    checks.append("startup bridge restores the original helper after exceptions")

    class CublasFp8GemmRunner:
        def get_cache_key_extras(self, inputs):
            return ()

    runner = CublasFp8GemmRunner()

    def public_case(tier="memory", mutate=None):
        e = worker.Experiment.__new__(worker.Experiment)
        e.torch = SimpleNamespace(
            float8_e4m3fn="fp8",
            bfloat16="bf16",
            uint8="u8",
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        )
        e.weight, e.scale_a, e.scale_b = (Tensor((1,), pointer=p) for p in (2, 3, 4))
        e.original_choose = lambda *a, **kw: (runner, -1)
        e.original_search = lambda *a, **kw: (tier == "memory", 0, -1, None)
        e.tuner = SimpleNamespace(
            _get_cache_key=lambda *a: SimpleNamespace(file_key="MLP-key")
        )
        e.scope_state, e.result = lambda: {}, {"selections": []}
        inputs = [
            Tensor((1, 144, 3072)),
            Tensor((1, 3072, 1024), pointer=2, strides=(3145728, 1, 3072)),
            e.scale_a,
            e.scale_b,
            Tensor((1, 144, 1024), "bf16", pointer=5),
            Tensor((1,), "u8", pointer=6),
        ]
        if mutate:
            mutate(e, inputs)
        saved = {}
        actual = e.choose_recorder(-1, tier, "control", saved)(
            "fp8_gemm", [runner], object(), inputs
        )
        assert actual == (runner, -1) and saved["inputs"] == tuple(inputs)
        assert saved["selection"]["key"] == "MLP-key"
        return saved

    checked("MLP public operand guards accept an actual memory winner", public_case)
    checked(
        "MLP public operand guards preserve genuine default miss",
        lambda: public_case("miss"),
    )
    rejected(
        "old QKV shape cannot pass the MLP guard",
        lambda: public_case(
            mutate=lambda e, xs: setattr(xs[0], "shape", (1, 144, 1024))
        ),
    )
    rejected(
        "wrong weight layout rejected",
        lambda: public_case(
            mutate=lambda e, xs: setattr(xs[1], "strides", (3145728, 1024, 1))
        ),
    )
    rejected(
        "different static weight allocation rejected",
        lambda: public_case(mutate=lambda e, xs: setattr(xs[1], "pointer", 9)),
    )
    rejected(
        "lost natural cache hit rejected",
        lambda: public_case(
            mutate=lambda e, xs: setattr(
                e, "original_search", lambda *a, **kw: (False, 0, -1, None)
            )
        ),
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture, manifest = root / "fixture", root / "manifest.json"
        fixture.write_bytes(b"private fixture stand-in")
        valid = {
            "schema_version": 2,
            "fixture_kind": "qwen3_mlp_down_proj",
            "shape": [144, 1024, 3072],
            "capture": {
                "weight_order": ["down_proj"],
                "activation_byte_match": True,
                "checkpoint_tensor_byte_match": True,
                "full_bf16_forward_completed": True,
            },
        }
        options = SimpleNamespace(fixture=fixture, fixture_manifest=manifest)

        def constructor(payload, mutate_plan=None):
            manifest.write_text(json.dumps(payload))
            plan = {
                "fixture_sha256": digest(fixture),
                "fixture_manifest_sha256": digest(manifest),
            }
            if mutate_plan:
                mutate_plan(plan)

            def initialize_parent(self, args):
                self.plan, self.result = plan, {"limits": []}

            with patch.object(
                worker.ConfirmationExperiment, "__init__", initialize_parent
            ):
                e = worker.Experiment(options)
            assert (
                e.result["mlp_adapter"]["initializer_recipe"]
                == "original_synthetic_no_clone_copy"
            )

        checked(
            "frozen MLP manifest and adapter evidence accepted",
            lambda: constructor(valid),
        )
        for key, value in (("shape", [144, 4096, 1024]), ("fixture_kind", "qwen3_qkv")):
            malformed = copy.deepcopy(valid)
            malformed[key] = value
            rejected(
                "wrong fixture " + key + " rejected", lambda d=malformed: constructor(d)
            )
        malformed = copy.deepcopy(valid)
        malformed["capture"]["activation_byte_match"] = False
        rejected(
            "unverified captured activation rejected", lambda: constructor(malformed)
        )
        rejected(
            "unfrozen fixture byte hash rejected",
            lambda: constructor(valid, lambda p: p.update(fixture_sha256=None)),
        )

    report = {
        "status": "passed",
        "controls": len(checks),
        "cases": checks,
        "worker_sha256": digest(path),
        "controls_sha256": digest(__file__),
        "scope": "MLP adapter identity, original initializer and exception restoration; parent confirmation protocol controls remain separate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "controls", "worker_sha256")}))


if __name__ == "__main__":
    main()
