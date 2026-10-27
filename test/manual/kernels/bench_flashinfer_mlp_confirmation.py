"""Shadow finalist confirmation on the frozen real MLP down projection.

Reuse the qualified confirmation protocol while preserving the original MLP
startup initializer. Neither the initial winner nor its saved cache is changed.
The original and shadow decisions share the same two post-confirmation graphs.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_flashinfer_objective_values as original
from bench_flashinfer_finalist_confirmation import Experiment as ConfirmationExperiment
from bench_flashinfer_finalist_confirmation import parse_args
from bench_flashinfer_objective_values import require, sha, tactic_json

CONFIRMATION_SHA = "3ec7d31ea510b29b9e5e2c289e82a3cfe31867b11ba4a28d9d900d76c1fa6413"
SHAPE = (144, 1024, 3072)


def original_initializer(initialize, q, values, records, tensor_hash=None):
    """Observe the original draw without retaining, cloning or copying it."""
    require(
        values == "synthetic", "MLP confirmation requires original synthetic inputs"
    )

    def draw(shapes, dtype, device):
        generated = initialize(shapes, dtype, device)
        row = {
            "shape": list(shapes),
            "dtype": str(dtype),
            "device": str(device),
            "targeted": tuple(shapes) == tuple(q.shape),
            "cloned": False,
            "copied": False,
            "returned_same_object": True,
        }
        if tensor_hash is not None:
            value = tensor_hash(generated)
            row.update(generated_sha256=value, returned_sha256=value)
        records.append(row)
        return generated

    return draw


class Experiment(ConfirmationExperiment):
    def __init__(self, args):
        require(
            sha(inspect.getfile(ConfirmationExperiment)) == CONFIRMATION_SHA,
            "confirmation parent source changed",
        )
        super().__init__(args)
        manifest = json.loads(args.fixture_manifest.read_text())
        require(
            self.plan["fixture_sha256"] == sha(args.fixture)
            and self.plan["fixture_manifest_sha256"] == sha(args.fixture_manifest)
            and manifest["schema_version"] == 2
            and manifest["fixture_kind"] == "qwen3_mlp_down_proj"
            and manifest["shape"] == list(SHAPE)
            and manifest["capture"]["weight_order"] == ["down_proj"]
            and manifest["capture"]["activation_byte_match"] is True
            and manifest["capture"]["checkpoint_tensor_byte_match"] is True
            and manifest["capture"]["full_bf16_forward_completed"] is True,
            "MLP fixture or capture contract differs",
        )
        self.result["worker_sha256"] = sha(__file__)
        self.result["mlp_adapter"] = {
            "parent_worker_sha256": CONFIRMATION_SHA,
            "fixture_kind": manifest["fixture_kind"],
            "shape": list(SHAPE),
            "initializer_recipe": "original_synthetic_no_clone_copy",
            "fixture_sha256": self.plan["fixture_sha256"],
            "fixture_manifest_sha256": self.plan["fixture_manifest_sha256"],
            "original_initializer_returns_same_object": True,
        }
        self.result["limits"].append(
            "The original MLP synthetic initializer has no matched clone/copy. Retaining the final profiling operands and later reference/confirmation work still change allocation lifetime and history."
        )

    def setup(self):
        # These are identity/observation adapters in the unchanged manual parent,
        # scoped to setup; the actual runtime source and public APIs stay intact.
        with (
            patch.object(original, "FIXTURE_SHA", self.plan["fixture_sha256"]),
            patch.object(
                original, "FIXTURE_MANIFEST_SHA", self.plan["fixture_manifest_sha256"]
            ),
            patch.object(original, "make_initializer", original_initializer),
        ):
            super().setup()

    def _original_startup(self):
        """Run the real wrapper; parent validates all candidate/key/settings rows."""
        from sglang.srt.model_executor.runner import flashinfer_autotune as startup
        from sglang.srt.runtime_context import get_context, get_parallel

        require(
            Path(startup.__file__)
            .resolve()
            .is_relative_to(Path(__file__).resolve().parents[3] / "python"),
            "startup wrapper came from another checkout",
        )
        self.result["source"]["startup_wrapper_sha256"] = sha(startup.__file__)
        self.args.cache.parent.mkdir(parents=True, exist_ok=True)
        self.stage, self.allow_profiling = "startup", True
        self.result["startup"]["started_ns"] = time.perf_counter_ns()
        try:
            with (
                get_context().override_server_args(
                    moe_runner_backend="triton", flashinfer_autotune_skip_ops=[]
                ),
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                patch.object(
                    startup, "flashinfer_autotune_cache_path", lambda _: self.args.cache
                ),
                patch.object(self.flashinfer, "autotune_v2", self.startup_policy),
                startup.flashinfer_autotune_context(
                    self.model_runner, run_lm_head=False
                ),
            ):
                self.operation()
            self.torch.cuda.synchronize()
        finally:
            self.allow_profiling = False
            self.result["startup"]["ended_ns"] = time.perf_counter_ns()
        self.result["startup"].update(
            profile_attempts=self.profile_attempts,
            rng_after=self.rng(),
            cache_sha256=sha(self.args.cache),
            after_settings=self.settings(),
        )
        calls, profiles = (
            self.result["startup"][name] for name in ("initializer_calls", "profiles")
        )
        require(
            not self.result["protocol_errors"]
            and [row["shape"][1] for row in calls] == list(original.PROFILE_MS)
            and all(
                not row["cloned"] and not row["copied"] and row["returned_same_object"]
                for row in calls
            )
            and not self.tuner._file_configs
            and not self.tuner._active_managed_store,
            "startup initializer, profiling or cache protocol changed",
        )
        self.result["startup"]["candidate_order"] = {
            str(m): [row["tactic"] for row in profiles if row["m"] == m]
            for m in original.PROFILE_MS
        }
        data = json.loads(self.args.cache.read_text())
        self.result["startup"]["incidental_legacy_entries"] = sum(
            key.startswith("(") for key in data
        )
        self.cache_sha = sha(self.args.cache)
        self.memory_sha = self.cache_state_sha()
        self.memory_fields = dict(self.cache_state_field_sha256)
        self.result["startup"].update(
            cache_state_sha256=self.memory_sha,
            cache_state_field_sha256=self.memory_fields,
        )
        self.startup_attempts = self.profile_attempts
        self.assert_bare()

    def startup(self):
        # Reuse the parent's immutable first-pass winner and dynamic finalist
        # validation; replace only its QKV-specific matched-copy startup bridge.
        with patch.object(original.Experiment, "startup", type(self)._original_startup):
            super().startup()

    def choose_recorder(self, expected, tier, label, saved):
        def choose(op, runners, config, inputs, **kwargs):
            require(
                op == "fp8_gemm"
                and len(runners) == 1
                and type(runners[0]).__name__ == "CublasFp8GemmRunner"
                and not kwargs,
                "actual public runner pool differs",
            )
            result = self.original_choose(op, runners, config, inputs, **kwargs)
            hit, index, cached, _ = self.original_search(
                op,
                runners,
                tuple(tuple(t.shape) for t in inputs),
                config,
                inputs=inputs,
            )
            require(
                bool(hit) == (tier == "memory")
                and result[0] is runners[index]
                and (expected is None or result[1] == expected)
                and (not hit or cached == result[1]),
                "public lookup did not consume the expected winner",
            )
            a, b, sa, sb, out, workspace = inputs
            m, n, k = SHAPE
            require(
                list(a.shape) == [1, m, k]
                and a.is_contiguous()
                and a.dtype == self.torch.float8_e4m3fn
                and list(b.shape) == [1, k, n]
                and list(b.stride())[-2:] == [1, k]
                and b.data_ptr() == self.weight.data_ptr()
                and sa.data_ptr() == self.scale_a.data_ptr()
                and sb.data_ptr() == self.scale_b.data_ptr()
                and list(out.shape) == [1, m, n]
                and out.dtype == self.torch.bfloat16
                and workspace.dtype == self.torch.uint8,
                "actual MLP public operands differ",
            )
            key = self.tuner._get_cache_key(
                op,
                runners[0],
                tuple(tuple(t.shape) for t in inputs),
                config,
                runners[0].get_cache_key_extras(inputs),
            )
            row = {
                "stage": label,
                "tactic": tactic_json(result[1]),
                "cache_hit": bool(hit),
                "lookup_tier": tier,
                "capturing": self.torch.cuda.is_current_stream_capturing(),
                "key": key.file_key,
                **self.scope_state(),
            }
            self.result["selections"].append(row)
            saved.update(
                inputs=tuple(inputs),
                runner=runners[0],
                tactic=result[1],
                key=key,
                config=config,
                selection=row,
            )
            return result

        return choose


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), "refusing to overwrite worker result")
    experiment = None
    try:
        import torch

        with torch.inference_mode():
            experiment = Experiment.__new__(Experiment)
            experiment.__init__(args)
            result = experiment.run()
    except Exception as error:
        traceback.print_exc()
        result = getattr(
            experiment,
            "result",
            {
                "mode": args.mode,
                "trial": args.trial,
                "block": args.block,
                "pid": os.getpid(),
            },
        )
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
