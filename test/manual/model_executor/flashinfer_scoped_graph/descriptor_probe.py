"""Native descriptor persistence through SGLang's scoped graph wrapper.

The natural fixture uses ordinary default-inclusive selection. The explicit
fixture uses FlashInfer's real rank_tactics(k=2) to construct a descriptor-only
winner without fabricated timings; ordinary API lookup consumes that winner.
This is correctness/persistence qualification, not a performance comparison.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*.json")
    }


def run(args, result):
    import flashinfer
    import torch
    import torch.nn.functional as F
    from flashinfer import bmm_fp8
    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm import gemm_base

    from sglang.srt.model_executor.runner.flashinfer_autotune import (
        FlashInferGraphAutotune,
    )

    assert flashinfer.__git_commit__ == "0bdafda8f721aaa4981c017a61b99f1122267edf"
    assert torch.cuda.device_count() == 1
    result["environment"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "git_commit": flashinfer.__git_commit__,
        "gpu": torch.cuda.get_device_name(),
        "source": flashinfer.__file__,
    }
    tuner = AutoTuner.get()
    original_profile, original_choose = (
        AutoTuner._profile_single_kernel,
        AutoTuner.choose_one,
    )
    original_check = gemm_base._check_cublaslt_descriptor_ran
    profile_calls, choices, native = [], [], []
    last_call = {}

    def profile(self, runner, inputs, tactic, config, **kwargs):
        value = original_profile(self, runner, inputs, tactic, config, **kwargs)
        profile_calls.append(repr(tactic))
        return value

    def choose(self, *a, **k):
        winner = original_choose(self, *a, **k)
        assert not k, "Probe expects the production positional call"
        last_call.update(op=a[0], runners=a[1], config=a[2], inputs=a[3])
        choices.append(
            {
                "tactic": winner[1],
                "identity": repr(self._winner_cache_identity()),
                "strict": self._v2_local.strict_replay,
            }
        )
        return winner

    def check(used, tactic):
        native.append({"used_descriptor": used, "tactic": tactic})
        return original_check(used, tactic)

    AutoTuner._profile_single_kernel, AutoTuner.choose_one = profile, choose
    gemm_base._check_cublaslt_descriptor_ran = check
    result["cache_before"] = hashes(args.cache)
    if args.phase == "cold":
        assert not result["cache_before"]
    else:
        assert result["cache_before"]
        cold = json.loads(args.cold_result.read_text())
        assert cold["status"] == "passed"
    try:
        for fixture in ("natural", "explicit_descriptor"):
            for label, m, n, k in (
                ("exact_m1", 1, 4096, 14336),
                ("bucket_m500", 500, 1024, 2048),
            ):
                torch.manual_seed(142)
                a_ref = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
                b_ref = torch.randn(
                    1, n, k, device="cuda", dtype=torch.bfloat16
                ).transpose(-2, -1)
                reference = torch.bmm(a_ref, b_ref)

                def quantize(x):
                    scale = x.abs().max().float() / torch.finfo(torch.float8_e4m3fn).max
                    return (x.float() / scale).to(torch.float8_e4m3fn), scale

                a, sa = quantize(a_ref)
                b, sb = quantize(b_ref)
                saved_a = a.clone()
                out = torch.empty_like(reference)
                del a_ref, b_ref

                def forward():
                    bmm_fp8(a, b, sa, sb, torch.bfloat16, out, backend="cublas")

                def validate(expected):
                    torch.cuda.synchronize()
                    assert torch.isfinite(out).all()
                    cosine = F.cosine_similarity(
                        expected.float().flatten(), out.float().flatten(), dim=0
                    ).item()
                    assert cosine > 0.99, cosine
                    return cosine

                identity_before = repr(tuner._winner_cache_identity())
                forward()  # Materialize buffers and capture the real API's runner/config/inputs.
                assert choices[-1]["tactic"] == -1
                row = {
                    "fixture": fixture,
                    "label": label,
                    "shape": [m, n, k],
                    "bare_before_cosine": validate(reference),
                }
                result["cases"].append(row)
                scope = FlashInferGraphAutotune(
                    args.cache / f"{fixture}-{label}", set()
                )
                before = len(profile_calls)
                with scope._context("tune"):
                    assert tuner._effective_measure_policy is None
                    if fixture == "explicit_descriptor" and args.phase == "cold":
                        ranked = tuner.rank_tactics(
                            last_call["op"],
                            last_call["runners"],
                            last_call["config"],
                            last_call["inputs"],
                            k=2,
                        )
                        assert ranked and gemm_base._is_cublaslt_algo_tactic(ranked[0])
                        row["ranking_fixture"] = ranked
                    forward()
                    winner = choices[-1]["tactic"]
                row["winner"] = winner
                if fixture == "explicit_descriptor":
                    assert gemm_base._is_cublaslt_algo_tactic(winner)
                if args.phase == "warm":
                    saved = next(
                        c
                        for c in cold["cases"]
                        if c["fixture"] == fixture and c["label"] == label
                    )
                    assert json.loads(json.dumps(winner)) == saved["winner"]
                row["initial_profile_count"] = len(profile_calls) - before
                if args.phase == "warm":
                    assert row["initial_profile_count"] == 0
                begin_native = len(native)
                coverage_before = len(profile_calls)
                with scope.capture_context(forward, lambda: out.fill_(float("nan"))):
                    assert tuner._v2_local.strict_replay
                    forward()
                    assert choices[-1]["tactic"] == winner
                    row["eager_cosine"] = validate(reference)
                    eager = out.clone()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        forward()
                    assert choices[-1]["tactic"] == winner
                assert repr(tuner._winner_cache_identity()) == identity_before
                assert len(profile_calls) == coverage_before
                out.fill_(float("nan"))
                graph.replay()
                row["graph_cosine"] = validate(reference)
                assert torch.equal(out, eager)
                a.copy_((-saved_a.float()).to(a.dtype))
                out.fill_(float("nan"))
                graph.replay()
                row["changed_input_graph_cosine"] = validate(-reference)
                a.copy_(saved_a)
                out.fill_(float("nan"))
                graph.replay()
                row["restored_input_graph_cosine"] = validate(reference)
                assert torch.equal(out, eager)
                row["native_consumption"] = native[begin_native:]
                if fixture == "explicit_descriptor":
                    assert row["native_consumption"]
                    assert all(
                        r["used_descriptor"] == 1 and r["tactic"] == winner
                        for r in row["native_consumption"]
                    )
                forward()
                assert (
                    choices[-1]["tactic"] == -1
                    and choices[-1]["identity"] == identity_before
                )
                row["bare_after_cosine"] = validate(reference)
                row["strict_capture_no_profiles"] = True
                row["graph_matches_eager_exactly"] = True
                row["legacy_identity_restored"] = True
                graph.reset()
        result["profile_count"] = len(profile_calls)
        result["fallback_count"] = sum(
            gemm_base._CUBLASLT_TACTIC_REMATERIALIZED.values()
        )
        assert result["fallback_count"] == 0
        result["cache_after"] = hashes(args.cache)
        assert result["cache_after"]
        if args.phase == "warm":
            assert result["cache_before"] == result["cache_after"]
            assert not profile_calls
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/descriptor-workspace/" in line and ".so" in line
            }
        )
        assert libraries
        result["native_libraries"] = {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libraries
        }
        assert all("g0bdafda8f721" in p for p in libraries)
    finally:
        AutoTuner._profile_single_kernel, AutoTuner.choose_one = (
            original_profile,
            original_choose,
        )
        gemm_base._check_cublaslt_descriptor_ran = original_check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["cold", "warm"], required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cold-result", type=Path)
    args = p.parse_args()
    assert not args.output.exists()
    result = {"phase": args.phase, "cases": [], "status": "started"}
    try:
        run(args, result)
        result["status"] = "passed"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
