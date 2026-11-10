"""Inspect native FP8 tactic numerics on the first real batch-one down-projection."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    os.environ["SGLANG_CACHE_DIR"] = sys.argv[sys.argv.index("--cache") + 1]
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = "1"
    import model_study
    import torch
    from flashinfer.autotuner import AutoTuner

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    original = AutoTuner.choose_one
    findings = []

    def choose(tuner, op, runners, config, inputs, **kwargs):
        winner = original(tuner, op, runners, config, inputs, **kwargs)
        if (
            not findings
            and op == "fp8_gemm"
            and not tuner.is_tuning_mode
            and tuner._winner_cache_identity() is None
            and tuple(inputs[0].shape) == (1, 1, 14336)
            and tuple(inputs[1].shape) == (1, 14336, 4096)
        ):
            assert not torch.cuda.is_current_stream_capturing()
            runner, selected = winner
            a, b, sa, sb, out, workspace = inputs
            prior = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                reference = torch.bmm(a.float() * sa.float(), b.float() * sb.float())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prior
            _, count = runner._get_algos(inputs)
            tensors = {
                "a": a.cpu(),
                "b": b.cpu(),
                "scale_a": sa.cpu(),
                "scale_b": sb.cpu(),
                "reference": reference.cpu(),
            }
            results = []
            for tactic in (-1, 0, 1):
                assert tactic < count
                target = torch.empty_like(out)
                actual = (
                    runner.forward([a, b, sa, sb, target, workspace], tactic=tactic)
                    .float()
                    .clone()
                )
                error = (actual - reference).abs()
                results.append(
                    {
                        "tactic": tactic,
                        "finite": bool(torch.isfinite(actual).all()),
                        "max_abs_error_vs_fp32": float(error.max()),
                        "relative_l2_vs_fp32": float(
                            torch.linalg.vector_norm(actual - reference)
                            / torch.linalg.vector_norm(reference)
                        ),
                        "cosine_vs_fp32": float(
                            torch.nn.functional.cosine_similarity(
                                actual.flatten(), reference.flatten(), dim=0
                            )
                        ),
                        "allclose_vs_fp32_005_001": torch.allclose(
                            actual, reference, atol=0.05, rtol=0.01
                        ),
                        "max_abs_vs_default": 0.0
                        if tactic == -1
                        else float((actual.cpu() - tensors["out_-1"]).abs().max()),
                    }
                )
                tensors[f"out_{tactic}"] = actual.cpu()
            torch.save(tensors, output.with_suffix(".pt"))
            findings.append(
                {
                    "selected": selected,
                    "native_tactic_count": count,
                    "a_shape": list(a.shape),
                    "b_shape": list(b.shape),
                    "a_stride": list(a.stride()),
                    "b_stride": list(b.stride()),
                    "results": results,
                }
            )
        return winner

    try:
        with patch.object(AutoTuner, "choose_one", choose):
            model_study.main()
    finally:
        output.with_suffix(".kernel.json").write_text(
            json.dumps({"probes": findings}, indent=2) + "\n"
        )
    assert len(findings) == 1


if __name__ == "__main__":
    main()
