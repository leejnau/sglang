"""Matched SGLang decode-graph cache qualification; one process per boot."""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from unittest.mock import patch


def files(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "flashinfer" / "autotune").rglob("*.json")
    }


def run(args, result):
    os.environ["SGLANG_CACHE_DIR"] = str(args.cache)
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = str(
        int(args.arm == "scoped")
    )
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = str(int(not args.no_reuse))
    import flashinfer
    import numpy as np
    import torch
    from flashinfer.autotuner import AutoTuner

    from sglang.benchmark import one_batch as b
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import PortArgs, ServerArgs

    result["environment"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "gpu": torch.cuda.get_device_name(),
        "pid": os.getpid(),
        "sglang_source": b.__file__,
    }
    import inspect

    import flashinfer.autotune_cache as fi_cache
    import flashinfer.autotuner.autotuner as fi_tuner

    from sglang.srt.model_executor.runner import flashinfer_autotune as sg_tuner

    result["source_files"] = {
        name: {
            "path": str(Path(inspect.getfile(obj))),
            "sha256": hashlib.sha256(
                Path(inspect.getfile(obj)).read_bytes()
            ).hexdigest(),
        }
        for name, obj in (
            ("one_batch", b),
            ("sglang_autotune", sg_tuner),
            ("flashinfer_cache", fi_cache),
            ("flashinfer_tuner", fi_tuner),
        )
    }
    result["cache_before"] = files(args.cache)
    if args.phase == "cold":
        assert not result["cache_before"], "cold cache must be empty"
    else:
        assert result["cache_before"], "warm cache must be populated"
    torch.manual_seed(19)
    cfg = dict(
        model_path=str(args.model),
        tokenizer_path=str(args.model),
        quantization="modelopt_fp8",
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        max_running_requests=16,
        max_total_tokens=8192,
        context_length=512,
        mem_fraction_static=0.7,
        disable_overlap_schedule=True,
        attention_backend="triton",
        kv_cache_dtype="bfloat16",
        cuda_graph_config={
            "decode": {"backend": "full", "bs": [1, 4, 16], "max_bs": 16},
            "prefill": {"backend": "disabled"},
        },
    )
    if args.skip:
        cfg["flashinfer_autotune_skip_ops"] = ["fp8_gemm"]
    if args.disable:
        cfg["disable_flashinfer_autotune"] = True
    result["config"] = cfg
    server = ServerArgs(**cfg)
    server.resolve_once()
    b._set_envs_and_config(server)
    publish(server, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    b._init_process_global_configs()
    ports = PortArgs.init_new(server)
    tuner = AutoTuner.get()
    profiles, captures, selections = [], [], []
    result["profiles"], result["captures"], result["selections"] = (
        profiles,
        captures,
        selections,
    )
    original_profile, original_choose = (
        AutoTuner._profile_single_kernel,
        AutoTuner.choose_one,
    )
    original_capture = FullCudaGraphBackend.capture_one
    original_scope_capture = sg_tuner.FlashInferGraphAutotune.capture_context
    preparation_profiles = []
    result["capture_preparation_profile_deltas"] = preparation_profiles

    @contextlib.contextmanager
    def scoped_capture(scope, forward, reset):
        before = len(profiles)
        with original_scope_capture(scope, forward, reset):
            yield
        delta = len(profiles) - before
        preparation_profiles.append(delta)
        assert delta == 0, "startup preparation missed a tested capture shape"

    stage = ["startup"]

    def profile(t, runner, inputs, tactic, config, **kwargs):
        assert stage[0] == "startup", "profiling after startup"
        assert t._effective_measure_policy is None, "measurement policy changed"
        start = time.perf_counter()
        value = original_profile(t, runner, inputs, tactic, config, **kwargs)
        profiles.append(
            {
                "runner": type(runner).__name__,
                "tactic": repr(tactic),
                "policy": repr(t._effective_measure_policy),
                "ms": value,
                "seconds": time.perf_counter() - start,
                "shapes": [list(x.shape) for x in inputs if hasattr(x, "shape")],
            }
        )
        return value

    def choose(t, *a, **k):
        winner = original_choose(t, *a, **k)
        inputs = a[3] if len(a) > 3 else k["inputs"]
        capturing = torch.cuda.is_current_stream_capturing()
        native_tactic_count = None
        if capturing and winner[1] >= 0 and hasattr(winner[0], "_get_algos"):
            _, native_tactic_count = winner[0]._get_algos(inputs)
            assert winner[1] < native_tactic_count, (
                "capture would fall back from an invalid tactic"
            )
        selections.append(
            {
                "op": a[0],
                "runner": type(winner[0]).__name__,
                "tactic": repr(winner[1]),
                "identity": repr(t._winner_cache_identity()),
                "capturing": capturing,
                "strict": t._v2_local.strict_replay,
                "input_shapes": [list(x.shape) for x in inputs if hasattr(x, "shape")],
                "native_tactic_count": native_tactic_count,
            }
        )
        return winner

    def capture(backend, key, forward, *a, **k):
        assert tuner._effective_measure_policy is None, "measurement policy changed"
        before = len(profiles)
        report = tuner._v2_local.replay_report
        answer = original_capture(backend, key, forward, *a, **k)
        row = {
            "key": repr(key),
            "profile_delta": len(profiles) - before,
            "coverage": dataclasses.asdict(report) if report is not None else None,
        }
        captures.append(row)
        assert row["profile_delta"] == 0
        if args.arm == "scoped" and not args.disable and not args.no_reuse:
            assert report is not None and report.misses == 0
            assert report.hits + report.skipped > 0
        return answer

    with torch.inference_mode():
        start = time.perf_counter()
        with (
            patch.object(AutoTuner, "_profile_single_kernel", profile),
            patch.object(AutoTuner, "choose_one", choose),
            patch.object(FullCudaGraphBackend, "capture_one", capture),
            patch.object(
                sg_tuner.FlashInferGraphAutotune, "capture_context", scoped_capture
            ),
        ):
            bench, tokenizer = b.load_model(server, ports, 0, 0)
        torch.cuda.synchronize()
        result["startup_seconds"] = time.perf_counter() - start
        result["startup_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        result["loaded_native_paths"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if ".so" in line
                and any(name in line for name in ("flashinfer", "cublas", "cudnn"))
            }
        )
        result["cache_after_startup"] = files(args.cache)
        result["bare_identity"] = repr(tuner._winner_cache_identity())
        assert not tuner.is_tuning_mode and not tuner._v2_local.active
        assert tuner._winner_cache_identity() is None, (
            "graph identity leaked into serving"
        )
        if args.phase == "warm" and not args.no_reuse:
            assert len(profiles) == 0, "fresh warm boot profiled"
            assert result["cache_before"] == result["cache_after_startup"], (
                "warm boot rewrote caches"
            )
        if args.no_reuse and result["cache_before"]:
            stable = lambda inventory: {
                k: v for k, v in inventory.items() if "/runs/" not in k
            }
            assert stable(result["cache_before"]) == stable(
                result["cache_after_startup"]
            ), "reuse-off replaced stable winners"
        graph = bench.torch_runner.decode_cuda_graph_runner
        assert len(graph.backend._graphs) == 3
        tokens = [
            tokenizer.encode(s, add_special_tokens=False)[-1]
            for s in (" Rome", " Tokyo")
        ]
        assert tokens[0] != tokens[1]
        logits_out, numerics, perf = {}, [], []
        result["numerics"], result["performance"] = numerics, perf
        stage[0] = "serving"

        def forbidden(*a, **k):
            raise AssertionError("serving attempted additional profiling")

        def setup(bs, length):
            bench.clear()
            # Identical valid prompt token ids across every process/arm.
            seed = tokenizer.encode(
                "The capital of France is Paris. Describe this city.",
                add_special_tokens=False,
            )
            prompt = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
            reqs = b.prepare_synthetic_inputs_for_latency_test(
                bs, length, [prompt] * bs
            )
            _, _, batch = bench.extend(reqs)
            return batch

        with patch.object(AutoTuner, "_profile_single_kernel", forbidden):
            for bs in (1, 3, 4, 7, 16):
                for length in (32, 256):
                    outputs = []
                    for label, token in zip(
                        ("original", "changed", "restored"),
                        (tokens[0], tokens[1], tokens[0]),
                    ):
                        arms = {}
                        for mode in ("eager", "graph"):
                            batch = setup(bs, length)
                            ctx = (
                                patch.object(graph, "can_run_graph", lambda _: False)
                                if mode == "eager"
                                else contextlib.nullcontext()
                            )
                            if mode == "graph":
                                for out in graph.backend._outputs.values():
                                    out.next_token_logits.fill_(float("nan"))
                            with ctx:
                                _, logits = bench.decode(
                                    torch.full(
                                        (bs,), token, device="cuda", dtype=torch.int64
                                    ),
                                    batch,
                                )
                            arms[mode] = logits.detach().float().clone()
                            assert torch.isfinite(arms[mode]).all()
                            logits_out[f"b{bs}_l{length}_{label}_{mode}"] = (
                                arms[mode].cpu().numpy()
                            )
                        row = {
                            "batch": bs,
                            "length": length,
                            "case": label,
                            "max_abs_error": float(
                                (arms["eager"] - arms["graph"]).abs().max()
                            ),
                            "allclose": torch.allclose(
                                arms["eager"], arms["graph"], atol=0.05, rtol=0.01
                            ),
                            "argmax_equal": torch.equal(
                                arms["eager"].argmax(-1), arms["graph"].argmax(-1)
                            ),
                        }
                        numerics.append(row)
                        assert row["allclose"] and row["argmax_equal"], row
                        outputs.append(arms["graph"])
                    assert not torch.equal(outputs[0], outputs[1]), (
                        "changed input ineffective"
                    )
                    assert torch.allclose(outputs[0], outputs[2], atol=0.05, rtol=0.01)
                    if args.no_perf or bs not in (1, 4, 16):
                        continue
                    batch = setup(bs, length)
                    token = torch.full(
                        (bs,), tokens[0], device="cuda", dtype=torch.int64
                    )
                    bench.decode(token, batch)
                    native = graph.backend._graphs[graph._replay_graph_key]
                    for _ in range(30):
                        native.replay()
                    torch.cuda.synchronize()
                    graph_ms = []
                    for _ in range(20):
                        start_event, end_event = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        start_event.record()
                        for _ in range(50):
                            native.replay()
                        end_event.record()
                        end_event.synchronize()
                        graph_ms.append(start_event.elapsed_time(end_event) / 50)
                    decode_ms = []
                    for _ in range(5):
                        batch = setup(bs, length)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for step in range(64):
                            bench.decode(token, batch)
                        torch.cuda.synchronize()
                        decode_ms.append((time.perf_counter() - start) * 1000 / 64)
                    perf.append(
                        {
                            "batch": bs,
                            "length": length,
                            "graph_ms": graph_ms,
                            "decode_ms": decode_ms,
                        }
                    )
        np.savez(args.output.with_suffix(".npz"), **logits_out)
        result["cache_after_serving"] = files(args.cache)
        assert result["cache_after_serving"] == result["cache_after_startup"]
        result["status"] = "passed"
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("legacy", "scoped"), required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    for name in ("skip", "disable", "no-reuse", "no-perf"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "preserve previous attempt"
    result = {
        "status": "started",
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "started": time.time(),
    }
    try:
        run(args, result)
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished"] = time.time()
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
