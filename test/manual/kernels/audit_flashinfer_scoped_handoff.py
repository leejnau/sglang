"""Research-only control handoff for one dense TP1 target full-decode region.

There is no production caller. The caller owns prepared decode inputs, stream
dependencies and the original backend warmups/capture. CPU controls exercise
the real SGLang eligibility and FlashInfer cache paths through this adapter;
they do not qualify numerical correctness, CUDA capture or a production policy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class GraphSpec:
    root: Path
    policy: Any
    skip_ops: frozenset[str]


def require_coverage(report, expected_skips):
    """Configured exclusions are skips, never counted as cache hits."""
    if (
        report.misses
        or report.dropped_records
        or not report.records
        or report.hits + report.skipped != len(report.records)
        or report.hits != sum(row["status"] == "hit" for row in report.records)
        or report.skipped != sum(row["status"] == "skipped" for row in report.records)
    ):
        raise RuntimeError("Graph replay lacks complete lookup coverage")
    for row in report.records:
        expected = "skipped" if row["op"] in expected_skips else "hit"
        if row["status"] != expected or (
            expected == "skipped" and row["selected_tactic"] != "-1"
        ):
            raise RuntimeError("Graph replay did not preserve the skip policy")


class ScopedGraphRegion:
    """One immutable graph identity, separate from legacy startup persistence."""

    def __init__(self, spec):
        self.spec = spec
        self.complete = False
        self.report = None

    @classmethod
    def create(cls, model_runner, graph_base_root):
        """Resolve after startup; the caller supplies a dense full-decode region.

        ``graph_base_root`` must already distinguish deployments and skip
        policies and must be separate from the legacy JSON path. Reuse-off
        gives this region its own managed identity and isolated output, while
        retaining legacy memory and reuse within the region.
        """
        from flashinfer.autotune_cache import MeasurementPolicy

        from sglang.srt.environ import envs
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            get_flashinfer_autotune_skip_ops,
            should_run_flashinfer_autotune,
        )
        from sglang.srt.runtime_context import get_exec, get_parallel

        graph = get_exec().graph
        if (
            graph.disable_cuda_graph
            or not model_runner.is_generation
            or model_runner.is_draft_worker
            or model_runner.spec_algorithm.is_speculative()
            or not should_run_flashinfer_autotune(model_runner)
        ):
            return None
        if (
            graph.cuda_graph_config is None
            or graph.cuda_graph_config.decode.backend != Backend.FULL
        ):
            return None
        parallel = get_parallel()
        if any(
            getattr(parallel, name) != 1
            for name in (
                "tp_size",
                "pp_size",
                "attn_dp_size",
                "attn_cp_size",
                "moe_ep_size",
                "moe_dp_size",
            )
        ):
            return None
        root = Path(graph_base_root).resolve()
        if not envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
            root = root.parent / "runs" / f"{root.name}.{uuid4().hex}"
        return cls(
            GraphSpec(
                root=root,
                policy=MeasurementPolicy(execution_mode="cuda_graph"),
                skip_ops=frozenset(get_flashinfer_autotune_skip_ops(model_runner)),
            )
        )

    def context(self, mode, *, report=None):
        from flashinfer import autotune_v2

        return autotune_v2(
            mode=mode,
            persistent_cache=True,
            cache_root=self.spec.root,
            cache_scope="context",
            measurement_policy=self.spec.policy,
            skip_ops=self.spec.skip_ops,
            replay_report=report,
            strict_replay=mode == "replay",
        )

    def run(self, prepare_forward, capture_callback, reset, *, is_capturing=None):
        """Tune before capture, then cover the backend's warmups and recording.

        ``reset`` restores the supplied attention hook's state after the extra
        prepared forward, including on failure. It does not promise general
        CUDA-error recovery. Valid publications survive a later capture error.
        """
        from flashinfer.autotune_cache import ReplayCoverage
        from flashinfer.autotuner import AutoTuner

        if is_capturing is None:
            import torch

            is_capturing = torch.cuda.is_current_stream_capturing
        self.complete = False
        self.report = ReplayCoverage(max_records=8192)
        if is_capturing():
            raise RuntimeError("Graph preparation must precede CUDA capture")
        # Legitimate enclosing legacy skip scopes also apply to both contexts.
        expected_skips = AutoTuner.get()._effective_skip_ops | self.spec.skip_ops
        try:
            with self.context("tune"):
                prepare_forward()
        finally:
            reset()
        with self.context("replay", report=self.report):
            result = capture_callback()
        require_coverage(self.report, expected_skips)
        self.complete = True
        return result
