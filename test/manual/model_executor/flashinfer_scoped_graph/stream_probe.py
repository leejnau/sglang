"""Research-only probe of profiling context; not part of the proposed change."""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--switch-stream", action="store_true")
    options, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    os.environ["SGLANG_CACHE_DIR"] = remaining[remaining.index("--cache") + 1]
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = "1"
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    import model_study
    import torch
    from flashinfer.autotuner import AutoTuner

    from sglang.srt.model_executor.runner.flashinfer_autotune import (
        FlashInferGraphAutotune,
    )

    sys.argv = [sys.argv[0], *remaining]
    records = []
    original_profile = AutoTuner._profile_single_kernel
    original_create = FlashInferGraphAutotune.create.__func__
    original_context = FlashInferGraphAutotune.capture_context

    def profile(tuner, runner, inputs, tactic, config, **kwargs):
        records.append(
            {
                "identity": repr(tuner._winner_cache_identity()),
                "stream": torch.cuda.current_stream().cuda_stream,
                "tactic": repr(tactic),
                "config": {
                    k: v
                    for k, v in vars(config).items()
                    if isinstance(v, (bool, int, float, str)) or v is None
                },
                "inputs": [
                    {
                        "shape": list(x.shape),
                        "stride": list(x.stride()),
                        "dtype": str(x.dtype),
                        "pointer_mod4096": x.data_ptr() % 4096,
                    }
                    for x in inputs
                    if isinstance(x, torch.Tensor)
                ],
            }
        )
        return original_profile(tuner, runner, inputs, tactic, config, **kwargs)

    def create(cls, runner):
        scope = original_create(cls, runner)
        if scope is not None:
            scope._probe_stream = runner.model_runner.forward_stream
        return scope

    @contextlib.contextmanager
    def capture(scope, forward, reset):
        def prepared():
            if not options.switch_stream:
                return forward()
            previous = torch.cuda.current_stream()
            scope._probe_stream.wait_stream(previous)
            try:
                with torch.cuda.stream(scope._probe_stream):
                    return forward()
            finally:
                previous.wait_stream(scope._probe_stream)

        with original_context(scope, prepared, reset):
            yield

    try:
        with (
            patch.object(AutoTuner, "_profile_single_kernel", profile),
            patch.object(FlashInferGraphAutotune, "create", classmethod(create)),
            patch.object(FlashInferGraphAutotune, "capture_context", capture),
        ):
            model_study.main()
    finally:
        if output.exists():
            result = json.loads(output.read_text())
            result["stream_probe"] = {
                "switch_stream": options.switch_stream,
                "records": records,
            }
            output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
