"""Research-only check of preparing the managed cache at startup warmup."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    os.environ["SGLANG_CACHE_DIR"] = sys.argv[sys.argv.index("--cache") + 1]
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = "1"
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    import model_study
    import torch
    from flashinfer import autotune_v2

    from sglang.srt.layers.logits_processor import autotune_dummy_run_mode
    from sglang.srt.model_executor.runner import base_runner
    from sglang.srt.model_executor.runner import flashinfer_autotune as sg

    original = sg.run_flashinfer_autotune_forward
    preparations = []

    def prepare(mr, forward, *, run_lm_head):
        original(mr, forward, run_lm_head=run_lm_head)
        root = sg.flashinfer_autotune_cache_path(mr).with_suffix(".decode_graph")
        previous = torch.cuda.current_stream()
        mr.forward_stream.wait_stream(previous)
        try:
            with (
                torch.cuda.stream(mr.forward_stream),
                autotune_dummy_run_mode(run_lm_head=run_lm_head),
                autotune_v2(
                    mode="tune",
                    cache_scope="context",
                    cache_root=root,
                    skip_ops=sg.get_flashinfer_autotune_skip_ops(mr),
                ),
            ):
                forward()
        finally:
            previous.wait_stream(mr.forward_stream)
        preparations.append(str(root))

    with patch.object(base_runner, "run_flashinfer_autotune_forward", prepare):
        model_study.main()
    assert len(preparations) == 1, preparations
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(output.read_text())
    result["warmup_preparation_probe"] = preparations
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
