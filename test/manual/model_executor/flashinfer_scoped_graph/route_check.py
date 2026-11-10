"""Count real graph replays in the padded numerical cases; no timing claims."""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch


def main():
    assert "--no-perf" in sys.argv
    os.environ["SGLANG_CACHE_DIR"] = sys.argv[sys.argv.index("--cache") + 1]
    os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"] = "1"
    os.environ["SGLANG_FLASHINFER_AUTOTUNE_CACHE"] = "1"
    import model_study

    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    counts = Counter()
    original = FullCudaGraphBackend.replay

    def replay(backend, key, batch, *args, **kwargs):
        counts[batch.batch_size] += 1
        return original(backend, key, batch, *args, **kwargs)

    with patch.object(FullCudaGraphBackend, "replay", replay):
        model_study.main()
    assert counts == {1: 6, 3: 6, 4: 6, 7: 6, 16: 6}, counts
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(output.read_text())
    result["actual_backend_replays_by_request_batch"] = dict(counts)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
