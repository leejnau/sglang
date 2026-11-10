"""Execute the existing registered test unchanged and retain its outcome."""

import hashlib
import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

out = Path(os.environ["SCOPED_QUAL_OUTPUT"])
source = Path(
    "/study/sglang-reviewed/test/registered/e2e/quantization/test_modelopt_fp8.py"
)
spec = importlib.util.spec_from_file_location("registered_modelopt_fp8", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
start = time.time()
suite = unittest.TestSuite([module.TestModeloptFP8("test_gsm8k")])
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {
    "test": "TestModeloptFP8.test_gsm8k",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "started": start,
    "finished": time.time(),
    "tests_run": result.testsRun,
    "failures": [(str(t), s) for t, s in result.failures],
    "errors": [(str(t), s) for t, s in result.errors],
    "skipped": result.skipped,
    "successful": result.wasSuccessful(),
    "opt_in": os.environ["SGLANG_ENABLE_FLASHINFER_GRAPH_AUTOTUNE"],
}
(out / "test-result.json").write_text(json.dumps(record, indent=2) + "\n")
raise SystemExit(0 if result.wasSuccessful() else 1)
