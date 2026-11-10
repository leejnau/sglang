"""Mutation controls for the production-scope invariants; never used for serving."""

import hashlib
import inspect
import json
import sys

import control_tests

case = sys.argv[1]
sg = control_tests.sg
source = inspect.getsource(sg.FlashInferGraphAutotune)
changes = {
    "policy": (
        'cache_scope="context",',
        'cache_scope="context", measurement_policy=__import__("flashinfer.autotune_cache", fromlist=["MeasurementPolicy"]).MeasurementPolicy(execution_mode="cuda_graph"),',
    ),
    "scope": ('cache_scope="context",', 'cache_scope="process",'),
    "skip": ("skip_ops=self.skip_ops,", "skip_ops=(),"),
    "reuse": ("and envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get()", "and True"),
}
a, b = changes[case]
assert source.count(a) == 1
mutated = source.replace(a, b)
exec(compile(mutated, "<negative-" + case + ">", "exec"), sg.__dict__)
try:
    control_tests.run()
except (AssertionError, RuntimeError, ValueError) as error:
    expected = {
        "policy": "measurement policy unchanged",
        "scope": "strict_replay requires an identified context-scoped cache",
        "skip": "skip overrides populated graph winner",
        "reuse": "reuse-off retains the v1 path",
    }
    assert expected[case] in str(error), str(error)
    print(
        json.dumps(
            {
                "case": case,
                "status": "expected_failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "mutated_sha256": hashlib.sha256(mutated.encode()).hexdigest(),
            },
            indent=2,
        )
    )
else:
    raise AssertionError("negative control passed unexpectedly")
