"""Check that legacy and scoped preparation each receive a fresh canary scope."""

import argparse
import ast
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner import base_runner


def check():
    events = []

    @contextlib.contextmanager
    def canary(index):
        events.append(("enter", index))
        try:
            yield
        finally:
            events.append(("exit", index))

    def dummy(*, run_ctx, **kwargs):
        with run_ctx:
            events.append(("forward", 0))

    def prepare(mr, forward, **kwargs):
        assert kwargs["prepare_decode_graph"]
        forward()
        forward()

    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            canary_manager=SimpleNamespace(with_active_single_forward_manager=canary)
        ),
        _dummy_run=dummy,
    )
    with patch.object(base_runner, "run_flashinfer_autotune_forward", prepare):
        base_runner.BaseRunner._flashinfer_autotune(
            runner, buffers=object(), batch_size=1
        )
    assert events == [("enter", 0), ("forward", 0), ("exit", 0)] * 2, events
    return {"status": "passed", "canary_events": events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path)
    args = parser.parse_args()
    result = check()
    if args.before:
        source = ast.parse(args.before.read_text())
        cls = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "BaseRunner"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_flashinfer_autotune"
        )
        namespace = dict(base_runner.__dict__)
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), str(args.before), "exec"
            ),
            base_runner.__dict__,
            namespace,
        )
        with patch.object(
            base_runner.BaseRunner, "_flashinfer_autotune", namespace[method.name]
        ):
            try:
                check()
            except AttributeError as error:
                assert "_GeneratorContextManager" in str(error)
                result["before_control"] = {
                    "status": "expected_failure",
                    "error": str(error),
                }
            else:
                raise AssertionError("pre-fix canary control did not fail")
    print(json.dumps(result, indent=2))
