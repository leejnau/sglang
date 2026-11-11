"""Check the merged startup helper's input and canary lifetime across forwards."""

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner import base_runner
from sglang.srt.model_executor.runner import flashinfer_autotune as sg


def check(fail_on=None):
    events, seen = [], []
    original = torch.arange(32, device="cuda", dtype=torch.int64)
    buffers = SimpleNamespace(input_ids=original.clone())
    pointer = buffers.input_ids.data_ptr()

    @contextlib.contextmanager
    def canary(index):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def dummy(*, run_ctx, **kwargs):
        with run_ctx:
            seen.append(buffers.input_ids.clone())
            events.append("forward")
            if len(seen) == fail_on:
                raise RuntimeError("injected forward failure")

    def prepare(mr, forward, **kwargs):
        assert kwargs["prepare_decode_graph"]
        for _ in range(2):
            forward()
            assert torch.equal(buffers.input_ids, original)

    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            canary_manager=SimpleNamespace(with_active_single_forward_manager=canary),
            model_config=SimpleNamespace(vocab_size=4096),
        ),
        _dummy_run=dummy,
    )
    rng = torch.get_rng_state().clone()
    with (
        patch.object(base_runner, "run_flashinfer_autotune_forward", prepare),
        patch.object(sg, "get_parallel", return_value=SimpleNamespace(attn_dp_rank=0)),
        patch.object(sg, "log_info_on_rank0"),
    ):
        try:
            base_runner.BaseRunner._flashinfer_autotune(
                runner, buffers=buffers, batch_size=32
            )
        except RuntimeError as error:
            assert fail_on is not None and str(error) == "injected forward failure"
        else:
            assert fail_on is None
    assert len(seen) == (fail_on or 2)
    assert events == ["enter", "forward", "exit"] * len(seen)
    assert torch.equal(buffers.input_ids, original)
    assert buffers.input_ids.data_ptr() == pointer
    assert torch.equal(torch.get_rng_state(), rng)
    for tokens in seen:
        assert tokens.unique().numel() > 1
        assert int(tokens.min()) >= 0 and int(tokens.max()) < 4096
        assert not torch.equal(tokens, original)
        assert torch.equal(tokens, seen[0])
    return {"fail_on": fail_on, "forwards": len(seen), "events": events}


if __name__ == "__main__":
    results = [check(), check(1), check(2)]
    # Negative control: accidentally hoisting one context across both forwards
    # must fail on the second entry, rather than silently passing this probe.
    create = sg.autotune_dummy_run_ctx
    held = []

    def hoisted(mr, buffers):
        if not held:
            held.append(create(mr, buffers))
        return held[0]

    with patch.object(base_runner, "autotune_dummy_run_ctx", hoisted):
        try:
            check()
        except AttributeError as error:
            assert "_GeneratorContextManager" in str(error)
        else:
            raise AssertionError("hoisted-context negative control unexpectedly passed")
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": results,
                "negative_control": "expected_failure",
            },
            indent=2,
        )
    )
