"""Isolated CPU imports for scoped-control research, never a serving bootstrap.

The SGLang startup-tuning helper, runtime/configuration modules, and FlashInfer
tuner/cache modules run unchanged. GPU execution, model loading, and unrelated import side effects are
explicit boundaries. Run once in a fresh process; no kernel or GPU is available.
"""

import contextlib
import importlib
import logging
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def _bootstrap(sglang_root, flashinfer_root):
    import torch

    if any(name in sys.modules for name in ("sglang", "flashinfer")):
        raise RuntimeError("bootstrap requires a fresh interpreter")
    if torch.cuda.is_available():
        raise RuntimeError("this qualification must run without CUDA")
    source = Path(sglang_root) / "python"
    fi = Path(flashinfer_root)
    boundaries = []
    calls = []

    def unavailable(*args, **kwargs):
        raise AssertionError("CPU research reached an unqualified import/GPU boundary")

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        parent, _, leaf = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, mod)
        boundaries.append(name)
        return mod

    def package(name, root):
        return module(name, __path__=[str(root / name.replace(".", "/"))])

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.hardware_backend",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.quantization",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.runner",
        "sglang.srt.speculative",
        "sglang.srt.configs",
        "sglang.srt.arg_groups.model_overrides",
    ):
        package(name, source)

    # Unused utility imports fail closed. Constants below are only parser
    # metadata; none of the cases qualifies LoRA, networking, or model loading.
    utils = sys.modules["sglang.srt.utils"]
    utils.empty_context = contextlib.nullcontext
    utils.log_info_on_rank0 = lambda logger, message: logger.info(message)
    utils.ConcurrentCounter = unavailable
    common = module(
        "sglang.srt.utils.common",
        is_mps=lambda: False,
        torch_release=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")),
        LORA_TARGET_ALL_MODULES="all",
        SUPPORTED_LORA_TARGET_MODULES=[],
    )
    for name in (
        "is_no_spec_infer_or_topk_one",
        "get_quantization_config",
        "is_fi_a2a_supported",
        "is_gfx95_supported",
        "xpu_has_xmx_support",
        "json_list_type",
        "human_readable_int",
        "nullable_str",
        "get_device_memory_capacity",
        "parse_connector_type",
        "ceil_align",
        "get_device",
        "is_mnnvl_fabric_device",
        "is_host_cpu_arm64",
        "get_eager_max_batch_size",
    ):
        setattr(common, name, unavailable)

    def media_defaults(domains, limit):
        if domains not in (None, []):
            raise AssertionError("only default media configuration is qualified")
        return domains

    common.configure_media_url_security = media_defaults
    module(
        "sglang.srt.utils.network",
        NetworkAddress=type("NetworkAddress", (), {}),
        get_free_port_below_ephemeral=unavailable,
        wait_port_available=unavailable,
    )
    module(
        "sglang.srt.distributed.device_communicators.mooncake_transfer_engine",
        parse_ib_device_config=unavailable,
    )
    module("sglang.srt.configs.model_config", is_deepseek_dsa=unavailable)
    module("sglang.srt.configs.hybrid_arch", mambaish_config=unavailable)
    module(
        "sglang.srt.connector",
        ConnectorType=types.SimpleNamespace(
            FS="filesystem", KV="KV", INSTANCE="instance"
        ),
    )
    module("sglang.utils", is_in_ci=lambda: False)
    module("sglang.srt.model_executor.forward_batch_info", ForwardMode=unavailable)

    @contextlib.contextmanager
    def dummy_mode(*, run_lm_head):
        calls.append({"boundary": "dummy_mode", "run_lm_head": run_lm_head})
        yield

    module("sglang.srt.layers.logits_processor", autotune_dummy_run_mode=dummy_mode)
    module(
        "sglang.srt.layers.moe.flashinfer_megamoe_autotune",
        megamoe_autotune_context=unavailable,
    )
    module(
        "sglang.srt.layers.quantization.fp4_utils",
        get_fp4_gemm_runner_backend=unavailable,
    )
    module(
        "sglang.srt.layers.quantization.fp8_utils",
        flashinfer_per_tensor_fp8_supported=lambda: True,
        resolve_mxfp8_dense_gemm_backend=unavailable,
    )

    package("flashinfer", fi)
    package("flashinfer.jit", fi)
    module("flashinfer.jit.core", logger=logging.getLogger("scoped-control-cpu"))
    module(
        "flashinfer.utils",
        next_positive_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        is_confidential_compute=lambda: False,
        get_globaltimer_kernel=unavailable,
    )
    module(
        "flashinfer.tllm_utils",
        delay_kernel=unavailable,
        get_trtllm_utils_module=unavailable,
    )
    api = importlib.import_module("flashinfer.autotuner")
    engine = importlib.import_module("flashinfer.autotuner.autotuner")
    cache = importlib.import_module("flashinfer.autotune_cache")
    sys.modules["flashinfer"].autotune_v2 = cache.autotune_v2
    sys.modules["flashinfer"].__version__ = importlib.import_module(
        "flashinfer.version"
    ).__version__
    runtime = importlib.import_module("sglang.srt.runtime_context")
    environ = importlib.import_module("sglang.srt.environ").envs
    sg = importlib.import_module("sglang.srt.model_executor.runner.flashinfer_autotune")
    graph = importlib.import_module("sglang.srt.model_executor.cuda_graph_config")

    class Stream:
        def wait_stream(self, other):
            if not isinstance(other, Stream):
                raise AssertionError("unexpected stream outside CPU fixture")
            calls.append({"boundary": "stream_wait"})

    stream = Stream()

    def runner():
        return types.SimpleNamespace(
            device="cuda",
            gpu_id=0,
            dtype=torch.bfloat16,
            is_draft_worker=False,
            is_generation=True,
            forward_stream=Stream(),
            spec_algorithm=types.SimpleNamespace(is_speculative=lambda: False),
            model_config=types.SimpleNamespace(
                quantization="modelopt",
                hf_config=types.SimpleNamespace(),
            ),
        )

    metadata = {"qualification": "cpu-scoped-control", "device": "synthetic-sm90"}
    with contextlib.ExitStack() as stack:
        for owner, name, replacement in (
            (engine, "_collect_metadata", lambda: dict(metadata)),
            (api, "_collect_metadata", lambda: dict(metadata)),
            (torch.cuda, "is_current_stream_capturing", lambda: False),
            (torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0)),
            (torch.cuda, "current_stream", lambda *args, **kwargs: stream),
            (torch.cuda, "stream", lambda value: contextlib.nullcontext()),
        ):
            stack.enter_context(patch.object(owner, name, replacement))
        stack.enter_context(
            runtime.get_context().override_server_args(
                model_path="dummy",
                quantization="modelopt",
                device="cuda",
                moe_runner_backend="auto",
                cuda_graph_config=graph.CudaGraphConfig(
                    decode=graph.PhaseConfig(backend=graph.Backend.FULL),
                    prefill=graph.PhaseConfig(backend=graph.Backend.DISABLED),
                ),
            )
        )
        stack.enter_context(
            runtime.get_parallel().override(
                tp_rank=0,
                pp_rank=0,
                dp_rank=0,
                tp_group=types.SimpleNamespace(world_size=1),
            )
        )
        yield types.SimpleNamespace(
            torch=torch,
            api=api,
            engine=engine,
            cache=cache,
            sg=sg,
            runtime=runtime,
            environ=environ,
            model_runner=runner,
            graph=graph,
            boundaries=boundaries,
            boundary_calls=calls,
            controls={
                "torch": "real CPU tensors; no CUDA execution",
                "metadata": "synthetic fixed manifest in both API and tuner module",
                "capability": "synthetic sm90; per-tensor FP8 support true",
                "streams": "recorded waits and no-op stream contexts",
                "dummy_mode": "recorded entry only; no model execution",
                "runtime": "real dummy ServerArgs resolution and namespace publication",
                "imports": "explicit namespace/peripheral shims; unused calls fail closed",
                "media_defaults": "empty domain policy passthrough for dummy resolution",
                "environment": "timer cuda_event; bundled file reuse disabled; blocklist unset",
            },
            metadata=metadata,
        )


@contextlib.contextmanager
def bootstrap(sglang_root, flashinfer_root):
    """Install documented boundaries for one fresh, isolated CPU child."""
    environment = dict(os.environ)
    environment.pop("FLASHINFER_TACTICS_BLOCKLIST", None)
    environment.pop("FLASHINFER_AUTOTUNE_CACHE_DIR", None)
    environment.update(
        FLASHINFER_AUTOTUNE_TIMER="cuda_event",
        FLASHINFER_AUTOTUNER_LOAD_FROM_FILE="0",
    )
    with patch.dict(os.environ, environment, clear=True):
        with _bootstrap(sglang_root, flashinfer_root) as loaded:
            yield loaded
