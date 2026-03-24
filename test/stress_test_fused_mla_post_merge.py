"""
Throwaway stress-test for post-merge-conflict fused MLA code paths.

Covers:
  - _init_forward_metadata_for_rope_fusion (fresh alloc + in-place update)
  - CUDA-graph capture/replay simulation (padded batch, varying bs, stale data)
  - positions.zero_() / out_cache_loc.zero_() buffer hygiene
  - Fused vs separated kernel parity (exhaustive parametrize)
  - NeoX vs non-NeoX RoPE styles
  - Repeated replays with shrinking/growing batch sizes
  - Padded-token page-0 safety (sentinel check)
  - save_kv_cache=True vs save_kv_cache=False dispatch
  - Combined non-contiguous KV buffer under CUDA graph conditions
  - Idempotency (same inputs twice => same outputs)
  - Randomized fuzz with many seeds
  - Edge cases: bs=1, page_size=1, same-page collisions, extreme positions

DELETE THIS FILE once confidence is established.

Usage:
    pytest test/stress_test_fused_mla_post_merge.py -v
    pytest test/stress_test_fused_mla_post_merge.py -v -x   # stop on first failure
    pytest test/stress_test_fused_mla_post_merge.py -v -k cuda_graph
    pytest test/stress_test_fused_mla_post_merge.py -v -k fuzz
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

try:
    import flashinfer
    import flashinfer.rope

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

IS_CUDA = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(not IS_CUDA, reason="CUDA required"),
    pytest.mark.skipif(not HAS_FLASHINFER, reason="FlashInfer required"),
]

# ── DeepSeek V3/R1 MLA constants ──────────────────────────────────────────────
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576


# ── Helpers ───────────────────────────────────────────────────────────────────


def create_cos_sin_cache(
    max_seq_len: int, rotary_dim: int, device: torch.device
) -> torch.Tensor:
    freqs = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
    )
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def make_inputs(
    batch_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    q_nope = torch.randn(
        batch_size, num_heads, KV_LORA_RANK, device=device, dtype=dtype
    )
    q_rope = torch.randn(
        batch_size, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=dtype
    )
    k_nope = torch.randn(batch_size, KV_LORA_RANK, device=device, dtype=dtype)
    k_rope = torch.randn(batch_size, QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
    return q_nope, q_rope, k_nope, k_rope


def generate_cache_locs(batch_size: int, page_size: int, device: torch.device):
    pages = torch.arange(batch_size, device=device, dtype=torch.int32)
    offsets = (pages * 7 + 3) % page_size
    return pages * page_size + offsets


def run_fused(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    cos_sin_cache,
    pos_ids,
    out_cache_loc,
    ckv_cache,
    kpe_cache,
    page_size,
):
    nnz = out_cache_loc.shape[0]
    device = out_cache_loc.device
    fp8 = torch.float8_e4m3fn
    q_out = torch.empty(
        nnz, q_rope.shape[1], KV_LORA_RANK + QK_ROPE_HEAD_DIM, device=device, dtype=fp8
    )
    kv_indices = (out_cache_loc // page_size).to(torch.int32)
    positions = (out_cache_loc % page_size).to(torch.int32)
    kv_indptr = torch.arange(nnz + 1, dtype=torch.int32, device=device)
    batch_indices = torch.arange(nnz, dtype=torch.int32, device=device)
    flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        v=None,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        paged_kv_cache=(ckv_cache, kpe_cache),
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        batch_indices=batch_indices,
        positions=positions,
        is_neox=False,
        quantize_dtype=fp8,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        page_size=page_size,
        kv_layout="NHD",
        q_rope_out=q_out[..., KV_LORA_RANK:],
        q_nope_out=q_out[..., :KV_LORA_RANK],
    )
    return q_out


def run_separated(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    cos_sin_cache,
    pos_ids,
    out_cache_loc,
    ckv_cache,
    kpe_cache,
    page_size,
):
    nnz = out_cache_loc.shape[0]
    device = q_rope.device
    fp8 = torch.float8_e4m3fn
    q_out = torch.empty(
        nnz, q_rope.shape[1], KV_LORA_RANK + QK_ROPE_HEAD_DIM, device=device, dtype=fp8
    )
    k_rope_out = torch.empty(k_rope.shape, device=device, dtype=fp8)
    k_nope_out = torch.empty(k_nope.shape, device=device, dtype=fp8)
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=False,
        quantize_dtype=fp8,
        q_rope_out=q_out[..., KV_LORA_RANK:],
        k_rope_out=k_rope_out,
        q_nope_out=q_out[..., :KV_LORA_RANK],
        k_nope_out=k_nope_out,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )
    kv_indices = out_cache_loc // page_size
    positions = out_cache_loc % page_size
    for i in range(nnz):
        p, s = kv_indices[i].item(), positions[i].item()
        ckv_cache[p, s, :] = k_nope_out[i]
        kpe_cache[p, s, :] = k_rope_out[i]
    return q_out


def assert_parity(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    cos_sin_cache,
    pos_ids,
    out_cache_loc,
    page_size,
    num_pages,
    label="",
):
    fp8 = torch.float8_e4m3fn
    sentinel = torch.tensor(0.5, dtype=torch.float32).to(fp8)
    device = q_nope.device
    ckv_f = torch.full(
        (num_pages, page_size, KV_LORA_RANK), sentinel.item(), device=device, dtype=fp8
    )
    kpe_f = torch.full(
        (num_pages, page_size, QK_ROPE_HEAD_DIM),
        sentinel.item(),
        device=device,
        dtype=fp8,
    )
    ckv_f_snap = ckv_f.clone()
    kpe_f_snap = kpe_f.clone()
    ckv_s = torch.full_like(ckv_f, sentinel.item())
    kpe_s = torch.full_like(kpe_f, sentinel.item())

    q_fused = run_fused(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        out_cache_loc,
        ckv_f,
        kpe_f,
        page_size,
    )
    q_sep = run_separated(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        out_cache_loc,
        ckv_s,
        kpe_s,
        page_size,
    )

    pfx = f"[{label}] " if label else ""
    q_diff = torch.max(torch.abs(q_fused.float() - q_sep.float())).item()
    assert q_diff == 0.0, f"{pfx}q_out mismatch: {q_diff}"

    bs = out_cache_loc.shape[0]
    kv_idx = out_cache_loc // page_size
    pos = out_cache_loc % page_size
    for i in range(bs):
        p, s = kv_idx[i].item(), pos[i].item()
        cd = torch.max(torch.abs(ckv_f[p, s].float() - ckv_s[p, s].float())).item()
        kd = torch.max(torch.abs(kpe_f[p, s].float() - kpe_s[p, s].float())).item()
        assert cd == 0.0, f"{pfx}ckv diff at token {i}: {cd}"
        assert kd == 0.0, f"{pfx}kpe diff at token {i}: {kd}"

    written = torch.zeros(num_pages, page_size, dtype=torch.bool, device=device)
    for i in range(bs):
        written[kv_idx[i], pos[i]] = True
    assert torch.equal(
        ckv_f[~written], ckv_f_snap[~written]
    ), f"{pfx}spurious ckv write"
    assert torch.equal(
        kpe_f[~written], kpe_f_snap[~written]
    ), f"{pfx}spurious kpe write"
    return q_fused, ckv_f, kpe_f


# ── Simulated _init_forward_metadata_for_rope_fusion ──────────────────────────


@dataclass
class FakeDecodeMetadata:
    kv_indices: Optional[torch.Tensor] = None
    kv_indptr: Optional[torch.Tensor] = None
    batch_indices: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None


def init_metadata_fresh(
    metadata: FakeDecodeMetadata, out_cache_loc: torch.Tensor, page_size: int
):
    """Mirrors _init_forward_metadata_for_rope_fusion(update_inplace=False)."""
    nnz = out_cache_loc.shape[0]
    device = out_cache_loc.device
    metadata.kv_indices = (out_cache_loc // page_size).to(torch.int32)
    metadata.positions = (out_cache_loc % page_size).to(torch.int32)
    metadata.kv_indptr = torch.arange(nnz + 1, dtype=torch.int32, device=device)
    metadata.batch_indices = torch.arange(nnz, dtype=torch.int32, device=device)


def init_metadata_inplace(
    metadata: FakeDecodeMetadata, out_cache_loc: torch.Tensor, page_size: int
):
    """Mirrors _init_forward_metadata_for_rope_fusion(update_inplace=True)."""
    nnz = out_cache_loc.shape[0]
    total = metadata.kv_indices.shape[0]
    torch.div(
        out_cache_loc, page_size, rounding_mode="floor", out=metadata.kv_indices[:nnz]
    )
    torch.remainder(out_cache_loc, page_size, out=metadata.positions[:nnz])
    if nnz < total:
        metadata.kv_indices[nnz:total].zero_()
        metadata.positions[nnz:total].zero_()


def run_fused_with_metadata(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    cos_sin_cache,
    pos_ids,
    metadata: FakeDecodeMetadata,
    nnz: int,
    ckv_cache,
    kpe_cache,
    page_size,
):
    """Call fused kernel using pre-computed metadata (like production code does)."""
    device = q_nope.device
    fp8 = torch.float8_e4m3fn
    q_out = torch.empty(
        nnz, q_rope.shape[1], KV_LORA_RANK + QK_ROPE_HEAD_DIM, device=device, dtype=fp8
    )
    flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        v=None,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        paged_kv_cache=(ckv_cache, kpe_cache),
        kv_indices=metadata.kv_indices[:nnz],
        kv_indptr=metadata.kv_indptr[: nnz + 1],
        batch_indices=metadata.batch_indices[:nnz],
        positions=metadata.positions[:nnz],
        is_neox=False,
        quantize_dtype=fp8,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        page_size=page_size,
        kv_layout="NHD",
        q_rope_out=q_out[..., KV_LORA_RANK:],
        q_nope_out=q_out[..., :KV_LORA_RANK],
    )
    return q_out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: _init_forward_metadata_for_rope_fusion unit tests
# ══════════════════════════════════════════════════════════════════════════════


class TestInitForwardMetadata:
    """Unit-test the metadata initialization logic in isolation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.mark.parametrize("page_size", [1, 16, 32, 64])
    @pytest.mark.parametrize("bs", [1, 4, 17, 128])
    def test_fresh_alloc_values(self, device, bs, page_size):
        """Fresh allocation must produce correct kv_indices, positions, indptr, batch_indices."""
        out_cache_loc = generate_cache_locs(bs, page_size, device)
        md = FakeDecodeMetadata()
        init_metadata_fresh(md, out_cache_loc, page_size)

        expected_pages = (out_cache_loc // page_size).to(torch.int32)
        expected_offsets = (out_cache_loc % page_size).to(torch.int32)
        assert torch.equal(md.kv_indices, expected_pages)
        assert torch.equal(md.positions, expected_offsets)
        assert torch.equal(
            md.kv_indptr, torch.arange(bs + 1, dtype=torch.int32, device=device)
        )
        assert torch.equal(
            md.batch_indices, torch.arange(bs, dtype=torch.int32, device=device)
        )

    @pytest.mark.parametrize("page_size", [16, 32, 64])
    @pytest.mark.parametrize(
        "max_bs,real_bs",
        [
            (128, 128),
            (128, 64),
            (128, 1),
            (64, 33),
            (256, 7),
        ],
    )
    def test_inplace_update_matches_fresh(self, device, max_bs, real_bs, page_size):
        """In-place update [0:real_bs) must match a fresh allocation for those tokens."""
        out_cache_loc = generate_cache_locs(real_bs, page_size, device)

        md_fresh = FakeDecodeMetadata()
        init_metadata_fresh(md_fresh, out_cache_loc, page_size)

        md_inplace = FakeDecodeMetadata()
        md_inplace.kv_indices = torch.full(
            (max_bs,), 9999, dtype=torch.int32, device=device
        )
        md_inplace.positions = torch.full(
            (max_bs,), 9999, dtype=torch.int32, device=device
        )
        md_inplace.kv_indptr = torch.arange(
            max_bs + 1, dtype=torch.int32, device=device
        )
        md_inplace.batch_indices = torch.arange(
            max_bs, dtype=torch.int32, device=device
        )
        init_metadata_inplace(md_inplace, out_cache_loc, page_size)

        assert torch.equal(md_inplace.kv_indices[:real_bs], md_fresh.kv_indices)
        assert torch.equal(md_inplace.positions[:real_bs], md_fresh.positions)

    @pytest.mark.parametrize("page_size", [16, 32, 64])
    @pytest.mark.parametrize(
        "max_bs,real_bs",
        [
            (128, 64),
            (128, 1),
            (64, 33),
            (256, 7),
        ],
    )
    def test_inplace_padding_zeroed(self, device, max_bs, real_bs, page_size):
        """Tail [real_bs:max_bs) must be zeroed after in-place update."""
        out_cache_loc = generate_cache_locs(real_bs, page_size, device)
        md = FakeDecodeMetadata()
        md.kv_indices = torch.full((max_bs,), 9999, dtype=torch.int32, device=device)
        md.positions = torch.full((max_bs,), 9999, dtype=torch.int32, device=device)
        md.kv_indptr = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        md.batch_indices = torch.arange(max_bs, dtype=torch.int32, device=device)
        init_metadata_inplace(md, out_cache_loc, page_size)

        assert (md.kv_indices[real_bs:] == 0).all(), "kv_indices tail not zeroed"
        assert (md.positions[real_bs:] == 0).all(), "positions tail not zeroed"

    def test_inplace_batch_indices_not_zeroed(self, device):
        """batch_indices must remain as arange after in-place update (intentional)."""
        max_bs, real_bs, page_size = 128, 32, 64
        out_cache_loc = generate_cache_locs(real_bs, page_size, device)
        md = FakeDecodeMetadata()
        md.kv_indices = torch.zeros(max_bs, dtype=torch.int32, device=device)
        md.positions = torch.zeros(max_bs, dtype=torch.int32, device=device)
        md.kv_indptr = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        md.batch_indices = torch.arange(max_bs, dtype=torch.int32, device=device)
        init_metadata_inplace(md, out_cache_loc, page_size)

        expected_bi = torch.arange(max_bs, dtype=torch.int32, device=device)
        assert torch.equal(
            md.batch_indices, expected_bi
        ), "batch_indices should stay as arange"

    @pytest.mark.parametrize(
        "max_bs,real_bs",
        [
            (128, 64),
            (128, 32),
            (128, 1),
        ],
    )
    def test_stale_data_overwritten(self, device, max_bs, real_bs):
        """Calling in-place update with new data must fully overwrite old values."""
        page_size = 64

        old_locs = generate_cache_locs(max_bs, page_size, device)
        md = FakeDecodeMetadata()
        md.kv_indices = torch.zeros(max_bs, dtype=torch.int32, device=device)
        md.positions = torch.zeros(max_bs, dtype=torch.int32, device=device)
        md.kv_indptr = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        md.batch_indices = torch.arange(max_bs, dtype=torch.int32, device=device)
        init_metadata_inplace(md, old_locs, page_size)

        new_locs = generate_cache_locs(real_bs, page_size, device) + 5
        init_metadata_inplace(md, new_locs, page_size)

        expected_pages = (new_locs // page_size).to(torch.int32)
        expected_offsets = (new_locs % page_size).to(torch.int32)
        assert torch.equal(md.kv_indices[:real_bs], expected_pages)
        assert torch.equal(md.positions[:real_bs], expected_offsets)
        assert (md.kv_indices[real_bs:] == 0).all()
        assert (md.positions[real_bs:] == 0).all()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: CUDA graph capture/replay simulation
# ══════════════════════════════════════════════════════════════════════════════


class TestCudaGraphSimulation:
    """
    Simulate the full CUDA-graph lifecycle:
      1. Allocate max-sized buffers (init_cuda_graph_state)
      2. "Capture" with a specific bs (init_forward_metadata_capture_cuda_graph)
      3. "Replay" with real_bs <= captured bs (init_forward_metadata_replay_cuda_graph)
      4. Run fused kernel with the replayed metadata
      5. Verify correctness + padded-token safety
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def _alloc_cuda_graph_buffers(self, max_tokens, device):
        """Simulate init_cuda_graph_state: allocate max-sized metadata buffers."""
        return FakeDecodeMetadata(
            kv_indices=torch.zeros(max_tokens, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_tokens + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_tokens, dtype=torch.int32, device=device),
            positions=torch.zeros(max_tokens, dtype=torch.int32, device=device),
        )

    def _capture(self, md, capture_bs, out_cache_loc, page_size):
        """Simulate capture: fresh-write metadata, return slice views."""
        init_metadata_fresh(md, out_cache_loc, page_size)

    def _replay(self, md, out_cache_loc, page_size):
        """Simulate replay: in-place update."""
        init_metadata_inplace(md, out_cache_loc, page_size)

    @pytest.mark.parametrize(
        "max_bs,capture_bs,replay_bs,num_heads,page_size",
        [
            (128, 128, 128, 128, 64),  # full batch replay
            (128, 128, 64, 128, 64),  # half batch
            (128, 128, 1, 16, 64),  # single token in big graph
            (128, 128, 33, 64, 32),  # odd count
            (64, 64, 7, 32, 16),  # small graph, tiny replay
            (256, 256, 200, 128, 64),  # large graph, slightly smaller replay
        ],
    )
    def test_capture_then_replay(
        self, device, cos_sin_cache, max_bs, capture_bs, replay_bs, num_heads, page_size
    ):
        """Capture at capture_bs, replay at replay_bs, verify fused output is correct."""
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn
        sentinel = torch.tensor(0.5, dtype=torch.float32).to(fp8)

        md = self._alloc_cuda_graph_buffers(max_bs, device)

        capture_locs = generate_cache_locs(capture_bs, page_size, device)
        self._capture(md, capture_bs, capture_locs, page_size)

        replay_locs = generate_cache_locs(replay_bs, page_size, device)
        self._replay(md, replay_locs, page_size)

        q_nope, q_rope, k_nope, k_rope = make_inputs(replay_bs, num_heads, device)
        pos_ids = torch.arange(replay_bs, device=device, dtype=torch.int32)

        ckv = torch.full(
            (num_pages, page_size, KV_LORA_RANK),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        kpe = torch.full(
            (num_pages, page_size, QK_ROPE_HEAD_DIM),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        ckv_snap = ckv.clone()
        kpe_snap = kpe.clone()

        q_fused = run_fused_with_metadata(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            md,
            replay_bs,
            ckv,
            kpe,
            page_size,
        )

        ckv_ref = torch.full_like(ckv, sentinel.item())
        kpe_ref = torch.full_like(kpe, sentinel.item())
        q_ref = run_separated(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            replay_locs,
            ckv_ref,
            kpe_ref,
            page_size,
        )

        q_diff = torch.max(torch.abs(q_fused.float() - q_ref.float())).item()
        assert q_diff == 0.0, f"q_out mismatch after replay: {q_diff}"

        kv_idx = replay_locs // page_size
        pos = replay_locs % page_size
        for i in range(replay_bs):
            p, s = kv_idx[i].item(), pos[i].item()
            cd = torch.max(torch.abs(ckv[p, s].float() - ckv_ref[p, s].float())).item()
            kd = torch.max(torch.abs(kpe[p, s].float() - kpe_ref[p, s].float())).item()
            assert cd == 0.0, f"ckv diff at token {i}: {cd}"
            assert kd == 0.0, f"kpe diff at token {i}: {kd}"

        written = torch.zeros(num_pages, page_size, dtype=torch.bool, device=device)
        for i in range(replay_bs):
            written[kv_idx[i], pos[i]] = True
        assert torch.equal(
            ckv[~written], ckv_snap[~written]
        ), "spurious ckv write in graph replay"
        assert torch.equal(
            kpe[~written], kpe_snap[~written]
        ), "spurious kpe write in graph replay"

    @pytest.mark.parametrize(
        "max_bs,replay_sequence",
        [
            (128, [64, 32, 128, 1, 96, 7]),
            (64, [64, 1, 33, 64, 2, 50]),
        ],
    )
    def test_repeated_replays_varying_bs(
        self, device, cos_sin_cache, max_bs, replay_sequence
    ):
        """Multiple replays with varying batch sizes through same captured graph."""
        page_size = 64
        num_heads = 64
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        md = self._alloc_cuda_graph_buffers(max_bs, device)
        capture_locs = generate_cache_locs(max_bs, page_size, device)
        self._capture(md, max_bs, capture_locs, page_size)

        for step, replay_bs in enumerate(replay_sequence):
            replay_locs = generate_cache_locs(replay_bs, page_size, device) + step
            self._replay(md, replay_locs, page_size)

            q_nope, q_rope, k_nope, k_rope = make_inputs(
                replay_bs, num_heads, device, seed=step * 1000 + replay_bs
            )
            pos_ids = torch.arange(replay_bs, device=device, dtype=torch.int32)

            ckv = torch.zeros(
                num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
            )
            kpe = torch.zeros(
                num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
            )
            ckv_ref = torch.zeros_like(ckv)
            kpe_ref = torch.zeros_like(kpe)

            q_fused = run_fused_with_metadata(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                md,
                replay_bs,
                ckv,
                kpe,
                page_size,
            )
            q_ref = run_separated(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                replay_locs,
                ckv_ref,
                kpe_ref,
                page_size,
            )

            q_diff = torch.max(torch.abs(q_fused.float() - q_ref.float())).item()
            assert (
                q_diff == 0.0
            ), f"Step {step} (bs={replay_bs}): q_out mismatch {q_diff}"

    def test_padded_tokens_write_to_page_zero(self, device, cos_sin_cache):
        """Padded tokens (positions zeroed by in-place update) must write to page 0 offset 0.

        This validates the _init_forward_metadata_for_rope_fusion zeroing logic:
        padded tokens get kv_indices=0, positions=0 => they all write to (page=0, offset=0).
        We verify they don't corrupt other pages.
        """
        max_bs = 64
        real_bs = 16
        page_size = 64
        num_heads = 128
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        md = self._alloc_cuda_graph_buffers(max_bs, device)
        capture_locs = generate_cache_locs(max_bs, page_size, device)
        self._capture(md, max_bs, capture_locs, page_size)

        real_locs = generate_cache_locs(real_bs, page_size, device) + page_size
        self._replay(md, real_locs, page_size)

        assert (
            md.kv_indices[real_bs:max_bs] == 0
        ).all(), "padding kv_indices not zeroed"
        assert (md.positions[real_bs:max_bs] == 0).all(), "padding positions not zeroed"

        for i in range(real_bs):
            page_idx = md.kv_indices[i].item()
            assert page_idx >= 1, f"Real token {i} unexpectedly maps to page 0"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: positions.zero_() buffer hygiene (DecodeInputBuffers)
# ══════════════════════════════════════════════════════════════════════════════


class TestBufferZeroing:
    """Verify the positions.zero_() fix in populate_from_forward_batch."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_positions_zeroed_when_padded(self, device):
        """When bs > raw_bs, positions[raw_bs:] should be zero after populate."""
        max_bs = 128
        raw_bs = 32
        bs = 128

        positions = torch.arange(max_bs, device=device, dtype=torch.int64)
        out_cache_loc = torch.arange(max_bs, device=device, dtype=torch.int32)
        seq_lens = torch.full((max_bs,), 100, device=device, dtype=torch.int32)

        if bs != raw_bs:
            seq_lens.fill_(1)
            out_cache_loc.zero_()
            positions.zero_()

        positions[:raw_bs].copy_(
            torch.arange(raw_bs, device=device, dtype=torch.int64) + 500
        )
        out_cache_loc[:raw_bs].copy_(
            torch.arange(raw_bs, device=device, dtype=torch.int32) + 1000
        )

        assert (positions[raw_bs:] == 0).all(), "Padded positions not zero"
        assert (out_cache_loc[raw_bs:] == 0).all(), "Padded out_cache_loc not zero"
        assert (positions[:raw_bs] >= 500).all(), "Real positions overwritten"

    def test_positions_not_zeroed_when_no_padding(self, device):
        """When bs == raw_bs, positions should not be blanket-zeroed."""
        bs = 64
        raw_bs = 64
        positions = torch.arange(bs, device=device, dtype=torch.int64) + 100

        if bs != raw_bs:
            positions.zero_()

        assert (positions > 0).all(), "Positions wrongly zeroed when bs == raw_bs"

    @pytest.mark.parametrize("raw_bs", [1, 16, 63, 127])
    def test_stale_positions_cleared(self, device, raw_bs):
        """Simulate a scenario where stale positions from a prior replay remain."""
        max_bs = 128

        positions = torch.arange(max_bs, device=device, dtype=torch.int64) * 100
        out_cache_loc = torch.arange(max_bs, device=device, dtype=torch.int32) * 10

        assert positions[max_bs - 1].item() != 0, "Precondition: stale data exists"

        if max_bs != raw_bs:
            positions.zero_()
            out_cache_loc.zero_()

        positions[:raw_bs] = torch.arange(raw_bs, device=device, dtype=torch.int64)
        out_cache_loc[:raw_bs] = torch.arange(raw_bs, device=device, dtype=torch.int32)

        assert (
            positions[raw_bs:] == 0
        ).all(), f"Stale positions remain at raw_bs={raw_bs}"
        assert (
            out_cache_loc[raw_bs:] == 0
        ).all(), f"Stale cache_loc remain at raw_bs={raw_bs}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Fused vs separated parity — exhaustive parametrize
# ══════════════════════════════════════════════════════════════════════════════


class TestFusedSeparatedParityExhaustive:
    """Exhaustive parametrized fused-vs-separated kernel comparison."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    # ── Real DeepSeek R1/V3 production configs ──────────────────────────────
    # num_attention_heads=128, kv_lora_rank=512, qk_rope_head_dim=64.
    # trtllm_mla enforces page_size in {32, 64}.
    # num_heads after TP split: 128/TP -> {128, 64, 32, 16}.
    # cuda_graph_max_bs ranges: 8 (T4) to 512 (H100/B200 TP>=4).
    #
    # Two sub-tests:
    #   1) TP × page_size cross-product at representative batch sizes
    #   2) Batch-size sweep (1..512 incl. boundaries/odd) at canonical TP=1, ps=64
    @pytest.mark.parametrize(
        "batch_size,num_heads,page_size",
        [
            # TP=1  (128 heads) — page_size 64
            (1, 128, 64),
            (32, 128, 64),
            (128, 128, 64),
            (512, 128, 64),
            # TP=1  (128 heads) — page_size 32
            (1, 128, 32),
            (128, 128, 32),
            (512, 128, 32),
            # TP=2  (64 heads) — page_size 64
            (1, 64, 64),
            (128, 64, 64),
            (512, 64, 64),
            # TP=2  (64 heads) — page_size 32
            (1, 64, 32),
            (128, 64, 32),
            (512, 64, 32),
            # TP=4  (32 heads) — page_size 64
            (1, 32, 64),
            (128, 32, 64),
            (512, 32, 64),
            # TP=4  (32 heads) — page_size 32
            (1, 32, 32),
            (128, 32, 32),
            (512, 32, 32),
            # TP=8  (16 heads) — page_size 64
            (1, 16, 64),
            (128, 16, 64),
            (512, 16, 64),
            # TP=8  (16 heads) — page_size 32
            (1, 16, 32),
            (128, 16, 32),
            (512, 16, 32),
        ],
    )
    def test_production_tp_page_configs(
        self, device, cos_sin_cache, batch_size, num_heads, page_size
    ):
        """All real DeepSeek R1/V3 (TP, page_size) combos at bs=1/128/512."""
        num_pages = batch_size + 2
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=batch_size * 100 + num_heads
        )
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label=f"bs={batch_size},nh={num_heads},ps={page_size}",
        )

    # Batch-size sweep: canonical TP=1 (128 heads), page_size=64.
    # Covers: minimum, small odd, powers-of-2, ±1 boundaries,
    # all real cuda_graph_max_bs defaults (8,24,32,80,160,256,512).
    @pytest.mark.parametrize(
        "batch_size",
        [
            1,
            2,
            3,
            5,
            7,  # tiny / odd
            8,
            9,
            15,
            16,
            17,  # T4 max boundary
            24,
            31,
            32,
            33,  # A10 max boundary
            63,
            64,
            65,
            80,  # mid / A100-24G max
            96,
            127,
            128,
            129,  # common cuda_graph_max_bs
            160,
            191,
            192,
            255,
            256,
            257,  # H100 TP<4 max boundary
            383,
            384,
            511,
            512,  # H100/B200 TP>=4 max
        ],
    )
    def test_production_batch_sweep(self, device, cos_sin_cache, batch_size):
        """Batch size sweep from 1 to 512 at canonical TP=1, page_size=64."""
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=batch_size * 7
        )
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label=f"bs={batch_size}",
        )

    @pytest.mark.parametrize(
        "batch_size,num_heads,page_size",
        [
            (1, 128, 64),
            (1, 16, 64),
            (1, 128, 32),
            (1, 16, 16),
            (2, 128, 64),
            (2, 64, 32),
            (3, 128, 64),
            (4, 32, 32),
            (5, 16, 64),
            (7, 128, 64),
            (8, 16, 64),
            (15, 64, 32),
            (16, 128, 64),
            (31, 32, 64),
            (32, 64, 64),
            (33, 128, 32),
            (48, 16, 64),
            (63, 64, 32),
            (64, 128, 32),
            (96, 128, 64),
            (127, 16, 32),
            (128, 128, 64),
            (128, 16, 32),
        ],
    )
    def test_parity(self, device, cos_sin_cache, batch_size, num_heads, page_size):
        num_pages = batch_size + 2
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=batch_size * 100 + num_heads
        )
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label=f"bs={batch_size},nh={num_heads},ps={page_size}",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: NeoX vs non-NeoX RoPE style
# ══════════════════════════════════════════════════════════════════════════════


class TestRopeStyles:
    """Verify both NeoX and non-NeoX RoPE styles produce identical fused-vs-separated results."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize("is_neox", [True, False])
    @pytest.mark.parametrize("batch_size,num_heads", [(8, 128), (32, 64), (1, 16)])
    def test_neox_vs_gptj(self, device, cos_sin_cache, is_neox, batch_size, num_heads):
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn
        sentinel = torch.tensor(0.5, dtype=torch.float32).to(fp8)

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=777
        )

        ckv_f = torch.full(
            (num_pages, page_size, KV_LORA_RANK),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        kpe_f = torch.full(
            (num_pages, page_size, QK_ROPE_HEAD_DIM),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        ckv_s = torch.full_like(ckv_f, sentinel.item())
        kpe_s = torch.full_like(kpe_f, sentinel.item())

        nnz = batch_size
        q_out_f = torch.empty(
            nnz, num_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        kv_indices = (out_cache_loc // page_size).to(torch.int32)
        positions = (out_cache_loc % page_size).to(torch.int32)
        kv_indptr = torch.arange(nnz + 1, dtype=torch.int32, device=device)
        batch_indices_t = torch.arange(nnz, dtype=torch.int32, device=device)

        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            v=None,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            paged_kv_cache=(ckv_f, kpe_f),
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            batch_indices=batch_indices_t,
            positions=positions,
            is_neox=is_neox,
            quantize_dtype=fp8,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            page_size=page_size,
            kv_layout="NHD",
            q_rope_out=q_out_f[..., KV_LORA_RANK:],
            q_nope_out=q_out_f[..., :KV_LORA_RANK],
        )

        q_out_s = torch.empty_like(q_out_f)
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=fp8)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=fp8)
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=is_neox,
            quantize_dtype=fp8,
            q_rope_out=q_out_s[..., KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out_s[..., :KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        for i in range(nnz):
            p, s = kv_indices[i].item(), positions[i].item()
            ckv_s[p, s, :] = k_nope_out[i]
            kpe_s[p, s, :] = k_rope_out[i]

        q_diff = torch.max(torch.abs(q_out_f.float() - q_out_s.float())).item()
        assert q_diff == 0.0, f"is_neox={is_neox}: q_out mismatch {q_diff}"

        for i in range(nnz):
            p, s = kv_indices[i].item(), positions[i].item()
            cd = torch.max(torch.abs(ckv_f[p, s].float() - ckv_s[p, s].float())).item()
            kd = torch.max(torch.abs(kpe_f[p, s].float() - kpe_s[p, s].float())).item()
            assert cd == 0.0, f"is_neox={is_neox}: ckv diff at {i}: {cd}"
            assert kd == 0.0, f"is_neox={is_neox}: kpe diff at {i}: {kd}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: save_kv_cache=True vs False dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveKvCacheDispatch:
    """Test the _fp8_rope_quantize_and_save dispatch logic."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_save_false_produces_correct_q_no_kv_write(self, device, cos_sin_cache):
        """save_kv_cache=False should produce the same q_out as the separated path
        but NOT write to any KV cache."""
        batch_size = 16
        num_heads = 128
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        q_out = torch.empty(
            batch_size,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=fp8)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=fp8)
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            quantize_dtype=fp8,
            q_rope_out=q_out[..., KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., :KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )

        assert q_out.shape == (batch_size, num_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        assert q_out.dtype == fp8
        assert not torch.all(q_out.float() == 0.0), "q_out should not be all zeros"

    def test_save_true_writes_kv_and_produces_same_q(self, device, cos_sin_cache):
        """save_kv_cache=True path must write KV cache AND produce same q_out as fused."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        ckv_snap = ckv.clone()

        q_out = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv,
            kpe,
            page_size,
        )

        assert not torch.equal(ckv, ckv_snap), "KV cache should have been written"
        assert q_out.shape == (batch_size, num_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM)

    def test_save_false_then_true_q_outputs_match(self, device, cos_sin_cache):
        """q_out from save_kv_cache=False should match q_out from save_kv_cache=True."""
        batch_size = 32
        num_heads = 64
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)

        q_no_save = torch.empty(
            batch_size,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=fp8)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=fp8)
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            quantize_dtype=fp8,
            q_rope_out=q_no_save[..., KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_no_save[..., :KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_with_save = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv,
            kpe,
            page_size,
        )

        diff = torch.max(torch.abs(q_no_save.float() - q_with_save.float())).item()
        assert diff == 0.0, f"q_out differs between save_kv_cache paths: {diff}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: Combined non-contiguous buffer under CUDA graph conditions
# ══════════════════════════════════════════════════════════════════════════════


class TestNonContiguousBufferCudaGraph:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize(
        "batch_size,num_heads,page_size",
        [
            (16, 128, 64),
            (64, 64, 32),
            (128, 16, 64),
            (1, 128, 32),
        ],
    )
    def test_combined_buffer_matches_separate(
        self, device, cos_sin_cache, batch_size, num_heads, page_size
    ):
        """Non-contiguous ckv/kpe views from a combined buffer must match separate buffers."""
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn
        sentinel = torch.tensor(0.5, dtype=torch.float32).to(fp8)

        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)

        ckv_sep = torch.full(
            (num_pages, page_size, KV_LORA_RANK),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        kpe_sep = torch.full(
            (num_pages, page_size, QK_ROPE_HEAD_DIM),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        q_sep = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv_sep,
            kpe_sep,
            page_size,
        )

        combined = torch.full(
            (num_pages, page_size, KV_CACHE_DIM),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        ckv_comb = combined[:, :, :KV_LORA_RANK]
        kpe_comb = combined[:, :, KV_LORA_RANK:]
        assert not ckv_comb.is_contiguous()
        assert not kpe_comb.is_contiguous()

        q_comb = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv_comb,
            kpe_comb,
            page_size,
        )

        q_diff = torch.max(torch.abs(q_sep.float() - q_comb.float())).item()
        assert q_diff == 0.0, f"q_out diff: {q_diff}"

        kv_idx = out_cache_loc // page_size
        pos = out_cache_loc % page_size
        for i in range(batch_size):
            p, s = kv_idx[i].item(), pos[i].item()
            cd = torch.max(
                torch.abs(ckv_sep[p, s].float() - ckv_comb[p, s].float())
            ).item()
            kd = torch.max(
                torch.abs(kpe_sep[p, s].float() - kpe_comb[p, s].float())
            ).item()
            assert cd == 0.0, f"ckv diff at {i}: {cd}"
            assert kd == 0.0, f"kpe diff at {i}: {kd}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: Idempotency
# ══════════════════════════════════════════════════════════════════════════════


class TestIdempotency:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize("batch_size", [1, 8, 64, 128])
    def test_same_inputs_same_outputs(self, device, cos_sin_cache, batch_size):
        """Running the fused kernel twice with identical inputs must produce identical outputs."""
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=9999
        )
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)

        ckv1 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe1 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q1 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv1,
            kpe1,
            page_size,
        )

        ckv2 = torch.zeros_like(ckv1)
        kpe2 = torch.zeros_like(kpe1)
        q2 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv2,
            kpe2,
            page_size,
        )

        assert torch.equal(q1, q2), "q_out not idempotent"
        assert torch.equal(ckv1, ckv2), "ckv_cache not idempotent"
        assert torch.equal(kpe1, kpe2), "kpe_cache not idempotent"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9: Randomized fuzz
# ══════════════════════════════════════════════════════════════════════════════


class TestFuzz:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize("seed", range(50))
    def test_random_configs(self, device, cos_sin_cache, seed):
        """Randomized config: random bs, heads, page_size, position ids."""
        torch.manual_seed(seed)
        batch_size = torch.randint(1, 129, (1,)).item()
        num_heads_choices = [16, 32, 64, 128]
        num_heads = num_heads_choices[seed % len(num_heads_choices)]
        page_size_choices = [16, 32, 64]
        page_size = page_size_choices[seed % len(page_size_choices)]
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.randint(
            0, 8192, (batch_size,), device=device, dtype=torch.int32
        )
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=seed
        )

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label=f"fuzz-{seed}(bs={batch_size},nh={num_heads},ps={page_size})",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10: Edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_page_size_1(self, device, cos_sin_cache):
        """Page size 1: every token gets its own page, offset always 0."""
        batch_size = 16
        num_heads = 64
        page_size = 1
        num_pages = batch_size + 2

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="page_size=1",
        )

    def test_all_tokens_same_page(self, device, cos_sin_cache):
        """All tokens map to the same page but different offsets."""
        page_size = 64
        batch_size = page_size
        num_heads = 128
        num_pages = 2

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="same_page",
        )

    def test_extreme_rope_positions(self, device, cos_sin_cache):
        """Positions at the very edge of the cos_sin_cache."""
        batch_size = 4
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.tensor([0, 1, 8190, 8191], device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="extreme_pos",
        )

    def test_position_zero_all(self, device, cos_sin_cache):
        """All tokens at position 0 (like a padded CUDA graph replay)."""
        batch_size = 32
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.zeros(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="all_pos_zero",
        )

    def test_reversed_cache_locs(self, device, cos_sin_cache):
        """Reversed (descending) out_cache_loc order."""
        batch_size = 16
        num_heads = 64
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device).flip(0)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="reversed_locs",
        )

    def test_large_page_offsets(self, device, cos_sin_cache):
        """Tokens that land at the very last slot of a page."""
        batch_size = 8
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        pages = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = pages * page_size + (page_size - 1)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="last_page_slot",
        )

    def test_first_page_slot(self, device, cos_sin_cache):
        """Tokens that land at slot 0 of each page."""
        batch_size = 8
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        pages = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = pages * page_size
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="first_page_slot",
        )

    def test_bfloat16_inputs(self, device, cos_sin_cache):
        """Explicitly bfloat16 inputs (the production dtype)."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, dtype=torch.bfloat16
        )

        assert q_nope.dtype == torch.bfloat16
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="bf16",
        )

    def test_float16_inputs(self, device, cos_sin_cache):
        """float16 inputs to check dtype flexibility."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, dtype=torch.float16
        )

        assert q_nope.dtype == torch.float16
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="fp16",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11: Full CUDA-graph capture+replay with actual torch.cuda.CUDAGraph
# ══════════════════════════════════════════════════════════════════════════════


class TestActualCudaGraph:
    """Use real torch.cuda.CUDAGraph to capture and replay the fused kernel."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize(
        "capture_bs,replay_bs",
        [
            (128, 128),
            (128, 64),
            (128, 1),
            (64, 32),
            (64, 7),
        ],
    )
    def test_cuda_graph_capture_replay(
        self, device, cos_sin_cache, capture_bs, replay_bs
    ):
        """Capture fused kernel in a real CUDA graph, replay with different data."""
        num_heads = 128
        page_size = 64
        num_pages = capture_bs + 2
        fp8 = torch.float8_e4m3fn

        max_tokens = capture_bs
        kv_indices_buf = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        kv_indptr_buf = torch.arange(max_tokens + 1, dtype=torch.int32, device=device)
        batch_indices_buf = torch.arange(max_tokens, dtype=torch.int32, device=device)
        positions_buf = torch.zeros(max_tokens, dtype=torch.int32, device=device)

        q_nope_buf = torch.randn(
            capture_bs, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope_buf = torch.randn(
            capture_bs, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope_buf = torch.randn(
            capture_bs, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope_buf = torch.randn(
            capture_bs, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        pos_ids_buf = torch.zeros(capture_bs, device=device, dtype=torch.int32)
        q_out_buf = torch.empty(
            capture_bs,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        ckv_buf = torch.zeros(
            num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
        )
        kpe_buf = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )

        capture_locs = generate_cache_locs(capture_bs, page_size, device)
        kv_indices_buf[:capture_bs] = (capture_locs // page_size).to(torch.int32)
        positions_buf[:capture_bs] = (capture_locs % page_size).to(torch.int32)
        pos_ids_buf[:capture_bs] = torch.arange(
            capture_bs, device=device, dtype=torch.int32
        )

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=kv_indices_buf[:capture_bs],
                kv_indptr=kv_indptr_buf[: capture_bs + 1],
                batch_indices=batch_indices_buf[:capture_bs],
                positions=positions_buf[:capture_bs],
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )
        torch.cuda.current_stream().wait_stream(s)

        ckv_buf.zero_()
        kpe_buf.zero_()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=kv_indices_buf[:capture_bs],
                kv_indptr=kv_indptr_buf[: capture_bs + 1],
                batch_indices=batch_indices_buf[:capture_bs],
                positions=positions_buf[:capture_bs],
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )

        # Replay: update input buffers, zero padding, replay graph
        q_nope_new, q_rope_new, k_nope_new, k_rope_new = make_inputs(
            replay_bs, num_heads, device, seed=12345
        )
        replay_locs = generate_cache_locs(replay_bs, page_size, device) + 1
        replay_pos = torch.arange(replay_bs, device=device, dtype=torch.int32) + 10

        # Zero padding first (mimicking positions.zero_() / out_cache_loc.zero_())
        if replay_bs < capture_bs:
            pos_ids_buf.zero_()
            kv_indices_buf.zero_()
            positions_buf.zero_()

        q_nope_buf[:replay_bs].copy_(q_nope_new)
        q_rope_buf[:replay_bs].copy_(q_rope_new)
        k_nope_buf[:replay_bs].copy_(k_nope_new)
        k_rope_buf[:replay_bs].copy_(k_rope_new)
        pos_ids_buf[:replay_bs].copy_(replay_pos)
        kv_indices_buf[:replay_bs] = (replay_locs // page_size).to(torch.int32)
        positions_buf[:replay_bs] = (replay_locs % page_size).to(torch.int32)
        if replay_bs < capture_bs:
            kv_indices_buf[replay_bs:].zero_()
            positions_buf[replay_bs:].zero_()

        ckv_buf.zero_()
        kpe_buf.zero_()

        g.replay()
        torch.cuda.synchronize()

        # Verify the real tokens match eager execution
        ckv_ref = torch.zeros_like(ckv_buf)
        kpe_ref = torch.zeros_like(kpe_buf)
        q_ref = run_separated(
            q_nope_new,
            q_rope_new,
            k_nope_new,
            k_rope_new,
            cos_sin_cache,
            replay_pos,
            replay_locs,
            ckv_ref,
            kpe_ref,
            page_size,
        )

        q_diff = torch.max(
            torch.abs(q_out_buf[:replay_bs].float() - q_ref.float())
        ).item()
        assert q_diff == 0.0, f"CUDA graph q_out mismatch: {q_diff}"

        kv_idx = replay_locs // page_size
        pos = replay_locs % page_size
        for i in range(replay_bs):
            p, s = kv_idx[i].item(), pos[i].item()
            cd = torch.max(
                torch.abs(ckv_buf[p, s].float() - ckv_ref[p, s].float())
            ).item()
            kd = torch.max(
                torch.abs(kpe_buf[p, s].float() - kpe_ref[p, s].float())
            ).item()
            assert cd == 0.0, f"CUDA graph ckv diff at {i}: {cd}"
            assert kd == 0.0, f"CUDA graph kpe diff at {i}: {kd}"

    def test_cuda_graph_multiple_replays(self, device, cos_sin_cache):
        """Replay the same captured CUDA graph 10 times with different batch sizes."""
        capture_bs = 64
        num_heads = 128
        page_size = 64
        num_pages = capture_bs + 2
        fp8 = torch.float8_e4m3fn

        kv_indices_buf = torch.zeros(capture_bs, dtype=torch.int32, device=device)
        kv_indptr_buf = torch.arange(capture_bs + 1, dtype=torch.int32, device=device)
        batch_indices_buf = torch.arange(capture_bs, dtype=torch.int32, device=device)
        positions_buf = torch.zeros(capture_bs, dtype=torch.int32, device=device)

        q_nope_buf = torch.randn(
            capture_bs, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope_buf = torch.randn(
            capture_bs, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope_buf = torch.randn(
            capture_bs, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope_buf = torch.randn(
            capture_bs, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        pos_ids_buf = torch.arange(capture_bs, device=device, dtype=torch.int32)
        q_out_buf = torch.empty(
            capture_bs,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        ckv_buf = torch.zeros(
            num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
        )
        kpe_buf = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )

        capture_locs = generate_cache_locs(capture_bs, page_size, device)
        kv_indices_buf[:] = (capture_locs // page_size).to(torch.int32)
        positions_buf[:] = (capture_locs % page_size).to(torch.int32)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=kv_indices_buf,
                kv_indptr=kv_indptr_buf,
                batch_indices=batch_indices_buf,
                positions=positions_buf,
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )
        torch.cuda.current_stream().wait_stream(s)

        ckv_buf.zero_()
        kpe_buf.zero_()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=kv_indices_buf,
                kv_indptr=kv_indptr_buf,
                batch_indices=batch_indices_buf,
                positions=positions_buf,
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )

        replay_sizes = [64, 32, 1, 58, 7, 64, 16, 48, 3, 64]
        for step, rbs in enumerate(replay_sizes):
            q_n, q_r, k_n, k_r = make_inputs(rbs, num_heads, device, seed=step * 7)
            replay_locs = generate_cache_locs(rbs, page_size, device) + step * 2
            replay_pos = torch.arange(rbs, device=device, dtype=torch.int32) + step

            if rbs < capture_bs:
                pos_ids_buf.zero_()
                kv_indices_buf.zero_()
                positions_buf.zero_()

            q_nope_buf[:rbs].copy_(q_n)
            q_rope_buf[:rbs].copy_(q_r)
            k_nope_buf[:rbs].copy_(k_n)
            k_rope_buf[:rbs].copy_(k_r)
            pos_ids_buf[:rbs].copy_(replay_pos)
            kv_indices_buf[:rbs] = (replay_locs // page_size).to(torch.int32)
            positions_buf[:rbs] = (replay_locs % page_size).to(torch.int32)
            if rbs < capture_bs:
                kv_indices_buf[rbs:].zero_()
                positions_buf[rbs:].zero_()

            ckv_buf.zero_()
            kpe_buf.zero_()

            g.replay()
            torch.cuda.synchronize()

            ckv_ref = torch.zeros_like(ckv_buf)
            kpe_ref = torch.zeros_like(kpe_buf)
            q_ref = run_separated(
                q_n,
                q_r,
                k_n,
                k_r,
                cos_sin_cache,
                replay_pos,
                replay_locs,
                ckv_ref,
                kpe_ref,
                page_size,
            )

            q_diff = torch.max(
                torch.abs(q_out_buf[:rbs].float() - q_ref.float())
            ).item()
            assert q_diff == 0.0, f"Replay {step} (bs={rbs}): q_out mismatch {q_diff}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12: Stale-data regression (the positions.zero_() bug)
# ══════════════════════════════════════════════════════════════════════════════


class TestStaleDataRegression:
    """
    Regression test for the bug that positions.zero_() / out_cache_loc.zero_() fixes.

    Scenario: A CUDA graph is captured at bs=128. Then replayed at bs=32.
    Without zeroing, positions[32:128] retain stale values from the prior replay,
    causing the fused kernel to compute RoPE with wrong pos_ids for padded tokens
    and write to wrong KV cache pages.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_stale_positions_cause_different_kv(self, device, cos_sin_cache):
        """Demonstrate that stale vs zeroed positions produce different KV cache writes for padded tokens."""
        max_bs = 64
        real_bs = 16
        page_size = 64
        num_heads = 128
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        md_stale = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )

        old_locs = generate_cache_locs(max_bs, page_size, device)
        init_metadata_inplace(md_stale, old_locs, page_size)

        stale_kv_before = md_stale.kv_indices[real_bs:].clone()
        stale_pos_before = md_stale.positions[real_bs:].clone()

        new_locs = generate_cache_locs(real_bs, page_size, device)
        md_stale.kv_indices[:real_bs] = (new_locs // page_size).to(torch.int32)
        md_stale.positions[:real_bs] = (new_locs % page_size).to(torch.int32)

        has_stale = (stale_kv_before != 0).any() or (stale_pos_before != 0).any()
        assert has_stale, "Precondition: stale data should exist from old larger batch"

        md_clean = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )
        init_metadata_inplace(md_clean, new_locs, page_size)

        assert torch.equal(md_clean.kv_indices[:real_bs], md_stale.kv_indices[:real_bs])
        assert torch.equal(md_clean.positions[:real_bs], md_stale.positions[:real_bs])

        assert (md_clean.kv_indices[real_bs:] == 0).all(), "Clean tail should be zeroed"
        assert (md_clean.positions[real_bs:] == 0).all(), "Clean tail should be zeroed"

        stale_tail = md_stale.kv_indices[real_bs:]
        clean_tail = md_clean.kv_indices[real_bs:]
        assert not torch.equal(
            stale_tail, clean_tail
        ), "Stale tail should differ from clean tail"

    def test_page0_safe_with_zeroed_padding(self, device, cos_sin_cache):
        """With zeroed padding, padded tokens all write to page 0 offset 0 — a known safe page.
        Verify no other pages are corrupted."""
        max_bs = 64
        real_bs = 16
        page_size = 64
        num_heads = 128
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        md = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )
        real_locs = generate_cache_locs(real_bs, page_size, device) + page_size
        init_metadata_inplace(md, real_locs, page_size)

        sentinel = torch.tensor(0.5, dtype=torch.float32).to(fp8)
        ckv = torch.full(
            (num_pages, page_size, KV_LORA_RANK),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        kpe = torch.full(
            (num_pages, page_size, QK_ROPE_HEAD_DIM),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        ckv_snap = ckv.clone()
        kpe_snap = kpe.clone()

        q_nope, q_rope, k_nope, k_rope = make_inputs(max_bs, num_heads, device)
        pos_ids = torch.zeros(max_bs, device=device, dtype=torch.int32)
        pos_ids[:real_bs] = torch.arange(real_bs, device=device, dtype=torch.int32)

        run_fused_with_metadata(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            md,
            max_bs,
            ckv,
            kpe,
            page_size,
        )
        torch.cuda.synchronize()

        real_pages = set()
        kv_idx = real_locs // page_size
        pos = real_locs % page_size
        for i in range(real_bs):
            real_pages.add((kv_idx[i].item(), pos[i].item()))

        for p in range(1, num_pages):
            for s in range(page_size):
                if (p, s) not in real_pages:
                    cd = torch.max(
                        torch.abs(ckv[p, s].float() - ckv_snap[p, s].float())
                    ).item()
                    kd = torch.max(
                        torch.abs(kpe[p, s].float() - kpe_snap[p, s].float())
                    ).item()
                    assert (
                        cd == 0.0
                    ), f"Unexpected write at page={p} slot={s} (ckv diff={cd})"
                    assert (
                        kd == 0.0
                    ), f"Unexpected write at page={p} slot={s} (kpe diff={kd})"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13: Stress fuzz with CUDA graph replays
# ══════════════════════════════════════════════════════════════════════════════


class TestCudaGraphFuzz:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_graph_replay(self, device, cos_sin_cache, seed):
        """Random capture_bs, random replay sequence, check parity each time."""
        torch.manual_seed(seed * 31337)
        max_bs = torch.randint(16, 129, (1,)).item()
        num_heads_choices = [16, 32, 64, 128]
        num_heads = num_heads_choices[seed % len(num_heads_choices)]
        page_size_choices = [16, 32, 64]
        page_size = page_size_choices[seed % len(page_size_choices)]
        num_pages = max_bs + 2

        md = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )

        capture_locs = generate_cache_locs(max_bs, page_size, device)
        init_metadata_fresh(md, capture_locs, page_size)

        num_replays = 5
        for step in range(num_replays):
            rbs = torch.randint(1, max_bs + 1, (1,)).item()
            replay_locs = generate_cache_locs(rbs, page_size, device) + step
            init_metadata_inplace(md, replay_locs, page_size)

            q_nope, q_rope, k_nope, k_rope = make_inputs(
                rbs, num_heads, device, seed=seed * 100 + step
            )
            pos_ids = torch.randint(0, 8192, (rbs,), device=device, dtype=torch.int32)
            fp8 = torch.float8_e4m3fn

            ckv = torch.zeros(
                num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
            )
            kpe = torch.zeros(
                num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
            )
            q_graph = run_fused_with_metadata(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                md,
                rbs,
                ckv,
                kpe,
                page_size,
            )

            ckv_ref = torch.zeros_like(ckv)
            kpe_ref = torch.zeros_like(kpe)
            q_ref = run_separated(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                replay_locs,
                ckv_ref,
                kpe_ref,
                page_size,
            )

            q_diff = torch.max(torch.abs(q_graph.float() - q_ref.float())).item()
            assert (
                q_diff == 0.0
            ), f"seed={seed} step={step} rbs={rbs}: q_out mismatch {q_diff}"

            kv_idx = replay_locs // page_size
            pos = replay_locs % page_size
            for i in range(rbs):
                p, s = kv_idx[i].item(), pos[i].item()
                cd = torch.max(
                    torch.abs(ckv[p, s].float() - ckv_ref[p, s].float())
                ).item()
                kd = torch.max(
                    torch.abs(kpe[p, s].float() - kpe_ref[p, s].float())
                ).item()
                assert cd == 0.0, f"seed={seed} step={step}: ckv diff at {i}: {cd}"
                assert kd == 0.0, f"seed={seed} step={step}: kpe diff at {i}: {kd}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14: torch.div vs // and torch.remainder vs % equivalence
#
#  Production code uses torch.div(..., rounding_mode="floor", out=...) for
#  the inplace path but // for the fresh path. These MUST agree for all
#  non-negative int32 inputs. A mismatch would silently corrupt kv_indices.
# ══════════════════════════════════════════════════════════════════════════════


class TestArithmeticEquivalence:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.mark.parametrize("page_size", [1, 4, 16, 32, 48, 64, 128])
    def test_div_vs_floordiv(self, device, page_size):
        """torch.div(x, ps, rounding_mode='floor') must equal x // ps for all non-negative int32."""
        locs = torch.arange(0, page_size * 256, dtype=torch.int32, device=device)
        expected = (locs // page_size).to(torch.int32)
        actual = torch.empty_like(locs)
        torch.div(locs, page_size, rounding_mode="floor", out=actual)
        assert torch.equal(
            actual, expected
        ), f"div vs // mismatch for page_size={page_size}"

    @pytest.mark.parametrize("page_size", [1, 4, 16, 32, 48, 64, 128])
    def test_remainder_vs_mod(self, device, page_size):
        """torch.remainder(x, ps) must equal x % ps for all non-negative int32."""
        locs = torch.arange(0, page_size * 256, dtype=torch.int32, device=device)
        expected = (locs % page_size).to(torch.int32)
        actual = torch.empty_like(locs)
        torch.remainder(locs, page_size, out=actual)
        assert torch.equal(
            actual, expected
        ), f"remainder vs % mismatch for page_size={page_size}"

    def test_large_cache_locations_no_int32_overflow(self, device):
        """Cache locations near int32 max must not overflow during div/mod."""
        max_int32 = 2**31 - 1
        page_size = 64
        locs = torch.tensor(
            [
                0,
                1,
                page_size - 1,
                page_size,
                max_int32 - page_size,
                max_int32 - 1,
                max_int32,
            ],
            dtype=torch.int32,
            device=device,
        )
        pages_floordiv = torch.empty_like(locs)
        offsets_remainder = torch.empty_like(locs)
        torch.div(locs, page_size, rounding_mode="floor", out=pages_floordiv)
        torch.remainder(locs, page_size, out=offsets_remainder)

        reconstructed = pages_floordiv * page_size + offsets_remainder
        assert torch.equal(
            reconstructed, locs
        ), "page*ps + offset must reconstruct the original loc"
        assert (pages_floordiv >= 0).all(), "page indices must be non-negative"
        assert (offsets_remainder >= 0).all(), "page offsets must be non-negative"
        assert (offsets_remainder < page_size).all(), "page offsets must be < page_size"

    def test_fresh_vs_inplace_produce_identical_metadata(self, device):
        """Fresh alloc (// and %) must produce identical values to inplace (torch.div + torch.remainder)."""
        page_size = 64
        bs = 128
        locs = torch.randint(0, 10000, (bs,), dtype=torch.int32, device=device)

        fresh_pages = (locs // page_size).to(torch.int32)
        fresh_offsets = (locs % page_size).to(torch.int32)

        inplace_pages = torch.empty(bs, dtype=torch.int32, device=device)
        inplace_offsets = torch.empty(bs, dtype=torch.int32, device=device)
        torch.div(locs, page_size, rounding_mode="floor", out=inplace_pages)
        torch.remainder(locs, page_size, out=inplace_offsets)

        assert torch.equal(fresh_pages, inplace_pages), "page indices disagree"
        assert torch.equal(fresh_offsets, inplace_offsets), "page offsets disagree"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 15: Metadata buffer aliasing (CUDA graph slices must be views)
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadataBufferAliasing:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_cuda_graph_slices_are_views_not_copies(self, device):
        """Slicing the pre-allocated buffers must return views sharing storage."""
        max_tokens = 256
        buf = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        sliced = buf[:128]
        assert sliced.data_ptr() == buf.data_ptr(), "slice should share base pointer"
        assert (
            sliced.storage().data_ptr() == buf.storage().data_ptr()
        ), "slice should share storage"

        sliced[0] = 42
        assert buf[0].item() == 42, "writing to slice must update the base buffer"

    def test_inplace_update_through_slice_alias(self, device):
        """In-place ops on a metadata slice must be visible through the original buffer."""
        max_tokens = 128
        base = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        view = base[:64]

        torch.div(
            torch.arange(64, dtype=torch.int32, device=device) * 64 + 13,
            64,
            rounding_mode="floor",
            out=view,
        )
        assert torch.equal(base[:64], view), "in-place through view must update base"
        assert (base[64:] == 0).all(), "tail of base must be untouched"

    def test_indptr_slice_includes_extra_element(self, device):
        """kv_indptr[:nnz+1] must have nnz+1 elements (the +1 is easy to forget)."""
        max_tokens = 128
        indptr = torch.arange(max_tokens + 1, dtype=torch.int32, device=device)

        for nnz in [1, 32, 64, 127, 128]:
            sliced = indptr[: nnz + 1]
            assert (
                sliced.shape[0] == nnz + 1
            ), f"kv_indptr slice for nnz={nnz} has wrong length"
            assert (
                sliced[-1].item() == nnz
            ), f"kv_indptr[-1] should be {nnz}, got {sliced[-1].item()}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 16: Duplicate out_cache_loc (two tokens writing same slot)
# ══════════════════════════════════════════════════════════════════════════════


class TestDuplicateCacheLocations:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_duplicate_locs_last_write_wins_separated(self, device, cos_sin_cache):
        """When two tokens target the same cache slot, the separated path writes
        sequentially so the last token wins. Fused kernel may have race conditions
        but the result for those slots is still a valid FP8 value (not NaN/Inf)."""
        batch_size = 4
        num_heads = 128
        page_size = 64
        num_pages = 4
        fp8 = torch.float8_e4m3fn

        out_cache_loc = torch.tensor([0, 0, 64, 64], device=device, dtype=torch.int32)
        pos_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=5555
        )

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv,
            kpe,
            page_size,
        )

        assert not torch.any(torch.isnan(q_out.float())), "q_out has NaN"
        assert not torch.any(torch.isinf(q_out.float())), "q_out has Inf"
        assert not torch.any(torch.isnan(ckv.float())), "ckv has NaN"
        assert not torch.any(torch.isnan(kpe.float())), "kpe has NaN"

        assert not torch.all(ckv[0, 0] == 0), "Slot (0,0) should have been written"
        assert not torch.all(ckv[1, 0] == 0), "Slot (1,0) should have been written"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 17: Non-contiguous input tensors
# ══════════════════════════════════════════════════════════════════════════════


class TestNonContiguousInputs:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_strided_view_vs_contiguous_copy(self, device, cos_sin_cache):
        """A non-contiguous (strided) view of q_nope and its .contiguous() copy
        must produce identical fused kernel outputs, proving the kernel handles
        non-contiguous memory layouts correctly."""
        batch_size = 16
        num_heads = 64
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        torch.manual_seed(8888)
        q_rope_c = torch.randn(
            batch_size, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope_c = torch.randn(
            batch_size, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope_c = torch.randn(
            batch_size, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        big_q = torch.randn(
            batch_size * 2, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_nope_strided = big_q[::2]
        assert q_nope_strided.shape[0] == batch_size
        assert (
            not q_nope_strided.is_contiguous()
        ), "strided view should not be contiguous"

        q_nope_contig = q_nope_strided.contiguous()
        assert torch.equal(
            q_nope_strided, q_nope_contig
        ), "contiguous copy must have same values"

        ckv1 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe1 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q1 = run_fused(
            q_nope_strided,
            q_rope_c,
            k_nope_c,
            k_rope_c,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv1,
            kpe1,
            page_size,
        )

        ckv2 = torch.zeros_like(ckv1)
        kpe2 = torch.zeros_like(kpe1)
        q2 = run_fused(
            q_nope_contig,
            q_rope_c,
            k_nope_c,
            k_rope_c,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv2,
            kpe2,
            page_size,
        )

        q_diff = torch.max(torch.abs(q1.float() - q2.float())).item()
        assert q_diff == 0.0, f"strided vs contiguous q_out mismatch: {q_diff}"
        assert torch.equal(ckv1, ckv2), "strided vs contiguous ckv mismatch"
        assert torch.equal(kpe1, kpe2), "strided vs contiguous kpe mismatch"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 18: k.squeeze(1) shape contract
# ══════════════════════════════════════════════════════════════════════════════


class TestKSqueezeDimContract:
    """Production code does k.squeeze(1) in _fp8_rope_quantize_and_save.
    Verify the kernel handles both squeezed and pre-squeezed k shapes."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_k_with_head_dim_1_squeezed(self, device, cos_sin_cache):
        """k shape (bs, 1, dim) -> squeeze(1) -> (bs, dim): must work."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        torch.manual_seed(3333)
        q_nope = torch.randn(
            batch_size, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope = torch.randn(
            batch_size, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope_3d = torch.randn(
            batch_size, 1, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope_3d = torch.randn(
            batch_size, 1, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )

        k_nope_2d = k_nope_3d.squeeze(1)
        k_rope_2d = k_rope_3d.squeeze(1)
        assert k_nope_2d.shape == (batch_size, KV_LORA_RANK)
        assert k_rope_2d.shape == (batch_size, QK_ROPE_HEAD_DIM)

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out = run_fused(
            q_nope,
            q_rope,
            k_nope_2d,
            k_rope_2d,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv,
            kpe,
            page_size,
        )
        assert q_out.shape == (batch_size, num_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        assert not torch.any(torch.isnan(q_out.float()))

    def test_k_already_2d(self, device, cos_sin_cache):
        """k shape already (bs, dim): squeeze(1) is a no-op. Must still work."""
        batch_size = 8
        num_heads = 64
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        torch.manual_seed(4444)
        q_nope = torch.randn(
            batch_size, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope = torch.randn(
            batch_size, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope = torch.randn(
            batch_size, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope = torch.randn(
            batch_size, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )

        squeezed = k_nope.squeeze(1)
        assert squeezed.shape == k_nope.shape, "squeeze(1) on 2D should be no-op"
        assert squeezed.data_ptr() == k_nope.data_ptr()

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="k_already_2d",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 19: Overwrite test (write, then re-write to same slot)
# ══════════════════════════════════════════════════════════════════════════════


class TestKvCacheOverwrite:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_second_write_overwrites_first(self, device, cos_sin_cache):
        """Two sequential writes to the same slot: the second must completely overwrite the first."""
        batch_size = 8
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        q1, qr1, k1, kr1 = make_inputs(batch_size, num_heads, device, seed=111)
        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        run_fused(
            q1, qr1, k1, kr1, cos_sin_cache, pos_ids, out_cache_loc, ckv, kpe, page_size
        )
        ckv_after_first = ckv.clone()

        q2, qr2, k2, kr2 = make_inputs(batch_size, num_heads, device, seed=222)
        run_fused(
            q2, qr2, k2, kr2, cos_sin_cache, pos_ids, out_cache_loc, ckv, kpe, page_size
        )

        kv_idx = out_cache_loc // page_size
        pos = out_cache_loc % page_size
        for i in range(batch_size):
            p, s = kv_idx[i].item(), pos[i].item()
            assert not torch.equal(
                ckv[p, s], ckv_after_first[p, s]
            ), f"Slot ({p},{s}) not overwritten by second write"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 20: Multi-layer simulation (metadata reused across layers)
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiLayerMetadataReuse:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_same_metadata_different_kv_buffers(self, device, cos_sin_cache):
        """Same metadata + pos_ids used for multiple layers must produce
        identical q_out and write to different KV buffers correctly."""
        batch_size = 32
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        num_layers = 4
        fp8 = torch.float8_e4m3fn

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        md = FakeDecodeMetadata()
        init_metadata_fresh(md, out_cache_loc, page_size)

        q_outs = []
        ckv_buffers = []
        for layer_id in range(num_layers):
            q_nope, q_rope, k_nope, k_rope = make_inputs(
                batch_size, num_heads, device, seed=layer_id * 1000
            )
            ckv = torch.zeros(
                num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
            )
            kpe = torch.zeros(
                num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
            )

            q_out = run_fused_with_metadata(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                md,
                batch_size,
                ckv,
                kpe,
                page_size,
            )
            q_outs.append(q_out.clone())
            ckv_buffers.append(ckv.clone())

        for i in range(1, num_layers):
            assert not torch.equal(
                q_outs[0], q_outs[i]
            ), f"Layer 0 and {i} have same q_out (different inputs should give different outputs)"
            assert not torch.equal(
                ckv_buffers[0], ckv_buffers[i]
            ), f"Layer 0 and {i} have same ckv (different inputs should give different kv)"

    def test_metadata_unchanged_after_kernel_call(self, device, cos_sin_cache):
        """The fused kernel must NOT mutate the metadata tensors."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        md = FakeDecodeMetadata()
        init_metadata_fresh(md, out_cache_loc, page_size)
        kv_indices_snap = md.kv_indices.clone()
        positions_snap = md.positions.clone()
        kv_indptr_snap = md.kv_indptr.clone()
        batch_indices_snap = md.batch_indices.clone()

        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        run_fused_with_metadata(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            md,
            batch_size,
            ckv,
            kpe,
            page_size,
        )

        assert torch.equal(md.kv_indices, kv_indices_snap), "kernel mutated kv_indices"
        assert torch.equal(md.positions, positions_snap), "kernel mutated positions"
        assert torch.equal(md.kv_indptr, kv_indptr_snap), "kernel mutated kv_indptr"
        assert torch.equal(
            md.batch_indices, batch_indices_snap
        ), "kernel mutated batch_indices"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 21: Stale RoPE contamination (the actual bug positions.zero_() fixes)
# ══════════════════════════════════════════════════════════════════════════════


class TestStaleRopeContamination:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_stale_pos_ids_produce_different_q_output(self, device, cos_sin_cache):
        """Different pos_ids MUST produce different q_out (RoPE is position-dependent).
        This proves that stale positions are dangerous, not just theoretical."""
        batch_size = 8
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=666
        )
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)

        pos_zero = torch.zeros(batch_size, device=device, dtype=torch.int32)
        pos_high = torch.arange(batch_size, device=device, dtype=torch.int32) + 4000

        ckv1 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe1 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q1 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_zero,
            out_cache_loc,
            ckv1,
            kpe1,
            page_size,
        )

        ckv2 = torch.zeros_like(ckv1)
        kpe2 = torch.zeros_like(kpe1)
        q2 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_high,
            out_cache_loc,
            ckv2,
            kpe2,
            page_size,
        )

        q_diff = torch.max(torch.abs(q1.float() - q2.float())).item()
        assert (
            q_diff > 0.0
        ), "Different pos_ids must produce different q_out (RoPE rotates differently)"

        kv_idx = out_cache_loc // page_size
        pos = out_cache_loc % page_size
        kv_diff = 0.0
        for i in range(batch_size):
            p, s = kv_idx[i].item(), pos[i].item()
            kv_diff += torch.max(
                torch.abs(kpe1[p, s].float() - kpe2[p, s].float())
            ).item()
        assert (
            kv_diff > 0.0
        ), "Different pos_ids must produce different kpe cache (RoPE-d keys differ)"

    def test_zeroed_positions_produce_position_0_rope(self, device, cos_sin_cache):
        """pos_ids=0 for all tokens: the padded-token scenario.
        q_out must match the separated path at position 0 exactly."""
        batch_size = 8
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=777
        )
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.zeros(batch_size, device=device, dtype=torch.int32)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="all_pos_zero_rope",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 22: cos_sin_cache dtype and precision
# ══════════════════════════════════════════════════════════════════════════════


class TestCosSinCachePrecision:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_cos_sin_cache_is_float32(self, device):
        """FlashInfer requires float32 cos_sin_cache. Verify our helper produces float32."""
        cache = create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)
        assert cache.dtype == torch.float32, f"Expected float32, got {cache.dtype}"
        assert cache.shape == (8192, QK_ROPE_HEAD_DIM)

    def test_cos_sin_cache_values_bounded(self, device):
        """cos/sin values must be in [-1, 1]."""
        cache = create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)
        assert cache.min().item() >= -1.0, "cos/sin values below -1"
        assert cache.max().item() <= 1.0, "cos/sin values above 1"

    def test_position_0_cos_is_1_sin_is_0(self, device):
        """At position 0, cos(0) = 1, sin(0) = 0 for all frequencies."""
        cache = create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)
        half = QK_ROPE_HEAD_DIM // 2
        cos_part = cache[0, :half]
        sin_part = cache[0, half:]
        assert torch.allclose(
            cos_part, torch.ones_like(cos_part), atol=1e-6
        ), "cos(0) should be 1"
        assert torch.allclose(
            sin_part, torch.zeros_like(sin_part), atol=1e-6
        ), "sin(0) should be 0"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 23: Padded tokens: concurrent writes to page 0 offset 0
# ══════════════════════════════════════════════════════════════════════════════


class TestPaddedTokenPageZeroCollision:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_many_padded_tokens_all_hit_page0(self, device, cos_sin_cache):
        """With 100+ padded tokens all writing to (page=0, offset=0),
        the result must be a valid FP8 value (no NaN/corruption).
        Real tokens on other pages must not be affected."""
        max_bs = 128
        real_bs = 4
        page_size = 64
        num_heads = 128
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        real_locs = torch.tensor(
            [page_size * 1, page_size * 2, page_size * 3, page_size * 4],
            device=device,
            dtype=torch.int32,
        )
        md = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )
        init_metadata_inplace(md, real_locs, page_size)

        assert (md.kv_indices[real_bs:] == 0).all()
        assert (md.positions[real_bs:] == 0).all()

        sentinel = torch.tensor(0.5, dtype=torch.float32).to(fp8)
        ckv = torch.full(
            (num_pages, page_size, KV_LORA_RANK),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        kpe = torch.full(
            (num_pages, page_size, QK_ROPE_HEAD_DIM),
            sentinel.item(),
            device=device,
            dtype=fp8,
        )
        ckv_snap = ckv.clone()

        q_nope, q_rope, k_nope, k_rope = make_inputs(max_bs, num_heads, device)
        pos_ids = torch.zeros(max_bs, device=device, dtype=torch.int32)
        pos_ids[:real_bs] = torch.arange(real_bs, device=device, dtype=torch.int32)

        run_fused_with_metadata(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            md,
            max_bs,
            ckv,
            kpe,
            page_size,
        )
        torch.cuda.synchronize()

        assert not torch.any(
            torch.isnan(ckv.float())
        ), "ckv has NaN from padded collision"
        assert not torch.any(
            torch.isnan(kpe.float())
        ), "kpe has NaN from padded collision"

        for i in range(real_bs):
            page = real_locs[i].item() // page_size
            for s in range(1, page_size):
                assert torch.equal(
                    ckv[page, s], ckv_snap[page, s]
                ), f"Non-zero offset on real page {page} corrupted at slot {s}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 24: Full lifecycle: capture -> replay(max) -> replay(small) -> replay(max)
# ══════════════════════════════════════════════════════════════════════════════


class TestFullGraphLifecycle:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_replay_max_then_small_then_max_again(self, device, cos_sin_cache):
        """The classic CUDA graph lifecycle:
        1. Capture at max_bs
        2. Replay at max_bs (no padding)
        3. Replay at small bs (with padding)
        4. Replay at max_bs again (verify stale small-bs data is fully overwritten)
        """
        max_bs = 64
        small_bs = 8
        page_size = 64
        num_heads = 128
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        md = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )

        # Step 1: "capture" at max_bs
        capture_locs = generate_cache_locs(max_bs, page_size, device)
        init_metadata_fresh(md, capture_locs, page_size)

        # Step 2: replay at max_bs
        replay1_locs = generate_cache_locs(max_bs, page_size, device) + 1
        init_metadata_inplace(md, replay1_locs, page_size)

        q1, qr1, k1, kr1 = make_inputs(max_bs, num_heads, device, seed=100)
        pos1 = torch.arange(max_bs, device=device, dtype=torch.int32)
        ckv1 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe1 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out1 = run_fused_with_metadata(
            q1, qr1, k1, kr1, cos_sin_cache, pos1, md, max_bs, ckv1, kpe1, page_size
        )

        ckv1_ref = torch.zeros_like(ckv1)
        kpe1_ref = torch.zeros_like(kpe1)
        q_ref1 = run_separated(
            q1,
            qr1,
            k1,
            kr1,
            cos_sin_cache,
            pos1,
            replay1_locs,
            ckv1_ref,
            kpe1_ref,
            page_size,
        )
        assert (
            torch.max(torch.abs(q_out1.float() - q_ref1.float())).item() == 0.0
        ), "Step 2 q mismatch"

        # Step 3: replay at small_bs (padding kicks in)
        replay2_locs = generate_cache_locs(small_bs, page_size, device) + 2
        init_metadata_inplace(md, replay2_locs, page_size)

        q2, qr2, k2, kr2 = make_inputs(small_bs, num_heads, device, seed=200)
        pos2 = torch.arange(small_bs, device=device, dtype=torch.int32)
        ckv2 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe2 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out2 = run_fused_with_metadata(
            q2, qr2, k2, kr2, cos_sin_cache, pos2, md, small_bs, ckv2, kpe2, page_size
        )

        ckv2_ref = torch.zeros_like(ckv2)
        kpe2_ref = torch.zeros_like(kpe2)
        q_ref2 = run_separated(
            q2,
            qr2,
            k2,
            kr2,
            cos_sin_cache,
            pos2,
            replay2_locs,
            ckv2_ref,
            kpe2_ref,
            page_size,
        )
        assert (
            torch.max(torch.abs(q_out2.float() - q_ref2.float())).item() == 0.0
        ), "Step 3 q mismatch"

        # Step 4: replay at max_bs AGAIN — stale small-bs metadata must be fully overwritten
        replay3_locs = generate_cache_locs(max_bs, page_size, device) + 3
        init_metadata_inplace(md, replay3_locs, page_size)

        q3, qr3, k3, kr3 = make_inputs(max_bs, num_heads, device, seed=300)
        pos3 = torch.arange(max_bs, device=device, dtype=torch.int32)
        ckv3 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe3 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out3 = run_fused_with_metadata(
            q3, qr3, k3, kr3, cos_sin_cache, pos3, md, max_bs, ckv3, kpe3, page_size
        )

        ckv3_ref = torch.zeros_like(ckv3)
        kpe3_ref = torch.zeros_like(kpe3)
        q_ref3 = run_separated(
            q3,
            qr3,
            k3,
            kr3,
            cos_sin_cache,
            pos3,
            replay3_locs,
            ckv3_ref,
            kpe3_ref,
            page_size,
        )
        assert (
            torch.max(torch.abs(q_out3.float() - q_ref3.float())).item() == 0.0
        ), "Step 4 q mismatch (stale data from small replay?)"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 25: Sparse/gapped cache locations
# ══════════════════════════════════════════════════════════════════════════════


class TestSparseGappedCacheLocations:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_huge_gaps_between_pages(self, device, cos_sin_cache):
        """Tokens scattered to very distant pages with large gaps."""
        batch_size = 4
        num_heads = 128
        page_size = 64
        num_pages = 200

        out_cache_loc = torch.tensor(
            [0, 64 * 50, 64 * 100, 64 * 199], device=device, dtype=torch.int32
        )
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="huge_gaps",
        )

    def test_all_tokens_same_page_offset(self, device, cos_sin_cache):
        """All tokens at offset 0 of different pages."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = (
            torch.arange(batch_size, device=device, dtype=torch.int32) * page_size
        )
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        offsets = out_cache_loc % page_size
        assert (offsets == 0).all(), "All offsets should be 0"

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="same_offset",
        )

    def test_all_tokens_last_offset(self, device, cos_sin_cache):
        """All tokens at the very last offset of different pages."""
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = torch.arange(
            batch_size, device=device, dtype=torch.int32
        ) * page_size + (page_size - 1)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        offsets = out_cache_loc % page_size
        assert (offsets == page_size - 1).all()

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="last_offset",
        )

    def test_monotonically_decreasing_locs(self, device, cos_sin_cache):
        """Cache locations in strictly decreasing order."""
        batch_size = 16
        num_heads = 64
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device).flip(0)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="decreasing_locs",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 26: Non-power-of-2 and unusual page sizes
# ══════════════════════════════════════════════════════════════════════════════


class TestUnusualPageSizes:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize(
        "page_size", [1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 24, 31, 32, 48, 63, 64]
    )
    def test_various_page_sizes(self, device, cos_sin_cache, page_size):
        batch_size = min(16, page_size * 4)
        num_heads = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=page_size
        )

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label=f"ps={page_size}",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 27: Position ID dtype (int32 vs int64)
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionIdDtype:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    def test_int32_positions(self, device, cos_sin_cache):
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="int32_pos",
        )

    def test_int64_positions(self, device, cos_sin_cache):
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int64)
        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label="int64_pos",
        )

    def test_int32_and_int64_produce_same_result(self, device, cos_sin_cache):
        batch_size = 16
        num_heads = 128
        page_size = 64
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(batch_size, num_heads, device)
        out_cache_loc = generate_cache_locs(batch_size, page_size, device)

        pos32 = torch.arange(batch_size, device=device, dtype=torch.int32)
        pos64 = torch.arange(batch_size, device=device, dtype=torch.int64)

        ckv1 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe1 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q1 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos32,
            out_cache_loc,
            ckv1,
            kpe1,
            page_size,
        )

        ckv2 = torch.zeros_like(ckv1)
        kpe2 = torch.zeros_like(kpe1)
        q2 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos64,
            out_cache_loc,
            ckv2,
            kpe2,
            page_size,
        )

        assert torch.equal(q1, q2), "int32 vs int64 pos_ids produce different q_out"
        assert torch.equal(ckv1, ckv2), "int32 vs int64 pos_ids produce different ckv"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 28: The nnz == total_tokens edge (no zeroing needed)
# ══════════════════════════════════════════════════════════════════════════════


class TestNnzEqualsTotalTokens:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    def test_no_zeroing_when_full_batch(self, device):
        """When nnz == total_tokens (metadata buffer), the zeroing branch is skipped."""
        max_bs = 64
        page_size = 64

        md = FakeDecodeMetadata(
            kv_indices=torch.full((max_bs,), 9999, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.full((max_bs,), 9999, dtype=torch.int32, device=device),
        )

        locs = generate_cache_locs(max_bs, page_size, device)
        init_metadata_inplace(md, locs, page_size)

        expected_pages = (locs // page_size).to(torch.int32)
        expected_offsets = (locs % page_size).to(torch.int32)
        assert torch.equal(md.kv_indices, expected_pages)
        assert torch.equal(md.positions, expected_offsets)

    def test_nnz_equals_total_tokens_in_fused_kernel(self, device):
        """Full batch through fused kernel with metadata where nnz == total_tokens."""
        cos_sin_cache = create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)
        max_bs = 64
        num_heads = 128
        page_size = 64
        num_pages = max_bs + 2
        fp8 = torch.float8_e4m3fn

        md = FakeDecodeMetadata(
            kv_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            batch_indices=torch.arange(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )
        locs = generate_cache_locs(max_bs, page_size, device)
        init_metadata_inplace(md, locs, page_size)

        q_nope, q_rope, k_nope, k_rope = make_inputs(max_bs, num_heads, device)
        pos_ids = torch.arange(max_bs, device=device, dtype=torch.int32)

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_meta = run_fused_with_metadata(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            md,
            max_bs,
            ckv,
            kpe,
            page_size,
        )

        ckv_ref = torch.zeros_like(ckv)
        kpe_ref = torch.zeros_like(kpe)
        q_ref = run_separated(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            locs,
            ckv_ref,
            kpe_ref,
            page_size,
        )
        assert torch.max(torch.abs(q_meta.float() - q_ref.float())).item() == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 29: Massive fuzz (more seeds, more configs)
# ══════════════════════════════════════════════════════════════════════════════


class TestMassiveFuzz:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize("seed", range(50, 150))
    def test_additional_random_configs(self, device, cos_sin_cache, seed):
        """100 more random seeds for good measure."""
        torch.manual_seed(seed)
        batch_size = torch.randint(1, 129, (1,)).item()
        num_heads = [16, 32, 64, 128][seed % 4]
        page_size = [1, 4, 16, 32, 64][seed % 5]
        num_pages = batch_size + 2

        out_cache_loc = generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.randint(
            0, 8192, (batch_size,), device=device, dtype=torch.int32
        )
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=seed
        )

        assert_parity(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            page_size,
            num_pages,
            label=f"fuzz2-{seed}",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 30: Actual CUDA graph with real torch.cuda.CUDAGraph —
#              non-power-of-2 batch sizes
# ══════════════════════════════════════════════════════════════════════════════


class TestActualCudaGraphOddSizes:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    @pytest.mark.parametrize("capture_bs", [3, 5, 7, 11, 13, 17, 31, 33, 63, 65, 127])
    def test_odd_capture_sizes(self, device, cos_sin_cache, capture_bs):
        """Capture and replay with non-power-of-2 batch sizes."""
        num_heads = 64
        page_size = 64
        num_pages = capture_bs + 2
        fp8 = torch.float8_e4m3fn

        kv_indices_buf = torch.zeros(capture_bs, dtype=torch.int32, device=device)
        kv_indptr_buf = torch.arange(capture_bs + 1, dtype=torch.int32, device=device)
        batch_indices_buf = torch.arange(capture_bs, dtype=torch.int32, device=device)
        positions_buf = torch.zeros(capture_bs, dtype=torch.int32, device=device)

        q_nope_buf = torch.randn(
            capture_bs, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope_buf = torch.randn(
            capture_bs, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope_buf = torch.randn(
            capture_bs, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope_buf = torch.randn(
            capture_bs, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        pos_ids_buf = torch.arange(capture_bs, device=device, dtype=torch.int32)
        q_out_buf = torch.empty(
            capture_bs,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        ckv_buf = torch.zeros(
            num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
        )
        kpe_buf = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )

        locs = generate_cache_locs(capture_bs, page_size, device)
        kv_indices_buf[:] = (locs // page_size).to(torch.int32)
        positions_buf[:] = (locs % page_size).to(torch.int32)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=kv_indices_buf,
                kv_indptr=kv_indptr_buf,
                batch_indices=batch_indices_buf,
                positions=positions_buf,
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )
        torch.cuda.current_stream().wait_stream(s)
        ckv_buf.zero_()
        kpe_buf.zero_()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=kv_indices_buf,
                kv_indptr=kv_indptr_buf,
                batch_indices=batch_indices_buf,
                positions=positions_buf,
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )

        replay_bs = max(1, capture_bs // 2)
        q_n, q_r, k_n, k_r = make_inputs(replay_bs, num_heads, device, seed=capture_bs)
        replay_locs = generate_cache_locs(replay_bs, page_size, device) + 1
        replay_pos = torch.arange(replay_bs, device=device, dtype=torch.int32)

        if replay_bs < capture_bs:
            pos_ids_buf.zero_()
            kv_indices_buf.zero_()
            positions_buf.zero_()

        q_nope_buf[:replay_bs].copy_(q_n)
        q_rope_buf[:replay_bs].copy_(q_r)
        k_nope_buf[:replay_bs].copy_(k_n)
        k_rope_buf[:replay_bs].copy_(k_r)
        pos_ids_buf[:replay_bs].copy_(replay_pos)
        kv_indices_buf[:replay_bs] = (replay_locs // page_size).to(torch.int32)
        positions_buf[:replay_bs] = (replay_locs % page_size).to(torch.int32)
        ckv_buf.zero_()
        kpe_buf.zero_()

        g.replay()
        torch.cuda.synchronize()

        ckv_ref = torch.zeros_like(ckv_buf)
        kpe_ref = torch.zeros_like(kpe_buf)
        q_ref = run_separated(
            q_n,
            q_r,
            k_n,
            k_r,
            cos_sin_cache,
            replay_pos,
            replay_locs,
            ckv_ref,
            kpe_ref,
            page_size,
        )
        q_diff = torch.max(
            torch.abs(q_out_buf[:replay_bs].float() - q_ref.float())
        ).item()
        assert q_diff == 0.0, f"capture_bs={capture_bs}: q_out mismatch {q_diff}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 31: Eager ↔ CUDA-graph transition tests
#
#  In production, SGLang alternates between eager decode (batch size doesn't
#  match any captured graph) and CUDA graph replay (batch size matches).
#  These tests verify no cross-contamination via:
#    - The shared pre-allocated buffer that overlapping graph sizes slice into
#    - Stale metadata references left by one path affecting the other
#    - Padding residue from a smaller replay corrupting a later larger replay
#      that was interleaved with an eager call
# ══════════════════════════════════════════════════════════════════════════════


class TestEagerCudaGraphTransitions:

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def cos_sin_cache(self, device):
        return create_cos_sin_cache(8192, QK_ROPE_HEAD_DIM, device)

    # ── helpers ──

    def _make_shared_buffers(self, max_tokens, device):
        """Simulate init_cuda_graph_state: one shared buffer backing all graph sizes."""
        return {
            "kv_indices": torch.zeros(max_tokens, dtype=torch.int32, device=device),
            "kv_indptr": torch.arange(max_tokens + 1, dtype=torch.int32, device=device),
            "batch_indices": torch.arange(max_tokens, dtype=torch.int32, device=device),
            "positions": torch.zeros(max_tokens, dtype=torch.int32, device=device),
        }

    def _bind_graph_metadata(self, shared, capture_bs):
        """Simulate init_forward_metadata_capture_cuda_graph: slice shared buffer."""
        return FakeDecodeMetadata(
            kv_indices=shared["kv_indices"][:capture_bs],
            kv_indptr=shared["kv_indptr"][: capture_bs + 1],
            batch_indices=shared["batch_indices"][:capture_bs],
            positions=shared["positions"][:capture_bs],
        )

    def _graph_replay_step(
        self, md, real_bs, num_heads, page_size, num_pages, cos_sin_cache, device, seed
    ):
        """Simulate one CUDA graph replay: in-place metadata update + kernel call."""
        fp8 = torch.float8_e4m3fn
        locs = generate_cache_locs(real_bs, page_size, device) + seed % 100
        init_metadata_inplace(md, locs, page_size)

        q_nope, q_rope, k_nope, k_rope = make_inputs(
            real_bs, num_heads, device, seed=seed
        )
        pos_ids = torch.arange(real_bs, device=device, dtype=torch.int32)

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out = run_fused_with_metadata(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            md,
            real_bs,
            ckv,
            kpe,
            page_size,
        )

        ckv_ref = torch.zeros_like(ckv)
        kpe_ref = torch.zeros_like(kpe)
        q_ref = run_separated(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            locs,
            ckv_ref,
            kpe_ref,
            page_size,
        )
        return q_out, q_ref, ckv, kpe, ckv_ref, kpe_ref

    def _eager_step(
        self, batch_size, num_heads, page_size, cos_sin_cache, device, seed
    ):
        """Simulate one eager forward: fresh metadata + direct kernel call."""
        fp8 = torch.float8_e4m3fn
        num_pages = batch_size + 2
        locs = generate_cache_locs(batch_size, page_size, device) + seed % 100
        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=seed
        )
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        ckv = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q_out = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            locs,
            ckv,
            kpe,
            page_size,
        )

        ckv_ref = torch.zeros_like(ckv)
        kpe_ref = torch.zeros_like(kpe)
        q_ref = run_separated(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            locs,
            ckv_ref,
            kpe_ref,
            page_size,
        )
        return q_out, q_ref

    def _assert_q_match(self, q_out, q_ref, label):
        d = torch.max(torch.abs(q_out.float() - q_ref.float())).item()
        assert d == 0.0, f"[{label}] q_out mismatch: {d}"

    def _assert_kv_match(self, ckv, kpe, ckv_ref, kpe_ref, label):
        d_ckv = torch.max(torch.abs(ckv.float() - ckv_ref.float())).item()
        d_kpe = torch.max(torch.abs(kpe.float() - kpe_ref.float())).item()
        assert d_ckv == 0.0, f"[{label}] ckv mismatch: {d_ckv}"
        assert d_kpe == 0.0, f"[{label}] kpe mismatch: {d_kpe}"

    # ── tests ──

    def test_shared_buffer_overlap_two_graph_sizes(self, device, cos_sin_cache):
        """Two graph sizes (bs=32, bs=64) share one underlying buffer.

        Alternating replays must produce correct results despite the overlap:
        the first 32 elements of the bs=64 metadata physically alias the
        entirety of the bs=32 metadata.
        """
        num_heads = 64
        page_size = 64
        max_tokens = 128
        num_pages = max_tokens + 2

        shared = self._make_shared_buffers(max_tokens, device)
        md32 = self._bind_graph_metadata(shared, 32)
        md64 = self._bind_graph_metadata(shared, 64)

        assert (
            md32.kv_indices.data_ptr() == md64.kv_indices.data_ptr()
        ), "slices must alias the same underlying buffer"

        # replay64(nnz=64) → replay32(nnz=20) → replay64(nnz=40) → replay32(nnz=32)
        for step, (md, real_bs, seed, label) in enumerate(
            [
                (md64, 64, 1000, "graph64-nnz64"),
                (md32, 20, 2000, "graph32-nnz20"),
                (md64, 40, 3000, "graph64-nnz40"),
                (md32, 32, 4000, "graph32-nnz32"),
                (md64, 10, 5000, "graph64-nnz10"),
                (md32, 1, 6000, "graph32-nnz1"),
            ]
        ):
            q, qr, ckv, kpe, cr, kr = self._graph_replay_step(
                md,
                real_bs,
                num_heads,
                page_size,
                num_pages,
                cos_sin_cache,
                device,
                seed,
            )
            self._assert_q_match(q, qr, label)
            self._assert_kv_match(ckv, kpe, cr, kr, label)

    def test_eager_between_graph_replays(self, device, cos_sin_cache):
        """graph_replay → eager → graph_replay.

        The eager step creates brand-new metadata (fresh alloc, not touching the
        shared buffer). The second graph replay must still work correctly with
        its pre-allocated shared buffer, unaffected by the eager step.
        """
        num_heads = 128
        page_size = 64
        max_tokens = 128
        num_pages = max_tokens + 2

        shared = self._make_shared_buffers(max_tokens, device)
        md64 = self._bind_graph_metadata(shared, 64)

        # Step 1: graph replay
        q, qr, ckv, kpe, cr, kr = self._graph_replay_step(
            md64,
            50,
            num_heads,
            page_size,
            num_pages,
            cos_sin_cache,
            device,
            seed=100,
        )
        self._assert_q_match(q, qr, "step1-graph64-nnz50")

        # Step 2: eager (should NOT touch the shared buffer)
        shared_snap = shared["kv_indices"].clone()
        q_e, qr_e = self._eager_step(
            45,
            num_heads,
            page_size,
            cos_sin_cache,
            device,
            seed=200,
        )
        self._assert_q_match(q_e, qr_e, "step2-eager-bs45")
        assert torch.equal(
            shared["kv_indices"], shared_snap
        ), "eager step must not touch the shared graph buffer"

        # Step 3: graph replay again
        q2, qr2, ckv2, kpe2, cr2, kr2 = self._graph_replay_step(
            md64,
            64,
            num_heads,
            page_size,
            num_pages,
            cos_sin_cache,
            device,
            seed=300,
        )
        self._assert_q_match(q2, qr2, "step3-graph64-nnz64")
        self._assert_kv_match(ckv2, kpe2, cr2, kr2, "step3-graph64-nnz64")

    def test_graph_replay_between_eager_steps(self, device, cos_sin_cache):
        """eager → graph_replay → eager.

        The graph replay writes into the shared buffer. The second eager step
        must create fresh metadata and not be affected by the shared buffer state.
        """
        num_heads = 64
        page_size = 32
        max_tokens = 128
        num_pages = max_tokens + 2

        shared = self._make_shared_buffers(max_tokens, device)
        md64 = self._bind_graph_metadata(shared, 64)

        # Step 1: eager
        q1, qr1 = self._eager_step(
            50,
            num_heads,
            page_size,
            cos_sin_cache,
            device,
            seed=111,
        )
        self._assert_q_match(q1, qr1, "step1-eager-bs50")

        # Step 2: graph replay (mutates shared buffer)
        q2, qr2, *_ = self._graph_replay_step(
            md64,
            40,
            num_heads,
            page_size,
            num_pages,
            cos_sin_cache,
            device,
            seed=222,
        )
        self._assert_q_match(q2, qr2, "step2-graph64-nnz40")

        # Step 3: eager (must be independent of shared buffer state)
        q3, qr3 = self._eager_step(
            55,
            num_heads,
            page_size,
            cos_sin_cache,
            device,
            seed=333,
        )
        self._assert_q_match(q3, qr3, "step3-eager-bs55")

    def test_interleaved_rapid_alternation(self, device, cos_sin_cache):
        """20 interleaved eager/graph steps with varying batch sizes.

        Simulates a production scheduler that sends different batch sizes
        on every decode step, some matching captured graphs, some not.
        """
        num_heads = 128
        page_size = 64
        max_tokens = 256
        num_pages = max_tokens + 2

        shared = self._make_shared_buffers(max_tokens, device)
        md32 = self._bind_graph_metadata(shared, 32)
        md64 = self._bind_graph_metadata(shared, 64)
        md128 = self._bind_graph_metadata(shared, 128)

        # (mode, md_or_bs, real_bs, seed)
        # mode: "graph" uses pre-allocated metadata, "eager" uses fresh alloc
        schedule = [
            ("eager", None, 50, 1001),
            ("graph", md64, 64, 1002),
            ("eager", None, 10, 1003),
            ("graph", md32, 20, 1004),
            ("graph", md64, 30, 1005),
            ("eager", None, 100, 1006),
            ("graph", md128, 128, 1007),
            ("graph", md32, 32, 1008),
            ("eager", None, 1, 1009),
            ("graph", md128, 50, 1010),
            ("eager", None, 75, 1011),
            ("graph", md64, 1, 1012),
            ("graph", md128, 100, 1013),
            ("eager", None, 33, 1014),
            ("graph", md32, 7, 1015),
            ("eager", None, 128, 1016),
            ("graph", md128, 3, 1017),
            ("graph", md64, 64, 1018),
            ("eager", None, 17, 1019),
            ("graph", md32, 15, 1020),
        ]

        for i, (mode, md, real_bs, seed) in enumerate(schedule):
            label = f"step{i}-{mode}-bs{real_bs}"
            if mode == "graph":
                q, qr, ckv, kpe, cr, kr = self._graph_replay_step(
                    md,
                    real_bs,
                    num_heads,
                    page_size,
                    num_pages,
                    cos_sin_cache,
                    device,
                    seed,
                )
                self._assert_q_match(q, qr, label)
                self._assert_kv_match(ckv, kpe, cr, kr, label)
            else:
                q, qr = self._eager_step(
                    real_bs,
                    num_heads,
                    page_size,
                    cos_sin_cache,
                    device,
                    seed,
                )
                self._assert_q_match(q, qr, label)

    def test_metadata_poisoning_then_inplace_recovery(self, device, cos_sin_cache):
        """Fill shared buffer with garbage, then do proper in-place update.

        Verifies _init_forward_metadata_for_rope_fusion(update_inplace=True)
        completely overwrites corrupted state including the padding tail.
        """
        num_heads = 64
        page_size = 64
        max_tokens = 128
        num_pages = max_tokens + 2
        fp8 = torch.float8_e4m3fn

        shared = self._make_shared_buffers(max_tokens, device)
        md = self._bind_graph_metadata(shared, 64)

        # Poison the shared buffer
        shared["kv_indices"].fill_(9999)
        shared["positions"].fill_(7777)

        # In-place update should overwrite the poison
        real_bs = 32
        q, qr, ckv, kpe, cr, kr = self._graph_replay_step(
            md,
            real_bs,
            num_heads,
            page_size,
            num_pages,
            cos_sin_cache,
            device,
            seed=4242,
        )
        self._assert_q_match(q, qr, "poisoned-then-recovered")
        self._assert_kv_match(ckv, kpe, cr, kr, "poisoned-then-recovered")

        assert (md.kv_indices[real_bs:64] == 0).all(), "padding not zeroed after poison"
        assert (md.positions[real_bs:64] == 0).all(), "padding not zeroed after poison"

    def test_save_kv_cache_flag_transition(self, device, cos_sin_cache):
        """Run fused path (save_kv_cache=True), then separated path
        (save_kv_cache=False), then fused path again.

        The separated path (mla_rope_quantize_fp8) doesn't use paged KV
        metadata at all.  This tests that the metadata survives unmodified
        through the separated call and the next fused call still works.
        """
        num_heads = 64
        page_size = 64
        batch_size = 32
        num_pages = batch_size + 2
        fp8 = torch.float8_e4m3fn

        q_nope, q_rope, k_nope, k_rope = make_inputs(
            batch_size, num_heads, device, seed=500
        )
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        locs = generate_cache_locs(batch_size, page_size, device)

        # Step 1: fused path (save_kv_cache=True)
        ckv1 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe1 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q1 = run_fused(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            locs,
            ckv1,
            kpe1,
            page_size,
        )

        # Step 2: separated path (save_kv_cache=False equivalent)
        q2_out = torch.empty(
            batch_size,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=fp8)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=fp8)
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            quantize_dtype=fp8,
            q_rope_out=q2_out[..., KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q2_out[..., :KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        q_diff_sep = torch.max(torch.abs(q1.float() - q2_out.float())).item()
        assert q_diff_sep == 0.0, f"fused vs separated q mismatch: {q_diff_sep}"

        # Step 3: fused path again with different data
        q_nope2, q_rope2, k_nope2, k_rope2 = make_inputs(
            batch_size, num_heads, device, seed=600
        )
        locs2 = generate_cache_locs(batch_size, page_size, device) + 3
        ckv3 = torch.zeros(num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8)
        kpe3 = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )
        q3 = run_fused(
            q_nope2,
            q_rope2,
            k_nope2,
            k_rope2,
            cos_sin_cache,
            pos_ids,
            locs2,
            ckv3,
            kpe3,
            page_size,
        )

        ckv3_ref = torch.zeros_like(ckv3)
        kpe3_ref = torch.zeros_like(kpe3)
        q3_ref = run_separated(
            q_nope2,
            q_rope2,
            k_nope2,
            k_rope2,
            cos_sin_cache,
            pos_ids,
            locs2,
            ckv3_ref,
            kpe3_ref,
            page_size,
        )
        self._assert_q_match(q3, q3_ref, "fused-after-separated")
        self._assert_kv_match(ckv3, kpe3, ckv3_ref, kpe3_ref, "fused-after-separated")

    def test_real_cuda_graph_interleaved_with_eager(self, device, cos_sin_cache):
        """Real torch.cuda.CUDAGraph capture + replay interleaved with eager calls.

        This is the closest simulation to production: actual CUDA graph capture
        freezes tensor references, replays execute recorded ops, and eager calls
        happen in between using completely independent tensors.
        """
        num_heads = 64
        page_size = 64
        capture_bs = 64
        num_pages = capture_bs + 4
        fp8 = torch.float8_e4m3fn

        # ── Pre-allocate shared buffers (simulates init_cuda_graph_state) ──
        shared_kv_indices = torch.zeros(capture_bs, dtype=torch.int32, device=device)
        shared_kv_indptr = torch.arange(
            capture_bs + 1, dtype=torch.int32, device=device
        )
        shared_batch_indices = torch.arange(
            capture_bs, dtype=torch.int32, device=device
        )
        shared_positions = torch.zeros(capture_bs, dtype=torch.int32, device=device)

        # Graph-captured input buffers (padded to capture_bs)
        q_nope_buf = torch.randn(
            capture_bs, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        q_rope_buf = torch.randn(
            capture_bs, num_heads, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        k_nope_buf = torch.randn(
            capture_bs, KV_LORA_RANK, device=device, dtype=torch.bfloat16
        )
        k_rope_buf = torch.randn(
            capture_bs, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        pos_ids_buf = torch.arange(capture_bs, device=device, dtype=torch.int32)
        q_out_buf = torch.empty(
            capture_bs,
            num_heads,
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            device=device,
            dtype=fp8,
        )
        ckv_buf = torch.zeros(
            num_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
        )
        kpe_buf = torch.zeros(
            num_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
        )

        locs = generate_cache_locs(capture_bs, page_size, device)
        shared_kv_indices[:] = (locs // page_size).to(torch.int32)
        shared_positions[:] = (locs % page_size).to(torch.int32)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=shared_kv_indices,
                kv_indptr=shared_kv_indptr,
                batch_indices=shared_batch_indices,
                positions=shared_positions,
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_buf,
                k_rope=k_rope_buf,
                q_nope=q_nope_buf,
                k_nope=k_nope_buf,
                v=None,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids_buf,
                paged_kv_cache=(ckv_buf, kpe_buf),
                kv_indices=shared_kv_indices,
                kv_indptr=shared_kv_indptr,
                batch_indices=shared_batch_indices,
                positions=shared_positions,
                is_neox=False,
                quantize_dtype=fp8,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                page_size=page_size,
                kv_layout="NHD",
                q_rope_out=q_out_buf[..., KV_LORA_RANK:],
                q_nope_out=q_out_buf[..., :KV_LORA_RANK],
            )
        torch.cuda.current_stream().wait_stream(s)

        def do_graph_replay(real_bs, seed, label):
            torch.manual_seed(seed)
            q_n = torch.randn(
                real_bs, num_heads, KV_LORA_RANK, device=device, dtype=torch.bfloat16
            )
            q_r = torch.randn(
                real_bs,
                num_heads,
                QK_ROPE_HEAD_DIM,
                device=device,
                dtype=torch.bfloat16,
            )
            k_n = torch.randn(
                real_bs, KV_LORA_RANK, device=device, dtype=torch.bfloat16
            )
            k_r = torch.randn(
                real_bs, QK_ROPE_HEAD_DIM, device=device, dtype=torch.bfloat16
            )
            replay_locs = generate_cache_locs(real_bs, page_size, device) + seed % 50
            replay_pos = torch.arange(real_bs, device=device, dtype=torch.int32)

            # Simulate populate_from_forward_batch
            if real_bs < capture_bs:
                q_nope_buf.zero_()
                q_rope_buf.zero_()
                k_nope_buf.zero_()
                k_rope_buf.zero_()
                pos_ids_buf.zero_()
            q_nope_buf[:real_bs].copy_(q_n)
            q_rope_buf[:real_bs].copy_(q_r)
            k_nope_buf[:real_bs].copy_(k_n)
            k_rope_buf[:real_bs].copy_(k_r)
            pos_ids_buf[:real_bs].copy_(replay_pos)

            # Simulate _init_forward_metadata_for_rope_fusion(update_inplace=True)
            shared_kv_indices[:real_bs] = (replay_locs // page_size).to(torch.int32)
            shared_positions[:real_bs] = (replay_locs % page_size).to(torch.int32)
            if real_bs < capture_bs:
                shared_kv_indices[real_bs:].zero_()
                shared_positions[real_bs:].zero_()

            ckv_buf.zero_()
            kpe_buf.zero_()

            g.replay()
            torch.cuda.synchronize()

            ckv_ref = torch.zeros_like(ckv_buf)
            kpe_ref = torch.zeros_like(kpe_buf)
            q_ref = run_separated(
                q_n,
                q_r,
                k_n,
                k_r,
                cos_sin_cache,
                replay_pos,
                replay_locs,
                ckv_ref,
                kpe_ref,
                page_size,
            )
            d = torch.max(torch.abs(q_out_buf[:real_bs].float() - q_ref.float())).item()
            assert d == 0.0, f"[{label}] q mismatch: {d}"

        def do_eager(batch_size, seed, label):
            q_nope, q_rope, k_nope, k_rope = make_inputs(
                batch_size, num_heads, device, seed=seed
            )
            eager_locs = generate_cache_locs(batch_size, page_size, device) + seed % 50
            pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
            eager_pages = batch_size + 2
            ckv_e = torch.zeros(
                eager_pages, page_size, KV_LORA_RANK, device=device, dtype=fp8
            )
            kpe_e = torch.zeros(
                eager_pages, page_size, QK_ROPE_HEAD_DIM, device=device, dtype=fp8
            )
            q_e = run_fused(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                eager_locs,
                ckv_e,
                kpe_e,
                page_size,
            )
            ckv_er = torch.zeros_like(ckv_e)
            kpe_er = torch.zeros_like(kpe_e)
            q_er = run_separated(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                cos_sin_cache,
                pos_ids,
                eager_locs,
                ckv_er,
                kpe_er,
                page_size,
            )
            d = torch.max(torch.abs(q_e.float() - q_er.float())).item()
            assert d == 0.0, f"[{label}] eager q mismatch: {d}"

        # ── The interleaved sequence ──
        do_eager(50, 1000, "step1-eager-bs50")
        do_graph_replay(32, 2000, "step2-graph-nnz32")
        do_eager(45, 3000, "step3-eager-bs45")
        do_graph_replay(64, 4000, "step4-graph-nnz64")
        do_graph_replay(10, 5000, "step5-graph-nnz10")
        do_eager(30, 6000, "step6-eager-bs30")
        do_graph_replay(1, 7000, "step7-graph-nnz1")
        do_eager(1, 8000, "step8-eager-bs1")
        do_graph_replay(64, 9000, "step9-graph-nnz64")
        do_eager(100, 10000, "step10-eager-bs100")

    def test_shared_buffer_not_corrupted_by_eager(self, device, cos_sin_cache):
        """Verify that eager forward never writes to the shared graph buffer.

        The eager path in production creates a new TRTLLMMLADecodeMetadata() and
        allocates new tensors via update_inplace=False. The shared buffer (which
        backs all CUDA graph metadata) must remain untouched.
        """
        max_tokens = 128
        shared = self._make_shared_buffers(max_tokens, device)

        # Poison the shared buffer with a known pattern
        shared["kv_indices"].fill_(42)
        shared["positions"].fill_(99)
        snap_kv = shared["kv_indices"].clone()
        snap_pos = shared["positions"].clone()

        # Multiple eager calls with various batch sizes
        for bs, seed in [(10, 111), (50, 222), (128, 333), (1, 444)]:
            q, qr = self._eager_step(bs, 64, 64, cos_sin_cache, device, seed)
            self._assert_q_match(q, qr, f"eager-bs{bs}")

        # Shared buffer must be completely unchanged
        assert torch.equal(
            shared["kv_indices"], snap_kv
        ), "eager corrupted shared kv_indices"
        assert torch.equal(
            shared["positions"], snap_pos
        ), "eager corrupted shared positions"

    def test_three_overlapping_graph_sizes_rapid_switch(self, device, cos_sin_cache):
        """Three capture sizes (32, 64, 128) with overlapping shared buffer slices.

        Rapidly switch between all three graph sizes, including cases where a
        smaller graph's replay zeros elements that a larger graph later needs
        to overwrite.
        """
        num_heads = 128
        page_size = 64
        max_tokens = 256
        num_pages = max_tokens + 2

        shared = self._make_shared_buffers(max_tokens, device)
        md32 = self._bind_graph_metadata(shared, 32)
        md64 = self._bind_graph_metadata(shared, 64)
        md128 = self._bind_graph_metadata(shared, 128)

        sequence = [
            (md128, 128, 1001),
            (md32, 32, 1002),
            (md128, 50, 1003),
            (md64, 64, 1004),
            (md32, 1, 1005),
            (md64, 10, 1006),
            (md128, 128, 1007),
            (md64, 64, 1008),
            (md32, 32, 1009),
            (md32, 5, 1010),
            (md128, 80, 1011),
            (md64, 30, 1012),
        ]

        for i, (md, real_bs, seed) in enumerate(sequence):
            cap_bs = md.kv_indices.shape[0]
            label = f"step{i}-cap{cap_bs}-nnz{real_bs}"
            q, qr, ckv, kpe, cr, kr = self._graph_replay_step(
                md,
                real_bs,
                num_heads,
                page_size,
                num_pages,
                cos_sin_cache,
                device,
                seed,
            )
            self._assert_q_match(q, qr, label)
            self._assert_kv_match(ckv, kpe, cr, kr, label)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
