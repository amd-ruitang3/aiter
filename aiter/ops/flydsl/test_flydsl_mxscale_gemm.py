# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL gfx1250 MXScale dense GEMM (fp8 + a8w4).

Module skipped when CUDA/ROCm is unavailable, when flydsl is not installed,
or when the runtime arch is not gfx1250.

Usage:
    pytest -q aiter/ops/flydsl/test_flydsl_mxscale_gemm.py
"""

from __future__ import annotations

import os

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL mxscale tests.",
        allow_module_level=True,
    )

try:
    from flydsl.runtime.device import get_rocm_arch
except ImportError as exc:
    pytest.skip(f"Unable to import flydsl runtime: {exc}", allow_module_level=True)

if str(get_rocm_arch()) != "gfx1250":
    pytest.skip(
        f"WMMA_SCALE requires gfx1250, got {get_rocm_arch()!r}",
        allow_module_level=True,
    )

from aiter.ops.flydsl.mxscale_gemm import (  # noqa: E402
    flydsl_mxscale_gemm,
    flydsl_mxscale_kernel_name,
    gemm_mxa8w4,
    gemm_mxfp8,
    parse_flydsl_mxscale_kernel_name,
)
from aiter.ops.flydsl.mxscale_layout import (  # noqa: E402
    SCALE_BLOCK,
    get_padded_problem_shape,
    pad_mxscale_inputs,
    preshuffle_b_16x16,
    preshuffle_e8m0_scale_wmma,
    recommended_num_buffers,
)


@pytest.fixture(scope="session", autouse=True)
def _release_flydsl_mxscale_cache():
    yield
    torch.cuda.synchronize()
    from aiter.ops.flydsl.kernels.gemm_fp8fp4_gfx1250 import (  # noqa: PLC0415
        compile_mxscale_gemm,
    )

    compile_mxscale_gemm.cache_clear()


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _fp8_e4m3_to_f32(x_uint8: torch.Tensor) -> torch.Tensor:
    """Decode a uint8 byte stream as float8_e4m3fn → float32."""
    return x_uint8.view(torch.float8_e4m3fn).to(torch.float32)


def _ref_mxscale_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
    *,
    is_a8w4: bool,
) -> torch.Tensor:
    a_f32 = _fp8_e4m3_to_f32(a.view(torch.uint8))[:M, :K]
    if is_a8w4:
        b_f32 = mxfp4_to_f32(b.view(torch.uint8))[:N, :K]
    else:
        b_f32 = _fp8_e4m3_to_f32(b.view(torch.uint8))[:N, :K]
    a_sc = e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


def _random_fp8_bytes(rows: int, cols: int, max_byte: int = 126) -> torch.Tensor:
    # Avoid NaN / inf encodings (0x7F / 0xFF for e4m3fn).
    # max_byte<=64 keeps decoded values strictly < 1.0 (bytes 0..63 cover
    # subnormals + exponents 1..7), useful for f16-output overflow safety.
    return torch.randint(0, max_byte, (rows, cols), dtype=torch.uint8)


def _random_fp4_packed(rows: int, cols: int) -> torch.Tensor:
    return torch.randint(0, 256, (rows, cols), dtype=torch.uint8)


def _random_e8m0(rows: int, cols: int, low: int = 124, high: int = 130) -> torch.Tensor:
    # Restrict the exponent range so we don't blow up the FP32 accumulator
    # (kernel still uses E8M0(127) padding internally).
    return torch.randint(low, high, (rows, cols), dtype=torch.uint8)


def _gen_inputs_for_dtype(M, N, K, *, out_dtype, is_a8w4):
    """Pick FP8/scale random ranges that keep the output within out_dtype range.

    f16 max = 65504 — with K up to 7168 and FP8 max ~448, default ranges easily
    overflow on downcast. Tighten to bytes <= 0x40 (decoded < 1.0) and
    scale exponent 127 (=1.0) so the accumulator stays well below 65504.
    """
    if out_dtype == "f16":
        a = _random_fp8_bytes(M, K, max_byte=0x40)
        if is_a8w4:
            b = _random_fp4_packed(N, K // 2)
        else:
            b = _random_fp8_bytes(N, K, max_byte=0x40)
        a_s = _random_e8m0(M, K // SCALE_BLOCK, low=127, high=128)
        b_s = _random_e8m0(N, K // SCALE_BLOCK, low=127, high=128)
    else:
        a = _random_fp8_bytes(M, K)
        if is_a8w4:
            b = _random_fp4_packed(N, K // 2)
        else:
            b = _random_fp8_bytes(N, K)
        a_s = _random_e8m0(M, K // SCALE_BLOCK)
        b_s = _random_e8m0(N, K // SCALE_BLOCK)
    return a, b, a_s, b_s


# ---------------------------------------------------------------------------
# Helper unit tests (no GPU work)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,M,N,K,tile_m,tile_n,tile_k,split_k,want",
    [
        (
            "fp8",
            128,
            256,
            256,
            128,
            128,
            128,
            1,
            {"M": 128, "N": 256, "K": 256, "K_scale": 8, "pack_a": 1, "pack_b": 1},
        ),
        (
            "a8w4",
            13,
            17,
            256,
            128,
            128,
            128,
            2,
            {"M": 128, "N": 128, "K": 256, "K_scale": 8, "pack_a": 1, "pack_b": 2},
        ),
        (
            "fp8",
            1,
            1,
            64,
            128,
            128,
            128,
            1,
            {"M": 128, "N": 128, "K": 128, "K_scale": 4, "pack_a": 1, "pack_b": 1},
        ),
    ],
)
def test_padded_problem_shape(fmt, M, N, K, tile_m, tile_n, tile_k, split_k, want):
    if K % SCALE_BLOCK != 0:
        pytest.skip("test config has K not divisible by SCALE_BLOCK")
    got = get_padded_problem_shape(
        fmt, M, N, K, tile_m, tile_n, tile_k, split_k=split_k
    )
    assert got == want


def test_pad_mxscale_inputs_data_zero_scale_127():
    fmt = "fp8"
    padded = get_padded_problem_shape(fmt, 13, 17, 64, 128, 128, 128, split_k=1)
    a = torch.zeros((13, 64), dtype=torch.uint8) + 5
    b = torch.zeros((17, 64), dtype=torch.uint8) + 7
    a_scale = torch.zeros((13, 2), dtype=torch.uint8) + 200
    b_scale = torch.zeros((17, 2), dtype=torch.uint8) + 200
    a_p, b_p, a_sp, b_sp = pad_mxscale_inputs(a, b, a_scale, b_scale, padded)
    # Original data preserved.
    assert torch.equal(a_p[:13, :64], a)
    assert torch.equal(b_p[:17, :64], b)
    assert torch.equal(a_sp[:13, :2], a_scale)
    assert torch.equal(b_sp[:17, :2], b_scale)
    # Padding regions: data → 0, scale → 127.
    assert (a_p[13:, :] == 0).all()
    assert (b_p[17:, :] == 0).all()
    assert (a_sp[13:, :] == 127).all()
    assert (b_sp[17:, :] == 127).all()


def test_preshuffle_b_round_trip_size():
    rows, cols = 64, 64
    b = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    out = preshuffle_b_16x16(b, rows, cols)
    assert out.shape == (rows, cols)
    assert out.dtype == torch.uint8
    # Permutation is byte-shuffle: same multiset of values.
    assert sorted(out.flatten().tolist()) == sorted(b.flatten().tolist())


def test_preshuffle_e8m0_scale_wmma_size():
    M, K_scale = 64, 8
    s = torch.randint(120, 130, (M, K_scale), dtype=torch.uint8)
    warp_tile = 64
    out = preshuffle_e8m0_scale_wmma(s, warp_tile, scale_k_per_tile=4)
    assert out.numel() == s.numel()
    assert out.dtype == torch.uint8


# ---------------------------------------------------------------------------
# kernelName encode / parse round-trip
# ---------------------------------------------------------------------------


def test_kernel_name_round_trip_fp8():
    cfg = dict(
        data_format="fp8",
        out_dtype="bf16",
        tile_m=128,
        tile_n=128,
        tile_k=128,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        split_k=1,
        use_tdm_store=True,
        use_scale_opsel=False,
        wave_specialized_tdm=False,
        l2_prefetch_distance=2,
        cluster_m=1,
        cluster_n=1,
        waves_per_eu=0,
    )
    name = flydsl_mxscale_kernel_name(**cfg)
    assert name == (
        "flydsl_mxscale_fp8_bf16_t128x128x128_mw2_nw2_buf2_sk1_"
        "tdms1_opsel0_wst0_l2pf2_cm1_cn1_wpe0_gfx1250"
    )
    parsed = parse_flydsl_mxscale_kernel_name(name)
    assert parsed["kind"] == "mxscale"
    for k, v in cfg.items():
        assert parsed[k] == v


def test_kernel_name_round_trip_a8w4():
    cfg = dict(
        data_format="a8w4",
        out_dtype="f32",
        tile_m=128,
        tile_n=128,
        tile_k=256,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        split_k=2,
        use_tdm_store=False,
        use_scale_opsel=True,
        wave_specialized_tdm=False,
        l2_prefetch_distance=4,
        cluster_m=2,
        cluster_n=2,
        waves_per_eu=2,
    )
    name = flydsl_mxscale_kernel_name(**cfg)
    parsed = parse_flydsl_mxscale_kernel_name(name)
    assert parsed["kind"] == "mxscale"
    for k, v in cfg.items():
        assert parsed[k] == v


def test_kernel_name_parser_rejects_other_families():
    assert (
        parse_flydsl_mxscale_kernel_name(
            "flydsl_gemm2_abf16_wbf16_bf16_t32x64x128_split_k1_block_m_warp1_block_n_warp1_async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950"
        )
        is None
    )
    assert (
        parse_flydsl_mxscale_kernel_name(
            "flydsl_bpreshuflle_128x128x128_F8_F8_B16_2x0x0x0_v3"
        )
        is None
    )
    assert parse_flydsl_mxscale_kernel_name("not_a_kernel") is None


# ---------------------------------------------------------------------------
# Negative-path tests (no GPU work)
# ---------------------------------------------------------------------------


def test_invalid_data_format_raises():
    a = torch.zeros((1, 32), dtype=torch.uint8)
    b = torch.zeros((1, 32), dtype=torch.uint8)
    s = torch.zeros((1, 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="data_format"):
        flydsl_mxscale_gemm(a, b, s, s, data_format="fp4")


def test_format_named_wrappers_reject_data_format_kwarg():
    a = torch.zeros((1, 32), dtype=torch.uint8)
    b = torch.zeros((1, 32), dtype=torch.uint8)
    s = torch.zeros((1, 1), dtype=torch.uint8)
    with pytest.raises(TypeError, match="data_format"):
        gemm_mxfp8(a, b, s, s, data_format="fp8")
    with pytest.raises(TypeError, match="data_format"):
        gemm_mxa8w4(a, b, s, s, data_format="a8w4")


# ---------------------------------------------------------------------------
# GPU correctness tests (gfx1250 + flydsl required, gated above)
# ---------------------------------------------------------------------------


_SHAPES = [
    (128, 256, 256),
    (256, 256, 256),
    (128, 512, 7168),
    (13, 17, 256),  # exercise pad path
]
_OUT_DTYPES = ["bf16", "f16", "f32"]
_RUN_DEEPSEEK_SHAPES = os.getenv("AITER_FLYDSL_MXSCALE_RUN_DEEPSEEK_SHAPES", "0") == "1"
_DEEPSEEK_REFERENCE_MAX_M = 64

_DEEPSEEK_FP8_SHAPES = [
    # M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, table_num_buffers
    (1, 256, 7168, 16, 256, 256, 1, 4, 4),
    (1, 512, 7168, 16, 256, 256, 1, 4, 4),
    (1, 2112, 7168, 16, 256, 256, 1, 4, 4),
    (1, 3072, 1536, 16, 256, 256, 1, 4, 4),
    (1, 4096, 512, 16, 256, 256, 1, 4, 4),
    (1, 7168, 2048, 16, 256, 256, 1, 4, 4),
    (64, 256, 7168, 16, 256, 256, 1, 4, 4),
    (64, 512, 7168, 16, 256, 256, 1, 4, 4),
    (64, 2112, 7168, 16, 256, 256, 1, 4, 4),
    (64, 3072, 1536, 16, 256, 256, 1, 4, 4),
    (64, 4096, 512, 16, 256, 256, 1, 4, 4),
    (64, 7168, 2048, 16, 256, 256, 1, 4, 4),
    (1024, 256, 7168, 256, 256, 128, 2, 2, 4),
    (1024, 512, 7168, 256, 256, 128, 2, 2, 4),
    (1024, 2112, 7168, 256, 256, 128, 2, 2, 4),
    (1024, 3072, 1536, 256, 256, 128, 2, 2, 4),
    (1024, 4096, 512, 256, 256, 128, 2, 2, 4),
    (1024, 7168, 2048, 256, 256, 128, 2, 2, 4),
    (65536, 256, 7168, 256, 256, 128, 2, 2, 4),
    (65536, 512, 7168, 256, 256, 128, 2, 2, 4),
    (65536, 2112, 7168, 256, 256, 128, 2, 2, 4),
    (65536, 3072, 1536, 256, 256, 128, 2, 2, 4),
    (65536, 4096, 512, 256, 256, 128, 2, 2, 4),
    (65536, 7168, 2048, 256, 256, 128, 2, 2, 4),
]


def _deepseek_shape_id(case):
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, _ = case
    return f"M{M}_N{N}_K{K}_t{tile_m}x{tile_n}x{tile_k}_" f"w{m_warp}x{n_warp}"


def _deepseek_shape_params():
    params = []
    for case in _DEEPSEEK_FP8_SHAPES:
        marks = []
        if not _RUN_DEEPSEEK_SHAPES:
            marks.append(
                pytest.mark.skip(
                    "DeepSeek MXScale shape; set "
                    "AITER_FLYDSL_MXSCALE_RUN_DEEPSEEK_SHAPES=1 to run"
                )
            )
        params.append(pytest.param(case, id=_deepseek_shape_id(case), marks=marks))
    return params


def _effective_num_buffers(K: int, tile_k: int, table_num_buffers: int) -> int:
    suggestion = recommended_num_buffers(K, tile_k)
    if suggestion is None:
        raise ValueError(f"no supported num_buffers for K={K}, tile_k={tile_k}")
    return min(table_num_buffers, suggestion)


@pytest.mark.parametrize("M,N,K", _SHAPES)
@pytest.mark.parametrize("out_dtype", _OUT_DTYPES)
@pytest.mark.parametrize("use_tdm_store", [True, False])
def test_mxfp8_gemm_correctness(M, N, K, out_dtype, use_tdm_store):
    if K % SCALE_BLOCK != 0:
        pytest.skip("K must be a multiple of SCALE_BLOCK")
    torch.manual_seed(0)
    a, b, a_s, b_s = _gen_inputs_for_dtype(M, N, K, out_dtype=out_dtype, is_a8w4=False)

    ref = _ref_mxscale_gemm(a, b, a_s, b_s, M, N, K, is_a8w4=False)

    out = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="fp8",
        out_dtype=out_dtype,
        use_tdm_store=use_tdm_store,
    )
    out_f = out.float().cpu()
    ref_f = ref.float()

    if out_dtype in ("bf16", "f16"):
        torch.testing.assert_close(out_f, ref_f, rtol=2e-2, atol=5e-2)
    else:
        atol = max(1e-2, K * 0.6)
        torch.testing.assert_close(out_f, ref_f, rtol=1e-3, atol=atol)


@pytest.mark.parametrize("case", _deepseek_shape_params())
def test_mxfp8_deepseek_shapes(case):
    (
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        table_num_buffers,
    ) = case
    torch.manual_seed(11)
    a = _random_fp8_bytes(M, K, max_byte=0x40)
    b = _random_fp8_bytes(N, K, max_byte=0x40)
    a_s = _random_e8m0(M, K // SCALE_BLOCK, low=127, high=128)
    b_s = _random_e8m0(N, K // SCALE_BLOCK, low=127, high=128)
    num_buffers = _effective_num_buffers(K, tile_k, table_num_buffers)

    try:
        a_dev = a.cuda()
        b_dev = b.cuda()
        a_s_dev = a_s.cuda()
        b_s_dev = b_s.cuda()
        out = flydsl_mxscale_gemm(
            a_dev,
            b_dev,
            a_s_dev,
            b_s_dev,
            data_format="fp8",
            out_dtype="bf16",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            num_buffers=num_buffers,
            cluster_m=1,
            cluster_n=1,
            wave_specialized_tdm=True,
        )
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16
        if M <= _DEEPSEEK_REFERENCE_MAX_M:
            ref = _ref_mxscale_gemm(
                a_dev, b_dev, a_s_dev, b_s_dev, M, N, K, is_a8w4=False
            )
            torch.testing.assert_close(
                out.float().cpu(), ref.float().cpu(), rtol=2e-2, atol=5e-2
            )
        torch.cuda.synchronize()
    finally:
        torch.cuda.empty_cache()


@pytest.mark.parametrize("M,N,K", _SHAPES)
@pytest.mark.parametrize("out_dtype", _OUT_DTYPES)
def test_a8w4_gemm_correctness(M, N, K, out_dtype):
    if K % SCALE_BLOCK != 0:
        pytest.skip("K must be a multiple of SCALE_BLOCK")
    torch.manual_seed(0)
    a, b, a_s, b_s = _gen_inputs_for_dtype(M, N, K, out_dtype=out_dtype, is_a8w4=True)

    ref = _ref_mxscale_gemm(a, b, a_s, b_s, M, N, K, is_a8w4=True)

    out = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="a8w4",
        out_dtype=out_dtype,
        # A8W4 is sensitive to scale range; use coarse but adequate tile.
    )
    out_f = out.float().cpu()
    ref_f = ref.float()

    # Coarse mixed-precision tolerance — see FlyDSL test _a8w4_tolerances for
    # the scale-range-aware variant; here scales are tightly bounded so a
    # constant tolerance is fine.
    if out_dtype in ("bf16", "f16"):
        torch.testing.assert_close(out_f, ref_f, rtol=5e-2, atol=K * 0.5)
    else:
        atol = max(1e-2, K * 1.5)
        torch.testing.assert_close(out_f, ref_f, rtol=1e-2, atol=atol)


def test_split_k_runs():
    """split_k=2 forces buffer-store + zero-fill; smoke-test it runs and is close to ref."""
    M, N, K = 128, 256, 512
    torch.manual_seed(1)
    a = _random_fp8_bytes(M, K)
    b = _random_fp8_bytes(N, K)
    a_s = _random_e8m0(M, K // SCALE_BLOCK)
    b_s = _random_e8m0(N, K // SCALE_BLOCK)
    ref = _ref_mxscale_gemm(a, b, a_s, b_s, M, N, K, is_a8w4=False)
    out = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="fp8",
        out_dtype="bf16",
        split_k=2,
    )
    torch.testing.assert_close(out.float().cpu(), ref.float(), rtol=2e-2, atol=5e-2)


def test_format_named_wrappers_match_low_level():
    """gemm_mxfp8 / gemm_mxa8w4 must produce the same numeric output as the
    low-level flydsl_mxscale_gemm with the corresponding ``data_format``."""
    M, N, K = 128, 256, 256
    torch.manual_seed(7)

    # MXFP8 path
    a, b, a_s, b_s = _gen_inputs_for_dtype(M, N, K, out_dtype="bf16", is_a8w4=False)
    ll = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="fp8",
        out_dtype="bf16",
    )
    hl = gemm_mxfp8(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        out_dtype="bf16",
    )
    torch.testing.assert_close(hl.float().cpu(), ll.float().cpu(), rtol=0, atol=0)

    # MXA8W4 path
    a, b, a_s, b_s = _gen_inputs_for_dtype(M, N, K, out_dtype="bf16", is_a8w4=True)
    ll = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="a8w4",
        out_dtype="bf16",
    )
    hl = gemm_mxa8w4(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        out_dtype="bf16",
    )
    torch.testing.assert_close(hl.float().cpu(), ll.float().cpu(), rtol=0, atol=0)


def test_user_provided_out_buffer():
    M, N, K = 128, 256, 256
    torch.manual_seed(2)
    a = _random_fp8_bytes(M, K)
    b = _random_fp8_bytes(N, K)
    a_s = _random_e8m0(M, K // SCALE_BLOCK)
    b_s = _random_e8m0(N, K // SCALE_BLOCK)
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    returned = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="fp8",
        out=out,
    )
    assert returned.data_ptr() == out.data_ptr()
    assert returned.shape == (M, N)


def test_irregular_shape_no_overflow():
    """Pad path: M=13, N=17, K=160 — output buffer must remain (M, N)."""
    M, N, K = 13, 17, 160
    torch.manual_seed(3)
    a = _random_fp8_bytes(M, K)
    b = _random_fp8_bytes(N, K)
    a_s = _random_e8m0(M, K // SCALE_BLOCK)
    b_s = _random_e8m0(N, K // SCALE_BLOCK)
    out = flydsl_mxscale_gemm(
        a.cuda(),
        b.cuda(),
        a_s.cuda(),
        b_s.cuda(),
        data_format="fp8",
        out_dtype="bf16",
    )
    assert out.shape == (M, N)
