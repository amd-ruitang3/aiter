# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side layout helpers for the FlyDSL gfx1250 MXScale dense GEMM kernel.

These helpers prepare A / B / scale tensors into the exact layout the
``compile_mxscale_gemm(...)`` kernel expects: padded to tile boundaries,
B preshuffled into 16x16 byte tiles, E8M0 scales preshuffled for the
WMMA scale-fragment access pattern.

Kept separate from ``aiter.utility.fp4_utils`` because:
  * ``utility/fp4_utils.py`` must not depend on FlyDSL.
  * The preshuffle layouts here are FlyDSL-WMMA private conventions
    (distinct from ``e8m0_shuffle`` which targets the ASM 256x8 layout).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# Compile-time constants of the gfx1250 mxscale kernel; see
# kernels/gemm_fp8fp4_gfx1250.py: SCALE_BLOCK / WMMA_M.
SCALE_BLOCK: int = 32
WMMA_DIM: int = 16
SCALES_PER_WMMA: int = 4

# The kernel pads scale rows with E8M0(127) which decodes to 2^0 = 1.0
# so padded contributions accumulate to zero (data is padded with 0).
E8M0_ONE: int = 127


def mxscale_pack_factors(data_format: str) -> Tuple[int, int]:
    """Return (PACK_FACTOR_A, PACK_FACTOR_B) for the given data_format.

    fp8:  A FP8 (1 value/byte), B FP8 (1 value/byte) -> (1, 1)
    a8w4: A FP8 (1 value/byte), B FP4 (2 values/byte) -> (1, 2)
    fp4:  A FP4 (2 values/byte), B FP4 (2 values/byte) -> (2, 2)
    """
    if data_format == "fp8":
        return 1, 1
    if data_format == "a8w4":
        return 1, 2
    if data_format == "fp4":
        return 2, 2
    raise ValueError(f"unsupported data_format={data_format!r}")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def get_padded_problem_shape(
    data_format: str,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int = 1,
) -> dict:
    """Pad runtime (M, N, K) so the kernel always runs full tiles.

    K is padded to ``tile_k * split_k`` boundary because each split-K chunk
    must be a multiple of tile_k. M and N are padded to tile_m / tile_n.
    """
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    pack_a, pack_b = mxscale_pack_factors(data_format)
    padded_k = _align_up(K, tile_k * split_k)
    return {
        "M": _align_up(M, tile_m),
        "N": _align_up(N, tile_n),
        "K": padded_k,
        "K_scale": padded_k // SCALE_BLOCK,
        "pack_a": pack_a,
        "pack_b": pack_b,
    }


def _pad_2d(t: Tensor, rows: int, cols: int, fill_value: int) -> Tensor:
    if t.shape == (rows, cols):
        return t
    padded = torch.full(
        (rows, cols), fill_value, dtype=t.dtype, device=t.device
    )
    padded[: t.shape[0], : t.shape[1]] = t
    return padded


def pad_mxscale_inputs(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    padded_shape: dict,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad A / B / A_scale / B_scale to padded_shape boundaries.

    Data is padded with 0 (decodes to 0); scales are padded with
    E8M0(127) = 2^0 = 1.0 so zero data contributes zero to the
    accumulator regardless of scale.
    """
    a = _pad_2d(
        a, padded_shape["M"], padded_shape["K"] // padded_shape["pack_a"], fill_value=0
    )
    b = _pad_2d(
        b, padded_shape["N"], padded_shape["K"] // padded_shape["pack_b"], fill_value=0
    )
    a_scale = _pad_2d(
        a_scale, padded_shape["M"], padded_shape["K_scale"], fill_value=E8M0_ONE
    )
    b_scale = _pad_2d(
        b_scale, padded_shape["N"], padded_shape["K_scale"], fill_value=E8M0_ONE
    )
    return a, b, a_scale, b_scale


def preshuffle_b_16x16(b: Tensor, rows: int, cols: int) -> Tensor:
    """Preshuffle B into 16x16 byte tiles for WMMA-friendly LDS loads.

    Works for both FP8 (cols = K) and FP4 (cols = K // 2).
    Vendored from FlyDSL/tests/kernels/utils/fp4_utils.py:preshuffle_b_16x16.
    """
    assert rows % 16 == 0, f"rows must be a multiple of 16, got {rows}"
    assert cols % 16 == 0, f"cols must be a multiple of 16, got {cols}"
    b = b.view(rows, cols)
    b = b.view(rows // 16, 16, cols // 16, 16)
    b = b.permute(0, 2, 1, 3).contiguous()
    return b.view(rows, cols)


def preshuffle_e8m0_scale_wmma(
    scale: Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
    wmma_dim: int = WMMA_DIM,
) -> Tensor:
    """Preshuffle E8M0 scale into the layout consumed by WMMA scale fragments.

    Distinct from ``aiter.utility.fp4_utils.e8m0_shuffle`` which targets a
    different (ASM 256x8) layout. Vendored from
    FlyDSL/tests/kernels/test_gemm_fp8fp4_gfx1250.py:preshuffle_e8m0_scale.
    """
    _, k_scale = scale.shape
    assert k_scale % 4 == 0, f"K_scale must be divisible by 4, got {k_scale}"
    wmma_rep = warp_tile // wmma_dim
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, wmma_dim, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def to_kernel_uint8(t: Tensor) -> Tensor:
    """Flatten an FP8 / E8M0 / packed-FP4 tensor to a 1-D uint8 view.

    The FlyDSL launcher takes raw byte buffers; viewing as uint8 sidesteps
    DLPack dtype quirks for the sub-byte / FP8 dtypes.
    """
    return t.contiguous().view(torch.uint8).view(-1)
