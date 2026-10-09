# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check native gfx1250 E4M3 block-scale rounding against PyTorch."""

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.parametrize("global_scale", [1.0, 0.7, 1.3])
def test_e4m3_block_scale_rounding(global_scale):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith
    from flydsl.expr.typing import Constexpr, T
    from flydsl.runtime.device import get_rocm_arch

    from aiter.ops.flydsl.kernels.fp4_scale import emit_fp4_block_scale

    if "gfx1250" not in get_rocm_arch().lower():
        pytest.skip("requires gfx1250 native FP8 conversions")

    @flyc.jit
    def launch(
        inp: fx.Tensor,
        decoded: fx.Tensor,
        encoded: fx.Tensor,
        stream: fx.Stream,
        count: Constexpr[int],
        tensor_scale: Constexpr[float],
    ):
        @flyc.kernel(name="e4m3_scale_rounding", known_block_size=[256, 1, 1])
        def kernel(inp: fx.Tensor, decoded: fx.Tensor, encoded: fx.Tensor):
            idx = fx.block_idx.x * 256 + fx.thread_idx.x
            if idx < count:
                scale, byte = emit_fp4_block_scale(inp[idx], 2, tensor_scale)
                decoded[idx] = scale
                encoded[idx] = fx.Int32(arith.extui(T.i32, byte))

        kernel(inp, decoded, encoded).launch(
            grid=((count + 255) // 256,), block=(256,), stream=stream
        )

    # Every nonnegative finite E4M3 value, every rounding midpoint, and the
    # immediately adjacent f32 values exercise RNE and the subnormal boundary.
    levels = (
        torch.arange(127, dtype=torch.uint8, device="cuda")
        .view(torch.float8_e4m3fn)
        .float()
    )
    midpoint = (levels[:-1] + levels[1:]) / 2
    generator = torch.Generator(device="cuda").manual_seed(61)
    random_values = torch.exp2(
        torch.rand(32768, device="cuda", generator=generator) * 150 - 130
    ) * torch.rand(32768, device="cuda", generator=generator)
    values = torch.cat(
        (
            levels,
            midpoint,
            torch.nextafter(midpoint, torch.zeros_like(midpoint)),
            torch.nextafter(midpoint, torch.full_like(midpoint, float("inf"))),
            torch.tensor([0, 448, 449, 512, 1e-38, 1e-20], device="cuda"),
            random_values,
        )
    )
    inp = (values * (6 * global_scale)).contiguous()
    decoded = torch.empty_like(inp)
    encoded = torch.empty(inp.shape, device="cuda", dtype=torch.int32)
    launch(
        inp,
        decoded,
        encoded,
        torch.cuda.current_stream(),
        inp.numel(),
        global_scale,
    )
    rounded = (
        (inp * (1.0 / (6.0 * global_scale))).clamp(max=448).to(torch.float8_e4m3fn)
    )
    expected_byte = rounded.view(torch.uint8).int()
    expected_scale = torch.where(expected_byte == 0, 1.0, rounded.float())
    torch.testing.assert_close(encoded, expected_byte, rtol=0, atol=0)
    torch.testing.assert_close(decoded, expected_scale, rtol=0, atol=0)
