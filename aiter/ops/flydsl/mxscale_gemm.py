# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public wrapper for the FlyDSL gfx1250 MXScale dense GEMM kernel.

Exposes a single first-class entry point ``flydsl_mxscale_gemm`` covering
``data_format`` ``"fp8"`` (MXFP8 E4M3 + E8M0) and ``"a8w4"`` (FP8 activation
+ FP4 weight, both with E8M0 1x32 scales). The fp4 path is **not** exposed
here even though the underlying kernel builder supports it.

This wrapper is intentionally not wired into ``gemm_a8w8_blockscale`` or
``gemm_a8w8_bpreshuffle`` dispatch tables; callers must invoke it explicitly
because the scale dtype/layout contract is not interchangeable with the
existing PTPC-FP8 / blockscale-FP8 paths.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx

from .mxscale_layout import (
    SCALE_BLOCK,
    get_padded_problem_shape,
    pad_mxscale_inputs,
    preshuffle_b_16x16,
    preshuffle_e8m0_scale_wmma,
    to_kernel_uint8,
    validate_mxscale_num_buffers,
)
from .utils import is_flydsl_available

# Sentinels for keeping the import surface small when flydsl is missing.
_compile_mxscale_gemm = None  # type: ignore[assignment]
_run_compiled = None  # type: ignore[assignment]
_fx = None  # type: ignore[assignment]

_TARGET_GFX = "gfx1250"
_VALID_FORMATS = ("fp8", "a8w4")
_VALID_OUT_DTYPES = ("bf16", "f16", "f32")

_TORCH_DTYPE_FROM_NAME = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}
_NAME_FROM_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_FROM_NAME.items()}


def _resolve_target_device(*tensors: Optional[Tensor]) -> torch.device:
    cuda_devices = []
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        if tensor.device not in cuda_devices:
            cuda_devices.append(tensor.device)
    if len(cuda_devices) > 1:
        devices = ", ".join(str(device) for device in cuda_devices)
        raise ValueError(f"all MXScale tensors must use one CUDA device, got {devices}")
    if cuda_devices:
        return cuda_devices[0]
    if not torch.cuda.is_available():
        raise RuntimeError("flydsl_mxscale_gemm requires an available CUDA device")
    return torch.device("cuda", torch.cuda.current_device())


def _to_target_device(tensor: Tensor, device: torch.device) -> Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _lazy_import_flydsl():
    """Import flydsl-dependent symbols lazily so the module loads without flydsl."""
    global _compile_mxscale_gemm, _run_compiled, _fx
    if _compile_mxscale_gemm is not None:
        return
    if not is_flydsl_available():
        raise RuntimeError(
            "flydsl is not installed; install the matching flydsl wheel to use "
            "flydsl_mxscale_gemm."
        )
    import flydsl.expr as fx_mod

    from .kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm as _compile
    from .kernels.tensor_shim import _run_compiled as _runner

    _compile_mxscale_gemm = _compile
    _run_compiled = _runner
    _fx = fx_mod


# ---------------------------------------------------------------------------
# kernelName encode / decode
# ---------------------------------------------------------------------------

# Format:
#   flydsl_mxscale_{fmt}_{out}_t{tm}x{tn}x{tk}_mw{mw}_nw{nw}_buf{buf}
#     _sk{sk}_tdms{0|1}_opsel{0|1}_wst{0|1}_l2pf{d}_cm{cm}_cn{cn}_wpe{wpe}_gfx1250
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_mxscale_"
    r"(?P<fmt>fp8|a8w4)_"
    r"(?P<out>bf16|f16|f32)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_buf(?P<num_buffers>\d+)_"
    r"sk(?P<split_k>\d+)_"
    r"tdms(?P<use_tdm_store>[01])_"
    r"opsel(?P<use_scale_opsel>[01])_"
    r"wst(?P<wave_specialized_tdm>[01])_"
    r"l2pf(?P<l2_prefetch_distance>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)_"
    r"wpe(?P<waves_per_eu>\d+)_"
    r"(?P<target_gfx>gfx1250)$"
)


def flydsl_mxscale_kernel_name(
    *,
    data_format: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int,
    use_tdm_store: bool,
    use_scale_opsel: bool,
    wave_specialized_tdm: bool,
    l2_prefetch_distance: int,
    cluster_m: int,
    cluster_n: int,
    waves_per_eu: int,
    target_gfx: str = _TARGET_GFX,
) -> str:
    """Encode a fully-qualified kernel name for the gfx1250 MXScale kernel."""
    if data_format not in _VALID_FORMATS:
        raise ValueError(f"data_format must be one of {_VALID_FORMATS}, got {data_format!r}")
    if out_dtype not in _VALID_OUT_DTYPES:
        raise ValueError(f"out_dtype must be one of {_VALID_OUT_DTYPES}, got {out_dtype!r}")
    return (
        f"flydsl_mxscale_{data_format}_{out_dtype}_"
        f"t{tile_m}x{tile_n}x{tile_k}_mw{m_warp}_nw{n_warp}_buf{num_buffers}_"
        f"sk{split_k}_tdms{int(bool(use_tdm_store))}_"
        f"opsel{int(bool(use_scale_opsel))}_wst{int(bool(wave_specialized_tdm))}_"
        f"l2pf{l2_prefetch_distance}_cm{cluster_m}_cn{cluster_n}_"
        f"wpe{waves_per_eu}_{target_gfx}"
    )


def parse_flydsl_mxscale_kernel_name(name: str) -> Optional[Dict]:
    """Parse a mxscale kernel name string into a config dict, or None if no match.

    Returned dict carries ``kind="mxscale"`` plus all codegen parameters; all
    integer fields are ints, booleans are bools.
    """
    m = _KERNEL_NAME_RE.fullmatch(name)
    if m is None:
        return None
    return {
        "kind": "mxscale",
        "data_format": m.group("fmt"),
        "out_dtype": m.group("out"),
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "m_warp": int(m.group("m_warp")),
        "n_warp": int(m.group("n_warp")),
        "num_buffers": int(m.group("num_buffers")),
        "split_k": int(m.group("split_k")),
        "use_tdm_store": m.group("use_tdm_store") == "1",
        "use_scale_opsel": m.group("use_scale_opsel") == "1",
        "wave_specialized_tdm": m.group("wave_specialized_tdm") == "1",
        "l2_prefetch_distance": int(m.group("l2_prefetch_distance")),
        "cluster_m": int(m.group("cluster_m")),
        "cluster_n": int(m.group("cluster_n")),
        "waves_per_eu": int(m.group("waves_per_eu")),
        "target_gfx": m.group("target_gfx"),
    }


# ---------------------------------------------------------------------------
# Public runtime entry
# ---------------------------------------------------------------------------


def _resolve_out_dtype(
    out: Optional[Tensor], out_dtype: Optional[str]
) -> Tuple[str, torch.dtype]:
    if out is not None:
        torch_dt = out.dtype
        name = _NAME_FROM_TORCH_DTYPE.get(torch_dt)
        if name is None:
            raise ValueError(
                f"out tensor dtype {torch_dt} not supported; expected one of "
                f"{list(_TORCH_DTYPE_FROM_NAME.values())}"
            )
        if out_dtype is not None and out_dtype != name:
            raise ValueError(
                f"out_dtype={out_dtype!r} conflicts with out.dtype={torch_dt}"
            )
        return name, torch_dt
    if out_dtype is None:
        out_dtype = "bf16"
    if out_dtype not in _TORCH_DTYPE_FROM_NAME:
        raise ValueError(
            f"out_dtype must be one of {_VALID_OUT_DTYPES}, got {out_dtype!r}"
        )
    return out_dtype, _TORCH_DTYPE_FROM_NAME[out_dtype]


def flydsl_mxscale_gemm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    *,
    data_format: str = "fp8",
    out: Optional[Tensor] = None,
    out_dtype: Optional[str] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 2,
    split_k: int = 1,
    use_tdm_store: bool = True,
    use_scale_opsel: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    waves_per_eu: int = 0,
    l2_prefetch_distance: int = 2,
    inst_prefetch: bool = False,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    kernel_name: Optional[str] = None,
) -> Tensor:
    """Run a FlyDSL gfx1250 MXScale GEMM (data_format ∈ {"fp8", "a8w4"}).

    Parameters
    ----------
    A : (M, K) for fp8 / (M, K) for a8w4 (always FP8 byte storage).
    B : (N, K) for fp8 / (N, K // 2) for a8w4 (FP4 packed bytes).
        Caller may pass an unshuffled tensor; it will be 16x16-preshuffled here.
    A_scale : (M, K // 32) E8M0 (uint8 storage).
    B_scale : (N, K // 32) E8M0 (uint8 storage).

    Notes
    -----
    * If ``out`` is provided it must have shape ``(M, N)``. The wrapper
      allocates an internal padded buffer when ``M`` / ``N`` / ``K`` are not
      tile-aligned, then slice-copies into ``out``.
    * ``split_k > 1`` forces ``use_tdm_store=False`` and zero-fills the padded
      output before launch (atomic add accumulation).
    * ``kernel_name`` may be provided when the dispatch was already resolved
      against a tuned CSV; codegen parameters from it override the keyword
      arguments. When ``None`` a kernel name is synthesised from the kwargs.
    """
    if data_format not in _VALID_FORMATS:
        raise ValueError(
            f"data_format must be one of {_VALID_FORMATS}, got {data_format!r}"
        )
    cur_gfx = get_gfx()
    if cur_gfx != _TARGET_GFX:
        raise RuntimeError(
            f"flydsl_mxscale_gemm requires {_TARGET_GFX}, current arch is {cur_gfx!r}"
        )
    _lazy_import_flydsl()

    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(
            f"A and B must be 2-D, got A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}"
        )
    M = A.shape[0]
    N = B.shape[0]

    # If a kernel name was provided, use its codegen parameters verbatim.
    if kernel_name is not None:
        parsed = parse_flydsl_mxscale_kernel_name(kernel_name)
        if parsed is None:
            raise ValueError(f"unrecognised mxscale kernel_name: {kernel_name!r}")
        if parsed["data_format"] != data_format:
            raise ValueError(
                f"kernel_name data_format={parsed['data_format']!r} != "
                f"data_format={data_format!r}"
            )
        tile_m = parsed["tile_m"]
        tile_n = parsed["tile_n"]
        tile_k = parsed["tile_k"]
        m_warp = parsed["m_warp"]
        n_warp = parsed["n_warp"]
        num_buffers = parsed["num_buffers"]
        split_k = parsed["split_k"]
        use_tdm_store = parsed["use_tdm_store"]
        use_scale_opsel = parsed["use_scale_opsel"]
        wave_specialized_tdm = parsed["wave_specialized_tdm"]
        l2_prefetch_distance = parsed["l2_prefetch_distance"]
        cluster_m = parsed["cluster_m"]
        cluster_n = parsed["cluster_n"]
        waves_per_eu = parsed["waves_per_eu"]
        out_dtype = parsed["out_dtype"]

    out_dtype_name, out_torch_dtype = _resolve_out_dtype(out, out_dtype)

    # split_k > 1 requires plain buffer-store (atomic adds) and a zero-init out.
    if split_k > 1 and use_tdm_store:
        logger.info(
            "[flydsl_mxscale_gemm] split_k>1 requires use_tdm_store=False; "
            "overriding."
        )
        use_tdm_store = False

    # ----- Recover unpadded K -----
    pack_a = 1
    pack_b = 1 if data_format == "fp8" else 2
    if A.shape[1] * pack_a != B.shape[1] * pack_b:
        raise ValueError(
            f"A and B contraction dimensions disagree: A.shape[1]={A.shape[1]} "
            f"(pack_a={pack_a}) vs B.shape[1]={B.shape[1]} (pack_b={pack_b})"
        )
    K = A.shape[1] * pack_a
    if K % SCALE_BLOCK != 0:
        raise ValueError(
            f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}"
        )
    if A_scale.shape != (M, K // SCALE_BLOCK):
        raise ValueError(
            f"A_scale shape must be {(M, K // SCALE_BLOCK)}, got {tuple(A_scale.shape)}"
        )
    if B_scale.shape != (N, K // SCALE_BLOCK):
        raise ValueError(
            f"B_scale shape must be {(N, K // SCALE_BLOCK)}, got {tuple(B_scale.shape)}"
        )

    validate_mxscale_num_buffers(K, tile_k, num_buffers, split_k=split_k)
    target_device = _resolve_target_device(A, B, A_scale, B_scale, out)

    if out is not None and tuple(out.shape) != (M, N):
        raise ValueError(
            f"out shape must be {(M, N)}, got {tuple(out.shape)}"
        )
    if out is not None and out.device != target_device:
        raise ValueError(
            f"out must be on the MXScale launch device {target_device}, "
            f"got {out.device}"
        )

    # ----- Pad + preshuffle -----
    padded = get_padded_problem_shape(
        data_format, M, N, K, tile_m, tile_n, tile_k, split_k=split_k
    )
    a_p, b_p, a_s_p, b_s_p = pad_mxscale_inputs(A, B, A_scale, B_scale, padded)
    K_packed_b = padded["K"] // padded["pack_b"]
    b_p = preshuffle_b_16x16(b_p, padded["N"], K_packed_b)
    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    a_s_p = preshuffle_e8m0_scale_wmma(a_s_p, warp_tile_m, scale_k_per_tile=skt)
    b_s_p = preshuffle_e8m0_scale_wmma(b_s_p, warp_tile_n, scale_k_per_tile=skt)

    a_dev = _to_target_device(a_p, target_device)
    b_dev = _to_target_device(b_p, target_device)
    a_s_dev = _to_target_device(a_s_p, target_device)
    b_s_dev = _to_target_device(b_s_p, target_device)

    # ----- Allocate padded out + (optional) zero-init for split-K -----
    out_buf = torch.empty(
        (padded["M"], padded["N"]),
        dtype=out_torch_dtype,
        device=a_dev.device,
    )
    if split_k > 1:
        out_buf.zero_()

    # ----- Compile + launch -----
    launch_fn = _compile_mxscale_gemm(
        data_format=data_format,
        M=padded["M"],
        N=padded["N"],
        K=padded["K"],
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu if waves_per_eu > 0 else None,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=out_dtype_name,
        inst_prefetch=inst_prefetch,
        wave_specialized_tdm=wave_specialized_tdm,
        split_k=split_k,
        use_scale_opsel=use_scale_opsel,
        expert_sched_mode=expert_sched_mode,
        atomic_barrier_enable=atomic_barrier_enable,
    )
    stream = _fx.Stream(torch.cuda.current_stream(device=a_dev.device))
    _run_compiled(
        launch_fn,
        out_buf.contiguous().view(-1),
        to_kernel_uint8(a_dev),
        to_kernel_uint8(b_dev),
        to_kernel_uint8(a_s_dev),
        to_kernel_uint8(b_s_dev),
        padded["M"],
        padded["N"],
        stream,
    )

    # ----- Slice padded buffer back to (M, N) -----
    if (padded["M"], padded["N"]) == (M, N):
        if out is None:
            return out_buf
        out.copy_(out_buf)
        return out
    if out is None:
        return out_buf[:M, :N].contiguous()
    out.copy_(out_buf[:M, :N])
    return out


# ---------------------------------------------------------------------------
# Format-named public wrappers
# ---------------------------------------------------------------------------
#
# These thin wrappers expose data-format-specific entry points so callers can
# write ``aiter.gemm_mxfp8(...)`` / ``aiter.gemm_mxa8w4(...)`` without caring
# which backend implements them. Today both route to the FlyDSL backend below;
# when a second backend (e.g. a Triton/Gluon mxfp8 kernel) lands, these
# entries can grow into proper dispatch points without changing the public
# API name — mirroring how PR 2332 evolved ``gemm_afp4wfp4`` from a single
# implementation into an arch-dispatched entry.


def gemm_mxfp8(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    **kwargs,
) -> Tensor:
    """Public entry for OCP MX FP8 dense GEMM (E4M3 + E8M0 1x32 scale)."""
    if "data_format" in kwargs:
        raise TypeError("gemm_mxfp8 does not accept data_format; format is fixed to 'fp8'")
    return flydsl_mxscale_gemm(A, B, A_scale, B_scale, data_format="fp8", **kwargs)


def gemm_mxa8w4(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    **kwargs,
) -> Tensor:
    """Public entry for OCP MX A8W4 dense GEMM (FP8 act, FP4 weight, E8M0 1x32 scale)."""
    if "data_format" in kwargs:
        raise TypeError("gemm_mxa8w4 does not accept data_format; format is fixed to 'a8w4'")
    return flydsl_mxscale_gemm(A, B, A_scale, B_scale, data_format="a8w4", **kwargs)


__all__ = [
    "flydsl_mxscale_gemm",
    "flydsl_mxscale_kernel_name",
    "parse_flydsl_mxscale_kernel_name",
    "gemm_mxfp8",
    "gemm_mxa8w4",
]
