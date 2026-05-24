// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 family heuristic dispatcher (gfx942).
//
// Available kernels:
//   * kid 6:   split-barrier a16w16 (512x128x128x64, MFMA 16x16x16) - no split-K
//   * kid 200: splitk a16w16       (512x128x128x64, MFMA 16x16x16) - independent reduce
//   * kid 201: splitk_fused a16w16 (512x128x128x64, MFMA 16x16x16) - fused reduce
//   * kid 202: splitk a16w16       (256x128x64x64,  MFMA 16x16x16) - small-N tile
//   * kid 203: splitk a16w16       (256x64x64x64,   MFMA 16x16x16) - small-MN (E_M=1)
#pragma once

#include <optional>

#include "aiter_tensor.h"
#include "../opus_gemm_common.cuh"

// -- gfx942 launcher forward declarations ------------------------------------
// Add new gfx942 kernel declarations here as they land.

// kid 6: split-barrier BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 200: splitk BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_splitk_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 201: splitk_fused BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_splitk_fused_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 202: splitk BS=256, B_M=128, B_N=64, B_K=64, T_M=2, T_N=2, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_splitk_256x128x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 203: splitk BS=256, B_M=64, B_N=64, B_K=64, T_M=2, T_N=2, MFMA=16x16x16 (E_M=1)
template <typename D_C>
void
opus_gemm_gfx942_splitk_256x64x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// -- a16w16 launcher signature (shared with gfx950) -------------------------
#ifndef OPUS_A16W16_NOSCALE_KERNEL_DEFINED
#define OPUS_A16W16_NOSCALE_KERNEL_DEFINED
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);
#endif

// Cached CU count for the current device. Avoids paying
// hipGetDeviceProperties on every dispatch call. cu_num is per-device but
// the runtime overwhelmingly runs on a single device per thread, so a
// thread_local cache is fine; the worst case is a recomputation when the
// thread switches device, which is rare.
inline int opus_gfx942_cu_num_cached()
{
  thread_local int cached = -1;
  if (cached < 0)
  {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) == hipSuccess &&
        hipGetDeviceProperties(&prop, dev) == hipSuccess)
    {
      cached = prop.multiProcessorCount;
    }
    if (cached <= 0) cached = 64;  // safe lower-bound for any gfx942 SKU
  }
  return cached;
}

// Single template body shared by both bf16 and fp32 specializations.
//
// Available gfx942 kernels:
//   kid 6:   split-barrier a16w16 (512x128x128x64, no split-K overhead,
//            lower latency for large shapes). Requires N%16==0, K%64==0,
//            ceil(K/64) even.
//   kid 200: splitk independent reduce (512x128x128x64, wins when problem
//            can't fill GPU and K is the only parallelism dim left).
//   kid 201: splitk fused reduce (same tile as 200, generally slower).
//   kid 202: splitk small-N tile (256x128x64x64), better N=32..96 fit.
//   kid 203: splitk small-MN tile (256x64x64x64, E_M=1), best for very
//            skinny shapes where per-WG work needs to be small to keep
//            occupancy + K-split together.
//
// Strategy: pick by tile coverage vs CU count.
//   1. If tiles_128x128 doesn't fill the GPU >=2x, the problem is too
//      small for split-barrier -- need K-parallelism (splitk).
//   2. Among splitk choices, pick the tile that matches N: small-N
//      (N<=64) -> 50203 (64x64) to maximize M-tile count too, mid-N
//      (N<=128) -> 50202 (128x64), otherwise 50200 (128x128).
//   3. Large problem + alignment ok -> split-barrier (kid 6, no reduce).
//   4. Large problem + alignment bad -> splitk 128x128 fallback.
template <typename CDataType>
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx942(
    int M, int N, int K, int /*batch*/, bool /*has_bias*/ = false)
{
  const int loops = (K + 63) / 64;  // ceil_div(K, B_K=64)
  const bool split_barrier_ok =
      (N % 16 == 0) && (K % 64 == 0) && (loops >= 2) && (loops % 2 == 0);

  const int cu_num    = opus_gfx942_cu_num_cached();
  const int tiles_128 = ((M + 127) / 128) * ((N + 127) / 128);
  const bool small_for_128 = tiles_128 < 2 * cu_num;

  if (small_for_128 || M <= 64)
  {
    // Problem can't keep the GPU busy on 128x128 tiles alone -- use
    // splitk. Pick the splitk tile that matches the shape.
    if (N <= 64)
    {
      return opus_gemm_gfx942_splitk_256x64x64x64_2x2_16x16x16_0x0x0<fp32_t>;
    }
    if (N <= 128)
    {
      return opus_gemm_gfx942_splitk_256x128x64x64_2x2_16x16x16_0x0x0<fp32_t>;
    }
    return opus_gemm_gfx942_splitk_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
  }

  // Large problem: split-barrier avoids reduce-kernel overhead.
  if (split_barrier_ok)
  {
    return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;
  }

  // Alignment prevents split-barrier; fall back to splitk.
  return opus_gemm_gfx942_splitk_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
}

// NOTE: splitK auto-pick lives inside each gfx942 splitk launcher itself
// (codegen'd from gen_instances.py: when splitK<=0 the launcher computes
// `ceil(cu_num / tiles_mn)` capped at 16, then runs the same
// iters-per-split clamp as caller-pinned splitK). The dispatcher passes
// splitK=0 unconditionally; arch-specific picking does not leak into
// opus_gemm.cu.
