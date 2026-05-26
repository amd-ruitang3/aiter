# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Self-contained chunked-GDN prefill bench: vLLM vs aiter-Triton vs FlyDSL.

All 407 unique (T, cu_seqlens) shapes from prefill_gdr.log are embedded
as a literal at the bottom of this file — no external CSV needed. Just
run:

    HIP_VISIBLE_DEVICES=7 python benchmarks/bench_chunked_gdr_inline.py

Three vk-layout backends are compared head-to-head per shape:

  * **vllm**   — vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule
                 (FLA upstream Triton, vk layout, no K-fusion across phases)
  * **triton** — aiter.ops.triton.gated_delta_net.gated_delta_rule
                 .chunk_gated_delta_rule_opt_vk
                 (K1+K2 fused triton, K3+K4 fused triton, K5 opt-vk triton,
                  K6 chunk_fwd_o_opt_vk triton)
  * **flydsl** — aiter.ops.flydsl.linear_attention_prefill_kernels
                 .flydsl_gdr_prefill
                 (same K1+K2 / K3+K4 / K6 as 'triton', K5 swapped to flydsl)

Same shape, same random inputs (deterministic seed per shape) → the
deltas are exactly K1+K2 fusion (vllm → triton), K3+K4 fusion (vllm →
triton), and K5 backend (triton → flydsl).

Flags:
    --backends   vllm | triton | flydsl | all   (default: all)
                 also accepts CSV: e.g. --backends vllm,flydsl
    --warmup N   warmup iters per shape         (default: 25)
    --iters N    timed iters per shape          (default: 100)
    --limit N    bench only top-N shapes by log_count
    --top K      console top-K summary
    --seed S     deterministic input seed
    --out PATH   write per-shape results CSV (default: stdout-only)

The aggregate block at the end reports, for every benched-backend pair:
    * Overall speedup (log-weighted by call frequency in the source workload)
    * Geometric mean of per-shape speedups (right average for ratios)
    * Arithmetic mean
    * Median
    * Min / Max
    * Per-shape win/loss/tie count

Shape data was extracted by parsing
/home/gyu_qle/ganyi/ATOM/prefill_gdr.log (28,152 prefill GDN calls)
and dedup'd to 407 unique (T, cu_seqlens) tuples. The `log_count` for
each shape is preserved so the impact-weighting in the aggregate
remains meaningful.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

# Fixed head dims for Qwen3-Next-80B-A3B-Instruct-FP8, TP=1 (constants
# of the model; the source log confirms every prefill call shares them).
QWEN3_NEXT_Hq = 16
QWEN3_NEXT_Hv = 32
QWEN3_NEXT_K = 128
QWEN3_NEXT_V = 128


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _load_vllm_backend() -> Callable:
    """FLA upstream vk pipeline as shipped in vLLM (chunk.py — no K-fusion
    across phases, each of K1/K2/K3/K4/K5/K6 launches separately)."""
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    return chunk_gated_delta_rule


def _load_atom_backend() -> Callable:
    """ATOM's vendored vk port of the vLLM FLA pipeline
    (atom.model_ops.fla_ops.chunk_vk.chunk_gated_delta_rule_vk).

    Should be ~bit-equal to the 'vllm' backend in steady state (verified
    by tests/test_chunk_gated_delta_rule_vk.py). Existing as a separate
    backend lets us catch any latent perf gap from ATOM's local edits
    (the o= inplace plumbing, the optional flydsl dispatch, etc.).

    Run with the flydsl dispatch OFF so we're comparing the ATOM Triton
    pipeline (= vendored vLLM kernels + ATOM's o= wiring) against the
    vLLM Triton pipeline — i.e. isolating only the ATOM-side wiring."""
    import os
    os.environ["ATOM_USE_FLYDSL_GDR_PREFILL"] = "0"
    from atom.model_ops.fla_ops.chunk_vk import chunk_gated_delta_rule_vk
    return chunk_gated_delta_rule_vk


def _load_triton_backend() -> Callable:
    """aiter Triton K1+K2/K3+K4 fused end-to-end vk pipeline."""
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_gated_delta_rule_opt_vk,
    )
    return chunk_gated_delta_rule_opt_vk


def _load_flydsl_backend() -> Callable:
    """aiter FlyDSL-K5 end-to-end vk pipeline (same K1+K2/K3+K4/K6 as the
    'triton' backend, only K5 differs)."""
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        flydsl_gdr_prefill,
    )
    return flydsl_gdr_prefill


# Single registry so the rest of the bench loop can iterate uniformly.
# Order matters: it determines the column order in the per-shape log line,
# the per-shape CSV, and the canonical baseline (first entry) used for
# the relative-speedup display.
_BACKEND_REGISTRY: dict[str, Callable[[], Callable]] = {
    "vllm": _load_vllm_backend,
    # "atom": _load_atom_backend,
    "triton": _load_triton_backend,
    "flydsl": _load_flydsl_backend,
}


# ---------------------------------------------------------------------------
# Shape table (embedded — parsed at startup from _SHAPES_RAW below)
# ---------------------------------------------------------------------------


def _parse_shapes() -> list[dict]:
    """Parse the embedded shape literal into a list of dicts."""
    shapes = []
    for line in _SHAPES_RAW.strip().splitlines():
        log_count_s, T_s, cu_s = line.split("|", 2)
        cu = [int(x) for x in cu_s.split()]
        T = int(T_s)
        n_seqs = len(cu) - 1
        # Sanity: cu_seqlens[-1] must equal T.
        assert cu[-1] == T, (
            f"embedded data corrupted: cu_seqlens[-1]={cu[-1]} != T={T} "
            f"for shape '{line[:60]}...'"
        )
        shapes.append({
            "T": T,
            "num_seqs": n_seqs,
            "cu_seqlens": cu,
            "log_count": int(log_count_s),
        })
    return shapes


# ---------------------------------------------------------------------------
# Input synthesis and timing
# ---------------------------------------------------------------------------


def _make_inputs(
    T: int,
    cu_seqlens: list[int],
    *,
    seed: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    rng = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, T, QWEN3_NEXT_Hq, QWEN3_NEXT_K, dtype=dtype,
                    device=device, generator=rng)
    k = torch.randn(1, T, QWEN3_NEXT_Hq, QWEN3_NEXT_K, dtype=dtype,
                    device=device, generator=rng)
    v = torch.randn(1, T, QWEN3_NEXT_Hv, QWEN3_NEXT_V, dtype=dtype,
                    device=device, generator=rng)
    g = -torch.rand(1, T, QWEN3_NEXT_Hv, dtype=torch.float32,
                    device=device, generator=rng)
    beta = torch.rand(1, T, QWEN3_NEXT_Hv, dtype=dtype,
                      device=device, generator=rng).sigmoid()
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    initial_state = torch.randn(
        N, QWEN3_NEXT_Hv, QWEN3_NEXT_V, QWEN3_NEXT_K,
        dtype=torch.float32, device=device, generator=rng,
    )
    return q, k, v, g, beta, cu, initial_state


def _time_call(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    """Median CUDA-event-timed latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    times_us = sorted(
        starts[i].elapsed_time(stops[i]) * 1000.0
        for i in range(iters)
    )
    return times_us[len(times_us) // 2]


def _bench_one(
    backend_fn: Callable,
    shape: dict,
    *,
    warmup: int,
    iters: int,
    seed: int,
) -> float:
    q, k, v, g, beta, cu, init = _make_inputs(
        shape["T"], shape["cu_seqlens"], seed=seed,
    )

    def _call():
        backend_fn(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=init.clone(),
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )

    return _time_call(_call, warmup=warmup, iters=iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_backends_arg(arg: str) -> list[str]:
    """Accept either the shorthand 'all' or a comma-separated list of
    registry keys. Preserves the registry's canonical order so output is
    deterministic regardless of CLI input ordering."""
    if arg == "all":
        return list(_BACKEND_REGISTRY.keys())
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in _BACKEND_REGISTRY]
    if unknown:
        sys.exit(
            f"unknown backend(s): {unknown}. "
            f"Valid: {list(_BACKEND_REGISTRY)} or 'all'."
        )
    # canonical order
    return [k for k in _BACKEND_REGISTRY if k in requested]


def _summarize_speedups(label: str, speedups: list[float], *,
                        a_total: float, b_total: float, a_name: str, b_name: str,
                        n_shapes_total: int):
    """Pretty-print the per-pair aggregate (geomean, arith, median, etc.)."""
    if not speedups or a_total == 0 or b_total == 0:
        print(f"  {label}: no comparable shapes")
        return
    geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    arith_mean = sum(speedups) / len(speedups)
    sorted_s = sorted(speedups)
    median = sorted_s[len(sorted_s) // 2]
    wins = sum(1 for s in speedups if s > 1.0)
    losses = sum(1 for s in speedups if s < 1.0)
    ties = len(speedups) - wins - losses
    p_min, p_max = sorted_s[0], sorted_s[-1]
    overall = a_total / b_total
    faster = b_name if overall > 1.0 else a_name
    print(f"  {label}  ({a_name} / {b_name}):")
    print(f"    {a_name} total wall time: {a_total/1e6:>10.3f} s")
    print(f"    {b_name} total wall time: {b_total/1e6:>10.3f} s")
    print(f"    Overall speedup (log-weighted): {overall:>7.3f}x   "
          f"(faster overall: {faster})")
    print(f"    Geometric mean:                 {geo_mean:>7.3f}x   "
          f"(right average for ratios)")
    print(f"    Arithmetic mean:                {arith_mean:>7.3f}x")
    print(f"    Median:                         {median:>7.3f}x")
    print(f"    Min / Max:                      {p_min:>7.3f}x  /  {p_max:.3f}x")
    print(f"    Per-shape outcome: {b_name} wins {wins} | loses {losses}"
          f"{f' | ties {ties}' if ties else ''}  "
          f"(out of {len(speedups)} comparable shapes, "
          f"{n_shapes_total} benched total)")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional output CSV (default: don't write)")
    ap.add_argument("--backends", default="all",
                    help="comma-separated subset of "
                         f"{list(_BACKEND_REGISTRY)}, or 'all' (default).")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="bench only top-N shapes by log_count")
    ap.add_argument("--sort", choices=("by-count", "by-T", "order"), default="by-count")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shapes = _parse_shapes()
    if args.sort == "by-count":
        shapes.sort(key=lambda s: -s["log_count"])
    elif args.sort == "by-T":
        shapes.sort(key=lambda s: s["T"])
    if args.limit is not None:
        shapes = shapes[: args.limit]
    total_log_calls = sum(s["log_count"] for s in shapes)

    backend_names = _parse_backends_arg(args.backends)
    backend_fns: dict[str, Callable] = {}
    for name in backend_names:
        try:
            backend_fns[name] = _BACKEND_REGISTRY[name]()
        except ImportError as e:
            sys.exit(f"{name} backend import failed: {e}")

    print(f"Device:   {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Shapes:   {len(shapes)} unique (covering {total_log_calls:,} log calls)",
          flush=True)
    print(f"Backends: {backend_names}    warmup={args.warmup}  iters={args.iters}",
          flush=True)
    print("", flush=True)

    # Per-shape results: each row stores us[name] for every benched backend.
    results: list[dict] = []
    t_start = time.time()
    try:
        for i, shape in enumerate(shapes):
            T = shape["T"]
            n_seqs = shape["num_seqs"]
            log_n = shape["log_count"]

            row = {
                "T": T, "num_seqs": n_seqs,
                "cu_seqlens": shape["cu_seqlens"], "log_count": log_n,
                "us": {},
            }
            for name in backend_names:
                row["us"][name] = _bench_one(
                    backend_fns[name], shape,
                    warmup=args.warmup, iters=args.iters, seed=args.seed,
                )
            results.append(row)

            cu_preview = shape["cu_seqlens"]
            cu_str = (
                str(cu_preview) if len(cu_preview) <= 6
                else f"[..., {', '.join(str(x) for x in cu_preview[-3:])}]"
            )
            cells = [f"[{i+1:>3}/{len(shapes)}]",
                     f"T={T:>6}", f"n={n_seqs:>3}",
                     f"cnt={log_n:>5}",
                     f"cu={cu_str}"]
            for name in backend_names:
                cells.append(f"{name}={row['us'][name]:>9.2f}us")
            # If at least two backends ran, show speedups relative to the
            # FIRST benched backend (the canonical-order baseline).
            if len(backend_names) >= 2:
                base = backend_names[0]
                for name in backend_names[1:]:
                    su = row["us"][base] / row["us"][name]
                    cells.append(f"{name}/{base}={su:>5.2f}x")
            print("  ".join(cells), flush=True)
    except KeyboardInterrupt:
        print("\n[interrupted, writing partial results if --out is set]", flush=True)

    elapsed = time.time() - t_start

    # Optional CSV output. Columns: T, num_seqs, cu_seqlens, log_count,
    # <name>_us for each backend, <a>_to_<b>_speedup for each pair,
    # <name>_impact_us for each backend.
    if args.out is not None:
        with args.out.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["T", "num_seqs", "cu_seqlens", "log_count"]
            header += [f"{n}_us" for n in backend_names]
            # Pairs: a/b speedup = us[a] / us[b]. Numerator is the
            # canonical-order earlier backend; positive >1 means the
            # later backend is faster.
            pairs: list[tuple[str, str]] = []
            for i, a in enumerate(backend_names):
                for b in backend_names[i+1:]:
                    pairs.append((a, b))
                    header.append(f"speedup_{b}_over_{a}")
            header += [f"{n}_impact_us" for n in backend_names]
            w.writerow(header)
            for r in results:
                cells = [
                    r["T"], r["num_seqs"],
                    " ".join(str(x) for x in r["cu_seqlens"]),
                    r["log_count"],
                ]
                for n in backend_names:
                    cells.append(f"{r['us'][n]:.2f}" if r["us"].get(n) else "")
                for a, b in pairs:
                    if r["us"].get(a) and r["us"].get(b):
                        cells.append(f"{r['us'][a] / r['us'][b]:.3f}")
                    else:
                        cells.append("")
                for n in backend_names:
                    imp = (r["us"].get(n) or 0) * r["log_count"]
                    cells.append(f"{imp:.0f}" if r["us"].get(n) else "")
                w.writerow(cells)
        print(f"\nWrote: {args.out}  ({len(results)} rows, {elapsed:.1f}s)",
              flush=True)
    else:
        print(f"\nBench done in {elapsed:.1f}s. "
              f"Pass --out PATH to write per-shape results to CSV.", flush=True)

    # Top-K shapes by total impact, using the FIRST benched backend as the
    # canonical impact metric (it's the baseline against which speedups
    # are reported).
    if not results:
        return
    impact_name = backend_names[0]
    top = sorted(
        (r for r in results if r["us"].get(impact_name)),
        key=lambda r: -(r["us"][impact_name] * r["log_count"]),
    )[: args.top]
    print(f"\nTop {min(args.top, len(top))} shapes by {impact_name} total "
          f"impact (log_count × {impact_name}_us):")
    header_cols = [f"{'T':>6}", f"{'n':>3}", f"{'cnt':>5}"]
    for n in backend_names:
        header_cols.append(f"{n+'_us':>12}")
    header_cols.append(f"{impact_name+'_impact_ms':>16}")
    header = "  ".join(header_cols)
    print(header)
    print("-" * len(header))
    for r in top:
        cells = [f"{r['T']:>6}", f"{r['num_seqs']:>3}", f"{r['log_count']:>5}"]
        for n in backend_names:
            u = r["us"].get(n)
            cells.append(f"{u:>12.2f}" if u else f"{'-':>12}")
        imp_ms = r["us"][impact_name] * r["log_count"] / 1000.0
        cells.append(f"{imp_ms:>16,.1f}")
        print("  ".join(cells))

    # Aggregate per-pair summary.
    if len(backend_names) >= 2:
        print(f"\nAggregate over {len(results)} benched shapes "
              f"({total_log_calls:,} log calls):")
        # For each ordered (a, b) pair, report `a / b` so values > 1 mean
        # b is faster (because a took more time per call than b).
        pairs = []
        for i, a in enumerate(backend_names):
            for b in backend_names[i+1:]:
                pairs.append((a, b))
        for a, b in pairs:
            a_total = sum(
                (r["us"].get(a) or 0) * r["log_count"] for r in results
                if r["us"].get(a) and r["us"].get(b)
            )
            b_total = sum(
                (r["us"].get(b) or 0) * r["log_count"] for r in results
                if r["us"].get(a) and r["us"].get(b)
            )
            speedups = [
                r["us"][a] / r["us"][b] for r in results
                if r["us"].get(a) and r["us"].get(b)
            ]
            _summarize_speedups(
                label=f"{a:>6} vs {b:>6}",
                speedups=speedups,
                a_total=a_total, b_total=b_total,
                a_name=a, b_name=b,
                n_shapes_total=len(results),
            )


# ---------------------------------------------------------------------------
# Embedded shape literal — 407 unique (T, cu_seqlens) tuples from
# prefill_gdr.log, sorted by log_count desc.
# Format: "<log_count>|<T>|<space_separated_cu_seqlens>"
# ---------------------------------------------------------------------------

_SHAPES_RAW = """
756|5063|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 5063
648|1000|0 1000
648|1063|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 1063
612|5031|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 5031
576|1015|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 1015
504|15001|0 1 5001 10001 15001
432|5000|0 5000
432|10000|0 10000
432|10015|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 10015
396|5007|0 1 2 3 4 5 6 7 5007
360|1031|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 1031
288|7001|0 1 1001 2001 3001 4001 5001 6001 7001
288|20012|0 1 2 3 4 5 6 7 8 9 10 11 12 1012 2012 3012 4012 5012 6012 7012 8012 9012 10012 11012 12012 13012 14012 15012 16012 17012 18012 19012 20012
288|30029|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 10029 20029 30029
252|5015|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 5015
252|10062|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 5062 10062
252|1003|0 1 2 3 1003
252|3001|0 1 1001 2001 3001
252|20000|0 10000 20000
252|20002|0 1 2 10002 20002
252|32768|0 1 10001 20001 30001 32768
252|32768|0 1 2 3 4 7237 17237 27237 32768
252|10063|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 10063
216|30001|0 1 5001 10001 15001 20001 25001 30001
216|32768|0 1 2 3 4 5004 10004 15004 20004 25004 30004 32768
216|30001|0 1 10001 20001 30001
216|32768|0 1 2 3 4 10004 20004 30004 32768
216|10031|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 10031
180|1001|0 1 1001
180|4476|0 1 2 3 4 5 6 7 4476
144|32768|0 1 5001 10001 15001 20001 25001 30001 32768
144|32768|0 1 2 3 4 5 6 7 5007 10007 15007 20007 25007 30007 32768
144|27246|0 1 2 3 4 5 6 7 8 9 10 2246 7246 12246 17246 22246 27246
144|10029|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 5029 10029
144|1002|0 1 2 1002
144|10002|0 1 2 1002 2002 3002 4002 5002 6002 7002 8002 9002 10002
144|30002|0 1 2 10002 20002 30002
144|32768|0 1 2 3 4 5 10005 20005 30005 32768
144|32768|0 1 2 3 4 5 6 7 8 7245 17245 27245 32768
144|32768|0 1 2 3 4 5 6 7 8 9 10 11 4488 14488 24488 32768
144|11734|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 1734 11734
144|30007|0 1 2 3 4 5 6 7 10007 20007 30007
144|32768|0 1 2 3 4 5 6 7 8 9 10 10010 20010 30010 32768
144|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 7255 17255 27255 32768
144|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 4503 14503 24503 32768
108|2240|0 1 2 3 4 5 6 7 2240
108|12252|0 1 2 3 4 5 6 7 8 9 10 11 12 13 2252 7252 12252
108|15028|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 5028 10028 15028
108|32768|0 1 2 3 4 5 6 7 8 9 5009 10009 15009 20009 25009 30009 32768
108|11001|0 1 1001 2001 3001 4001 5001 6001 7001 8001 9001 10001 11001
108|7243|0 1 2 3 4 5 6 7 7243
108|32768|0 1 2 3 4 5 6 7 7243 17243 27243 32768
108|32768|0 1 2 3 4 5 6 7 8 9 10 4485 14485 24485 32768
108|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 1754 11754 21754 31754 32768
108|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 9009 19009 29009 32768
108|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 6267 16267 26267 32768
108|23528|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 3528 13528 23528
108|30045|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 10045 20045 30045
72|5001|0 1 5001
72|30002|0 1 2 5002 10002 15002 20002 25002 30002
72|15004|0 1 2 3 4 5004 10004 15004
72|32768|0 1 2 3 4 5 5005 10005 15005 20005 25005 30005 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 2248 7248 12248 17248 22248 27248 32248 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 4498 9498 14498 19498 24498 29498 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 1754 6754 11754 16754 21754 26754 31754 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 2246 7246 12246 17246 22246 27246 32246 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 4495 9495 14495 19495 24495 29495 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 1750 6750 11750 16750 21750 26750 31750 32768
72|15029|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 5029 10029 15029
72|5009|0 1 2 3 4 5 6 7 8 9 5009
72|15027|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 5027 10027 15027
72|10030|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 5030 10030
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2256 7256 12256 17256 22256 27256 32256 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 4510 9510 14510 19510 24510 29510 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 1770 6770 11770 16770 21770 26770 31770 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 4037 9037 14037 19037 24037 29037 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 1310 6310 11310 16310 21310 26310 31310 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 3590 8590 13590 18590 23590 28590 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 876 5876 10876 15876 20876 25876 30876 32768
72|13169|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 3169 8169 13169
72|12004|0 1 2 3 4 1004 2004 3004 4004 5004 6004 7004 8004 9004 10004 11004 12004
72|1004|0 1 2 3 4 1004
72|3004|0 1 2 3 4 1004 2004 3004
72|9007|0 1 2 3 4 5 6 7 1007 2007 3007 4007 5007 6007 7007 8007 9007
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1013 2013 3013 4013 5013 6013 7013 8013 9013 10013 11013 12013 13013 14013 15013 16013 17013 18013 19013 20013 21013 22013 23013 24013 25013 26013 27013 28013 29013 30013 31013 32013 32768
72|18290|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 290 1290 2290 3290 4290 5290 6290 7290 8290 9290 10290 11290 12290 13290 14290 15290 16290 17290 18290
72|10003|0 1 2 3 1003 2003 3003 4003 5003 6003 7003 8003 9003 10003
72|32014|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 1014 2014 3014 4014 5014 6014 7014 8014 9014 10014 11014 12014 13014 14014 15014 16014 17014 18014 19014 20014 21014 22014 23014 24014 25014 26014 27014 28014 29014 30014 31014 32014
72|18046|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 1046 2046 3046 4046 5046 6046 7046 8046 9046 10046 11046 12046 13046 14046 15046 16046 17046 18046
72|6005|0 1 2 3 4 5 1005 2005 3005 4005 5005 6005
72|21043|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 1043 2043 3043 4043 5043 6043 7043 8043 9043 10043 11043 12043 13043 14043 15043 16043 17043 18043 19043 20043 21043
72|3013|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1013 2013 3013
72|32768|0 1 2 3 4 5 6 7 4476 14476 24476 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 1718 11718 21718 31718 32768
72|10001|0 1 10001
72|32768|0 1 2 10002 20002 30002 32768
72|32768|0 1 2 3 4 5 7239 17239 27239 32768
72|32768|0 1 2 3 4 5 6 7 8 4479 14479 24479 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 1722 11722 21722 31722 32768
72|30028|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 10028 20028 30028
72|10047|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 10047
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 10041 20041 30041 32768
72|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 10032 20032 30032 32768
72|10051|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 10051
36|32768|0 1 2 3 4 5 6 7 2240 7240 12240 17240 22240 27240 32240 32768
36|9486|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 4486 9486
36|20001|0 1 5001 10001 15001 20001
36|4017|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 4017
36|9012|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 4012 9012
36|32768|0 1 2 3 4 5 6 7 8 5008 10008 15008 20008 25008 30008 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 2254 7254 12254 17254 22254 27254 32254 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 4507 9507 14507 19507 24507 29507 32768
36|21766|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 1766 6766 11766 16766 21766
36|5018|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 5018
36|30008|0 1 2 3 4 5 6 7 8 5008 10008 15008 20008 25008 30008
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 5014 10014 15014 20014 25014 30014 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 2266 7266 12266 17266 22266 27266 32266 32768
36|24525|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 4525 9525 14525 19525 24525
36|5008|0 1 2 3 4 5 6 7 8 5008
36|22256|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2256 7256 12256 17256 22256
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 5020 10020 15020 20020 25020 30020 32768
36|27278|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 2278 7278 12278 17278 22278 27278
36|32768|0 1 2 3 4 5 6 7 8 9 10 5010 10010 15010 20010 25010 30010 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 2258 7258 12258 17258 22258 27258 32258 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 4513 9513 14513 19513 24513 29513 32768
36|11774|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 1774 6774 11774
36|5010|0 1 2 3 4 5 6 7 8 9 10 5010
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 5011 10011 15011 20011 25011 30011 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 2260 7260 12260 17260 22260 27260 32260 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 4516 9516 14516 19516 24516 29516 32768
36|6778|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 1778 6778
36|30011|0 1 2 3 4 5 6 7 8 9 10 11 5011 10011 15011 20011 25011 30011
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 5017 10017 15017 20017 25017 30017 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 2272 7272 12272 17272 22272 27272 32272 32768
36|9534|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 4534 9534
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 2252 7252 12252 17252 22252 27252 32252 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 4504 9504 14504 19504 24504 29504 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 1762 6762 11762 16762 21762 26762 31762 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 4027 9027 14027 19027 24027 29027 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 1298 6298 11298 16298 21298 26298 31298 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 3576 8576 13576 18576 23576 28576 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 860 5860 10860 15860 20860 25860 30860 32768
36|23151|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 3151 8151 13151 18151 23151
36|10001|0 1 5001 10001
36|32768|0 1 2 3 5003 10003 15003 20003 25003 30003 32768
36|32768|0 1 2 3 4 5 6 7 8 9 2244 7244 12244 17244 22244 27244 32244 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 4492 9492 14492 19492 24492 29492 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 1746 6746 11746 16746 21746 26746 31746 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 4007 9007 14007 19007 24007 29007 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 1274 6274 11274 16274 21274 26274 31274 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 3548 8548 13548 18548 23548 28548 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 828 5828 10828 15828 20828 25828 30828 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 3115 8115 13115 18115 23115 28115 32768
36|10408|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 408 5408 10408
36|5006|0 1 2 3 4 5 6 5006
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 4012 9012 14012 19012 24012 29012 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 1280 6280 11280 16280 21280 26280 31280 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 3555 8555 13555 18555 23555 28555 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 836 5836 10836 15836 20836 25836 30836 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 3124 8124 13124 18124 23124 28124 32768
36|5418|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 418 5418
36|5004|0 1 2 3 4 5004
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 4017 9017 14017 19017 24017 29017 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 1286 6286 11286 16286 21286 26286 31286 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 3562 8562 13562 18562 23562 28562 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 844 5844 10844 15844 20844 25844 30844 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 3133 8133 13133 18133 23133 28133 32768
36|428|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 428
36|5005|0 1 2 3 4 5 5005
36|32768|0 1 2 3 4 5 6 5006 10006 15006 20006 25006 30006 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 2250 7250 12250 17250 22250 27250 32250 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 4501 9501 14501 19501 24501 29501 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 1758 6758 11758 16758 21758 26758 31758 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 4022 9022 14022 19022 24022 29022 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 1292 6292 11292 16292 21292 26292 31292 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 3569 8569 13569 18569 23569 28569 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 852 5852 10852 15852 20852 25852 30852 32768
36|28142|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 3142 8142 13142 18142 23142 28142
36|5012|0 1 2 3 4 5 6 7 8 9 10 11 12 5012
36|10007|0 1 2 3 4 5 6 7 5007 10007
36|10017|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 5017 10017
36|25010|0 1 2 3 4 5 6 7 8 9 10 5010 10010 15010 20010 25010
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 5015 10015 15015 20015 25015 30015 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 2268 7268 12268 17268 22268 27268 32268 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 4528 9528 14528 19528 24528 29528 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 1794 6794 11794 16794 21794 26794 31794 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 4067 9067 14067 19067 24067 29067 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 1346 6346 11346 16346 21346 26346 31346 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 3632 8632 13632 18632 23632 28632 32768
36|15924|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 924 5924 10924 15924
36|3000|0 1000 2000 3000
36|15001|0 1 1001 2001 3001 4001 5001 6001 7001 8001 9001 10001 11001 12001 13001 14001 15001
36|14002|0 1 2 1002 2002 3002 4002 5002 6002 7002 8002 9002 10002 11002 12002 13002 14002
36|13003|0 1 2 3 1003 2003 3003 4003 5003 6003 7003 8003 9003 10003 11003 12003 13003
36|11005|0 1 2 3 4 5 1005 2005 3005 4005 5005 6005 7005 8005 9005 10005 11005
36|9003|0 1 2 3 1003 2003 3003 4003 5003 6003 7003 8003 9003
36|12001|0 1 1001 2001 3001 4001 5001 6001 7001 8001 9001 10001 11001 12001
36|1051|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 1051
36|11002|0 1 2 1002 2002 3002 4002 5002 6002 7002 8002 9002 10002 11002
36|32013|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1013 2013 3013 4013 5013 6013 7013 8013 9013 10013 11013 12013 13013 14013 15013 16013 17013 18013 19013 20013 21013 22013 23013 24013 25013 26013 27013 28013 29013 30013 31013 32013
36|19045|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 1045 2045 3045 4045 5045 6045 7045 8045 9045 10045 11045 12045 13045 14045 15045 16045 17045 18045 19045
36|1021|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 1021
36|1013|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1013
36|10004|0 1 2 3 4 1004 2004 3004 4004 5004 6004 7004 8004 9004 10004
36|1053|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 1053
36|32011|0 1 2 3 4 5 6 7 8 9 10 11 1011 2011 3011 4011 5011 6011 7011 8011 9011 10011 11011 12011 13011 14011 15011 16011 17011 18011 19011 20011 21011 22011 23011 24011 25011 26011 27011 28011 29011 30011 31011 32011
36|1025|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 1025
36|2011|0 1 2 3 4 5 6 7 8 9 10 11 1011 2011
36|27016|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 1016 2016 3016 4016 5016 6016 7016 8016 9016 10016 11016 12016 13016 14016 15016 16016 17016 18016 19016 20016 21016 22016 23016 24016 25016 26016 27016
36|1055|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 1055
36|8056|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 1056 2056 3056 4056 5056 6056 7056 8056
36|4016|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 1016 2016 3016 4016
36|32020|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 1020 2020 3020 4020 5020 6020 7020 8020 9020 10020 11020 12020 13020 14020 15020 16020 17020 18020 19020 20020 21020 22020 23020 24020 25020 26020 27020 28020 29020 30020 31020 32020
36|12052|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 1052 2052 3052 4052 5052 6052 7052 8052 9052 10052 11052 12052
36|18964|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 8964 18964
36|8969|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 8969
36|21730|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1730 11730 21730
36|20001|0 1 10001 20001
36|10003|0 1 2 3 10003
36|11730|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1730 11730
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 8964 18964 28964 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 6213 16213 26213 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 3465 13465 23465 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 720 10720 20720 30720 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 7979 17979 27979 32768
36|15241|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 5241 15241
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 1730 11730 21730 31730 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 8979 18979 28979 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 6231 16231 26231 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 3486 13486 23486 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 744 10744 20744 30744 32768
36|18006|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 8006 18006
36|30004|0 1 2 3 4 10004 20004 30004
36|32768|0 1 2 3 4 5 6 10006 20006 30006 32768
36|32768|0 1 2 3 4 5 6 7 8 9 7247 17247 27247 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 4491 14491 24491 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 1738 11738 21738 31738 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 8989 18989 28989 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 6243 16243 26243 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 3500 13500 23500 32768
36|30760|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 760 10760 20760 30760
36|10006|0 1 2 3 4 5 6 10006
36|32768|0 1 2 3 4 5 6 7 10007 20007 30007 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 7249 17249 27249 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 4494 14494 24494 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 1742 11742 21742 31742 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 8994 18994 28994 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 6249 16249 26249 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 3507 13507 23507 32768
36|20768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 768 10768 20768
36|10029|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 10029
36|20030|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 10030 20030
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 1753 11753 21753 31753 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 9005 19005 29005 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 6260 16260 26260 32768
36|13518|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 3518 13518
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 8969 18969 28969 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 6219 16219 26219 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 3472 13472 23472 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 728 10728 20728 30728 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 7988 17988 27988 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 5251 15251 25251 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 2517 12517 22517 32517 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 9787 19787 29787 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 7060 17060 27060 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4336 14336 24336 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 1615 11615 21615 31615 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 8898 18898 28898 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 6184 16184 26184 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 3473 13473 23473 32768
36|30765|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 765 10765 20765 30765
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 7314 17314 27314 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 4587 14587 24587 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 1859 11859 21859 31859 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 9132 19132 29132 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 6405 16405 26405 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 3678 13678 23678 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 950 10950 20950 30950 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 8223 18223 28223 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 5496 15496 25496 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 2769 12769 22769 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 7317 17317 27317 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 4596 14596 24596 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1878 11878 21878 31878 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 9164 19164 29164 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 6453 16453 26453 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 3745 13745 23745 32768
36|1040|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 1040
36|10038|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 10038
36|27296|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 7296 17296 27296
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 7295 17295 27295 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 4558 14558 24558 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 1821 11821 21821 31821 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 9084 19084 29084 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 6347 16347 26347 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 3610 13610 23610 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 873 10873 20873 30873 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 8139 18139 28139 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 5408 15408 25408 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 2680 12680 22680 32680 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 9956 19956 29956 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 7235 17235 27235 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 4517 14517 24517 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 1802 11802 21802 31802 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 9091 19091 29091 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 6383 16383 26383 32768
36|3678|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 3678
36|10044|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 10044
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 10039 20039 30039 32768
36|7310|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 7310
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 10036 20036 30036 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 7304 17304 27304 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 4572 14572 24572 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 1839 11839 21839 31839 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 9107 19107 29107 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 6375 16375 26375 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 3643 13643 23643 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 910 10910 20910 30910 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 8178 18178 28178 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 5446 15446 25446 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 2716 12716 22716 32716 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 9990 19990 29990 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 7267 17267 27267 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 4547 14547 24547 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 1830 11830 21830 31830 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 9117 19117 29117 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 6407 16407 26407 32768
36|23700|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 3700 13700 23700
36|10053|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 10053
36|30047|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 10047 20047 30047
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 10047 20047 30047 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 7326 17326 27326 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 4605 14605 24605 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 1883 11883 21883 31883 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 9162 19162 29162 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 6441 16441 26441 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 3719 13719 23719 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 997 10997 20997 30997 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 8276 18276 28276 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 5555 15555 25555 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 2833 12833 22833 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 111 10111 20111 30111 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 7390 17390 27390 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 4669 14669 24669 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1951 11951 21951 31951 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 9237 19237 29237 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 6526 16526 26526 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 3818 13818 23818 32768
36|1113|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 1113
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 10045 20045 30045 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 7322 17322 27322 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4598 14598 24598 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 1874 11874 21874 31874 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 9151 19151 29151 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 6428 16428 26428 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 3704 13704 23704 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 980 10980 20980 30980 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 8257 18257 28257 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 5533 15533 25533 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 2809 12809 22809 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 85 10085 20085 30085 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 7362 17362 27362 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 4641 14641 24641 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1923 11923 21923 31923 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 9209 19209 29209 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 6498 16498 26498 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 3790 13790 23790 32768
36|1085|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 1085
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 10042 20042 30042 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 7315 17315 27315 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 4588 14588 24588 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 1861 11861 21861 31861 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 9135 19135 29135 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 6408 16408 26408 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 3681 13681 23681 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 954 10954 20954 30954 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 8227 18227 28227 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 5500 15500 25500 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 2773 12773 22773 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 46 10046 20046 30046 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 7322 17322 27322 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 4601 14601 24601 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1883 11883 21883 31883 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 9169 19169 29169 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 6458 16458 26458 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 3750 13750 23750 32768
36|1045|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 1045
36|30042|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 10042 20042 30042
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 10038 20038 30038 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 7308 17308 27308 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 4578 14578 24578 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 1848 11848 21848 31848 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 9118 19118 29118 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 6388 16388 26388 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 3658 13658 23658 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 927 10927 20927 30927 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 8197 18197 28197 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 5467 15467 25467 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 2737 12737 22737 32737 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 10010 20010 30010 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 7286 17286 27286 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 4565 14565 24565 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 1847 11847 21847 31847 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 9133 19133 29133 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 6422 16422 26422 32768
36|32768|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 3714 13714 23714 32768
36|1009|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 1009
"""
if __name__ == "__main__":
    main()
