"""Sweep BV ? {16, 32, 64} for every PREFILL_PARAMS shape, find the per-shape
optimum under the *current* kernel code (e.g. OPT-VC rev17), and print a
final summary table.

Usage:
    HIP_VISIBLE_DEVICES=4 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        python3 scripts/sweep_flydsl_k5_bv.py

The script monkey-patches ``_lookup_tuned_bv`` so each PREFILL_PARAMS shape is
forced to use a specific BV. Each (shape, BV) combo is timed via the same
``_bench_fn`` (NUM_WARMUP=5 + NUM_ITERS=50, torch.profiler) used by the
existing TestPerformance harness, so numbers are directly comparable.

Output: one row per shape with the BV-sweep result, plus the existing tuned
csv's BV for context.
"""

from __future__ import annotations

import os
import sys

# Make sure the repo's aiter package is importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from aiter.ops.flydsl import (  # noqa: E402
    linear_attention_prefill_kernels as _lap,
)
from aiter.ops.flydsl.linear_attention_prefill_kernels import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h_flydsl,
)
from aiter.ops.flydsl.test_flydsl_linear_attention_prefill import (  # noqa: E402
    NUM_ITERS,
    NUM_WARMUP,
    PREFILL_PARAMS,
    PREFILL_TEST_IDS,
    _bench_fn,
    _build_context_lens,
    _make_inputs,
)


def _force_bv(bv_value: int):
    """Return a monkey-patched ``_lookup_tuned_bv`` that always returns
    ``bv_value`` (clamped to the kernel's legality constraint BV%16==0 and
    BV<=V which is 128).
    """
    assert bv_value in (16, 32, 64, 128), f"unexpected BV={bv_value}"

    def _fn(*args, **kwargs):
        return bv_value

    return _fn


def _orig_tuned_bv(args) -> int:
    """Look up the existing tuned BV for this PREFILL_PARAMS shape (no patch).
    Used purely for the "csv BV" column in the output table.
    """
    H = args.Hv // args.tp
    Hg = args.Hk // args.tp
    # Match the runtime call exactly. ``use_h0`` mirrors
    # ``initial_state is not None`` in chunk_gated_delta_rule_fwd_h_flydsl --
    # _make_inputs(with_initial_state=True) always allocates h0, so use_h0=True.
    # For varlen, the runtime N is the number of packed sequences =
    # max_num_batched_tokens // full_prompt_len (rounded down by
    # _build_context_lens). For non-varlen N=1 (B=1, no packing).
    if args.is_varlen:
        N_val = max(1, args.max_num_batched_tokens // args.full_prompt_len)
    else:
        N_val = 1
    return _lap._lookup_tuned_bv(
        dtype_str=str(args.dtype),
        K=args.K,
        V=args.V,
        BT=args.BT,
        H=H,
        Hg=Hg,
        T_flat=args.max_num_batched_tokens,
        N=N_val,
        use_g=True,
        use_gk=False,
        use_h0=True,
        store_fs=bool(args.output_final_state),
        save_vn=True,
        is_varlen=args.is_varlen,
        wu_contig=True,
    )


def _time_shape_with_bv(args, bv: int) -> float:
    """Time one (shape, BV) combo. Returns per-iter FlyDSL kernel time in us."""
    context_lens = _build_context_lens(
        args.full_prompt_len, args.max_num_batched_tokens
    )
    k, _w_orig, _u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)

    # Monkey-patch the BV lookup just before the timed window.
    saved = _lap._lookup_tuned_bv
    _lap._lookup_tuned_bv = _force_bv(bv)
    try:

        def launch():
            chunk_gated_delta_rule_fwd_h_flydsl(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        # Warmup once outside the timed window so JIT compile + autotune
        # do not leak into _bench_fn's NUM_WARMUP=5.
        launch()
        torch.cuda.synchronize()
        return _bench_fn(launch)
    finally:
        _lap._lookup_tuned_bv = saved


def main():
    candidate_bvs = (16, 32, 64)
    print(f"NUM_WARMUP={NUM_WARMUP} NUM_ITERS={NUM_ITERS}")
    print(f"Sweeping BV ? {candidate_bvs} for {len(PREFILL_PARAMS)} shapes")
    print()

    bv_col_headers = "  ".join(f"{'BV=' + str(bv):>10}" for bv in candidate_bvs)
    header = (
        f"{'#':>3}  {'shape':<60}  {'csv BV':>6}  "
        f"{bv_col_headers}  {'best BV':>7}  {'best us':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for idx, args in enumerate(PREFILL_PARAMS):
        tag = PREFILL_TEST_IDS[idx]
        csv_bv = _orig_tuned_bv(args)
        per_bv = {}
        for bv in candidate_bvs:
            try:
                t = _time_shape_with_bv(args, bv)
            except Exception as e:
                print(f"[warn] shape={tag} BV={bv} failed: {e!r}")
                t = float("inf")
            per_bv[bv] = t
        best_bv = min(per_bv, key=lambda b: per_bv[b])
        best_us = per_bv[best_bv]
        results.append((tag, csv_bv, per_bv, best_bv, best_us))
        row = (
            f"{idx + 1:>3}  {tag:<60}  {csv_bv:>6}  "
            + "  ".join(f"{per_bv[bv]:>10.1f}" for bv in candidate_bvs)
            + f"  {best_bv:>7}  {best_us:>9.1f}"
        )
        print(row, flush=True)

    # Final summary.
    print()
    print("Summary:")
    csv_optimal = sum(
        1 for _, csv_bv, per_bv, best_bv, _ in results if best_bv == csv_bv
    )
    print(f"  shapes where csv BV == new best BV: {csv_optimal}/{len(results)}")
    differ = [
        (tag, csv_bv, best_bv, per_bv[csv_bv], best_us)
        for (tag, csv_bv, per_bv, best_bv, best_us) in results
        if best_bv != csv_bv
    ]
    if differ:
        print("  shapes where the new best BV differs from csv:")
        for tag, csv_bv, best_bv, csv_us, best_us in differ:
            delta = (best_us - csv_us) / csv_us * 100.0
            print(
                f"    {tag}: csv BV={csv_bv} ({csv_us:.1f} us) "
                f"-> new best BV={best_bv} ({best_us:.1f} us, {delta:+.1f}%)"
            )


if __name__ == "__main__":
    main()
