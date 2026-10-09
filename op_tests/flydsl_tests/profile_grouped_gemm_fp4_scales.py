#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Profile validated FP4 GEMMs in ROCTx regions with the unchanged CSV tiles.

Run under rocprofv3 --selected-regions. Preparation and warmup are excluded.
For ATT, each eager launch has its own region so --repeat captures each one.
"""

import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "scale32", "scale16", "nvunit", "nvfp4"),
        default="all",
    )
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--roctx-library", default="librocprofiler-sdk-roctx.so.1")
    args = parser.parse_args()
    if args.tokens < 1 or args.repeat < 1:
        parser.error("--tokens and --repeat must be positive")

    os.environ["AITER_MOE_EXPERT_BALANCE"] = "true"
    test_path = (
        Path(__file__).resolve().parents[1] / "test_flydsl_grouped_gemm_gfx1250.py"
    )
    spec = importlib.util.spec_from_file_location("grouped_test", test_path)
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    test._require_gfx1250()
    test.set_data_format("a4w4")

    import torch

    roctx = ctypes.CDLL(args.roctx_library)
    roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
    roctx.roctxRangePushA.restype = ctypes.c_int
    roctx.roctxRangePop.argtypes = []
    roctx.roctxRangePop.restype = ctypes.c_int
    for name in ("roctxProfilerPause", "roctxProfilerResume"):
        fn = getattr(roctx, name)
        fn.argtypes = [ctypes.c_uint64]
        fn.restype = ctypes.c_int
    roctx.roctxProfilerPause(0)

    config_keys = (
        "tile_m",
        "tile_n",
        "tile_k",
        "m_warp",
        "n_warp",
        "num_buffers",
        "cluster_n",
        "waves_per_tensor_tdm",
        "next_stage_prefetch",
    )
    prepared = {}
    common_configs = None
    for name, block, fmt, tensor_scales in (
        ("scale32", 32, 0, False),
        ("scale16", 16, 0, False),
        ("nvunit", 16, 2, False),
        ("nvfp4", 16, 2, True),
    ):
        if args.mode not in ("all", name):
            continue
        hooks = []

        def collect(fn, *, hooks=hooks, **_kwargs):
            hooks.append(fn)
            return fn(), 0.0

        with patch("aiter.test_common.run_perftest", collect):
            out, ref, _, _ = test._run_grouped_via_fused_moe(
                experts=96,
                tokens=args.tokens,
                topk=6,
                model_dim=7168,
                inter_dim=3072,
                data_format="a4w4",
                activation=test.ActivationType.Silu,
                use_bias=False,
                kernel_bench=True,
                scale_block_size=block,
                scale_format_a=fmt,
                scale_format_b=fmt,
                global_scale_a1=0.7 if tensor_scales else 1.0,
                global_scale_a2=1.3 if tensor_scales else 1.0,
                global_scale_w1=0.8 if tensor_scales else 1.0,
                global_scale_w2=1.1 if tensor_scales else 1.0,
            )
        assert len(hooks) == 2, "Expected GEMM1 and GEMM2 benchmark hooks"
        configs = [{key: fn.keywords[key] for key in config_keys} for fn in hooks]
        if common_configs is None:
            common_configs = configs
            print(f"Shared CSV launch settings: {configs}", flush=True)
        assert configs == common_configs, "Scale modes must use identical tiles"
        diff = test._logits_diff(out, ref)
        assert diff < test.LOGITS_DIFF_TOL, (name, diff)
        prepared[name] = hooks
        print(f"Prepared {name}: logits_diff={diff:.6g}", flush=True)
        del out, ref

    def profile(fn, region):
        roctx.roctxProfilerResume(0)
        roctx.roctxRangePushA(region.encode())
        try:
            fn()
            torch.cuda.synchronize()
        finally:
            roctx.roctxRangePop()
            roctx.roctxProfilerPause(0)

    for name, hooks in prepared.items():
        for stage, fn in enumerate(hooks, start=1):
            if args.stage not in (0, stage):
                continue
            for _ in range(8):
                fn()
            torch.cuda.synchronize()
            region = f"{name}_gemm{stage}"
            if args.graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.repeat):
                        fn()
                graph.replay()
                torch.cuda.synchronize()
                profile(graph.replay, region)
            else:
                for repeat in range(args.repeat):
                    profile(fn, f"{region}_rep{repeat}")
    print("Profile complete", flush=True)


if __name__ == "__main__":
    main()
