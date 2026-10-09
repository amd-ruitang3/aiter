#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Summarize rocprofv3 counter CSVs and decoded ATT FP4 GEMM waves.

ATT instruction durations describe issue/execution and associated stalls;
asynchronous load completion is observed at waits, not at the load's duration.
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def instruction_group(asm):
    op = asm.split()[0]
    for prefix, group in (
        ("ds_load_2addr", "scale_lds"),
        ("ds_load", "payload_lds"),
        ("v_wmma", "wmma"),
        ("s_wait_tensor", "wait_tensor"),
        ("s_wait_ds", "wait_lds"),
        ("s_wait_store", "wait_store"),
        ("s_barrier_wait", "barrier_wait"),
        ("tensor_load", "tensor_load"),
        ("global_store", "global_store"),
    ):
        if op.startswith(prefix):
            return group
    return "other"


def summarize_att(root):
    kernels = defaultdict(list)
    for dispatch in sorted(root.glob("ui_output*")):
        code_path = dispatch / "code.json"
        if not code_path.exists():
            continue
        code = json.loads(code_path.read_text())["code"]
        if not code:
            continue
        name = code[0][0].removeprefix("; ")
        for path in sorted(dispatch.glob("se*_sm*_sl*_wv*.json")):
            instructions = json.loads(path.read_text())["wave"]["instructions"]
            wmma = [
                i
                for i, inst in enumerate(instructions)
                if instruction_group(code[inst[-1]][0]) == "wmma"
            ]
            # Padding expert waves have no main loop and must be excluded.
            if not wmma:
                continue
            first = next(
                i
                for i, inst in enumerate(instructions)
                if instruction_group(code[inst[-1]][0]) == "wait_tensor"
            )
            main = instructions[first : wmma[-1] + 1]
            epilogue = instructions[wmma[-1] + 1 :]
            categories = defaultdict(lambda: [0, 0, 0])
            for inst in main:
                _time, _category, stall, duration, index = inst
                value = categories[instruction_group(code[index][0])]
                value[0] += 1
                value[1] += duration
                value[2] += stall
            kernels[name].append(
                {
                    "dispatch": dispatch.name,
                    "wave": path.name,
                    "wmma_count": len(wmma),
                    "main_span_cycles": main[-1][0] + main[-1][3] - main[0][0],
                    "main_categories": dict(categories),
                    "epilogue_span_cycles": (
                        instructions[-1][0]
                        + instructions[-1][3]
                        - main[-1][0]
                        - main[-1][3]
                    ),
                    "epilogue_store_wait_cycles": sum(
                        inst[3]
                        for inst in epilogue
                        if instruction_group(code[inst[-1]][0]) == "wait_store"
                    ),
                    "epilogue_global_stores": dict(
                        Counter(
                            code[inst[-1]][0].split()[0]
                            for inst in epilogue
                            if instruction_group(code[inst[-1]][0]) == "global_store"
                        )
                    ),
                }
            )
    results = {}
    for name, waves in sorted(kernels.items()):
        groups = {group for wave in waves for group in wave["main_categories"]}
        categories = {}
        for group in sorted(groups):
            totals = [
                sum(wave["main_categories"].get(group, [0, 0, 0])[i] for wave in waves)
                for i in range(3)
            ]
            categories[group] = {
                "count_per_wave": totals[0] / len(waves),
                "cycles_per_wave": totals[1] / len(waves),
                "stall_cycles_per_wave": totals[2] / len(waves),
                "cycles_per_instruction": totals[1] / totals[0],
            }
        results[name] = {
            "active_waves": len(waves),
            "dispatches": len({wave["dispatch"] for wave in waves}),
            "wmma_counts": dict(Counter(wave["wmma_count"] for wave in waves)),
            "median_main_span_cycles": statistics.median(
                wave["main_span_cycles"] for wave in waves
            ),
            "main_categories": categories,
            "median_epilogue_span_cycles": statistics.median(
                wave["epilogue_span_cycles"] for wave in waves
            ),
            "mean_epilogue_store_wait_cycles": statistics.mean(
                wave["epilogue_store_wait_cycles"] for wave in waves
            ),
            "waves": waves,
        }
    return results


def summarize_counters(root):
    results = {}
    for path in sorted(root.rglob("*counter_collection.csv")):
        kernels = defaultdict(list)
        with path.open() as source:
            for row in csv.DictReader(source):
                kernels[row["Kernel_Name"]].append(row)
        group = str(path.parent.parent.relative_to(root))
        for name, rows in sorted(kernels.items()):
            values = defaultdict(list)
            durations = {}
            for row in rows:
                values[row["Counter_Name"]].append(float(row["Counter_Value"]))
                durations[row["Dispatch_Id"]] = (
                    int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                ) / 1000
            medians = {counter: statistics.median(v) for counter, v in values.items()}
            duration = statistics.median(durations.values())
            entry = {
                "dispatches": len(durations),
                "median_duration_us": duration,
                "counter_medians": medians,
            }
            sizes = (32, 64, 128, 256)
            if all(f"GL2C_EA_RDREQ_{size}B_sum" in medians for size in sizes):
                # Defined request sizes, not the undocumented RDREQ_DRAM units.
                total = sum(
                    size * medians[f"GL2C_EA_RDREQ_{size}B_sum"] for size in sizes
                )
                entry["read_request_bytes"] = total
                entry["read_request_TB_per_s"] = total / duration / 1e6
            if medians.get("GL2C_EA_RDREQ", 0):
                entry["mean_request_latency_cycles"] = (
                    medians["GL2C_EA_RDREQ_LEVEL"] / medians["GL2C_EA_RDREQ"]
                )
            results.setdefault(group, {})[name] = entry
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--att", type=Path, action="append", default=[])
    parser.add_argument("--counters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.att and not args.counters:
        parser.error("provide --att and/or --counters")
    results = {"att": {str(root): summarize_att(root) for root in args.att}}
    if args.counters:
        results["counters"] = summarize_counters(args.counters)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
