# Grouped GEMM FP4 block scales on gfx1250

Measured on 2026-10-09, gfx1250 with 256 CUs, PyTorch 2.11.0 + ROCm 10.1.
The problem is 96 experts, 512/2048/16384 tokens, topk 6, model dimension 7168,
intermediate dimension 3072, SiLU, no bias, and balanced routing.

All modes use the same existing CSV setting: **64 × 512 × 128** for both
GEMMs, waves 1 × 4, two buffers, cluster 1, two waves per TDM tensor, and
no next-stage prefetch. The CSV has not been modified. Scale size and format
do not select different tiles.

## Performance

The first tables cover 512 tokens; larger token counts follow below.

Median of five interleaved rounds after correcting the FP4 weight-read
scheduler count and packing adjacent output scales into wider stores. These
changes leave the tile and launch settings unchanged.
Each profiler measurement uses the same
`run_perftest(..., num_warmup=5, num_iters=101, testGraph=False)` as the
requested `--scenario kernel` command.

| Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---|---:|---:|---:|---:|
| Scale32, E8M0 | 156.07 | 92.07 | 248.14 | baseline |
| Real scale16, E8M0 | 158.80 | 95.95 | 254.76 | +2.67% |
| NVFP4, scale16 E4M3 with float scales | 160.49 | 96.23 | 256.72 | +3.46% |

The sum is the two isolated GEMM medians. GEMM1 includes its fused
SiLU and intermediate quantization epilogue in this no-bias configuration.
Routing, input quantization, and gather/reduce are measured separately and
excluded from this sum.

Graph/event timing gives a second view of device performance. Each graph has
16 repeated launches, three warmup replays, and five event samples; the table
uses the median across five interleaved rounds.

| Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---|---:|---:|---:|---:|
| Scale32, E8M0 | 150.04 | 79.75 | 229.79 | baseline |
| Real scale16, E8M0 | 155.67 | 82.88 | 238.55 | +3.81% |
| NVFP4, scale16 E4M3 with float scales | 158.13 | 83.36 | 241.49 | +5.09% |

Profiler timings are close, but graph timing still shows overhead with the
same tiles. These results do not establish exact equality. All samples and
launch settings are in [grouped_gemm_fp4_scaling_results.json](grouped_gemm_fp4_scaling_results.json).

Separate CLI invocations varied substantially: scale32 GEMM1 ranged from
159.07 to 184.11 µs, and scale16 from 176.24 to 186.78 µs in the observed
runs. The interleaved medians above compare modes within one prepared
process; individual CLI timings can differ from them.

## Why an earlier 512-token result appeared equal

Two earlier comparisons need to be distinguished. The 12:39 UTC temporary
comparison reported graph GEMM1/GEMM2 medians of 150.43/79.61 µs for scale32,
147.42/78.05 µs for scale16, and 147.80/80.58 µs for NVFP4. That harness did
not record or assert launch settings. Retained JIT artifacts from its
preparation show **different wave tiles**: the scale32 input quantizer compiled
at 12:39:26 uses `r4`, whereas the NVFP4 input quantizer compiled at 12:39:27
uses `r2`. Here `r = tile_m / (m_warp * 16)`; it fixes the preshuffled
activation layout consumed by GEMM1. The same-CSV comparison at 12:42 instead
compiled both scale16 input quantizers with `r4`.

The retained GEMM cache also includes scale16 E8M0 kernels with
32 × 256 × 128 tiles for both stages, and non-unit-float NVFP4 kernels with
32 × 256 × 128 for GEMM1 and 32 × 512 × 256 for GEMM2. Those artifacts have
128/126 and 128/248 VGPRs, respectively, versus 360/358 for the original
64 × 512 × 128 kernels. These are artifacts of the earlier experiments;
the exact launcher snapshot and dispatch trace for the 12:39 comparison
were not retained, so every launch setting cannot be reconstructed from
cache presence alone. The quantizer artifacts created during preparation
establish that the early comparison did not use identical wave tiles.

At 512 balanced tokens, each expert has only 32 valid routes. A smaller M
tile can avoid the padded half of an M64 tile and reduce accumulator pressure,
offsetting scale16 costs. The 12:39 result therefore cannot establish
scale16/scale32 parity under the user's fixed CSV configuration.

The **first committed fixed-CSV result** (`db7bd5621`, collected at 12:42)
was close under the CLI's eager PyTorch-profiler method: 252.14 versus
254.34 µs summed GEMMs, or +0.87% for scale16. Its graph measurements were
already 230.81 versus 240.90 µs, or **+4.37%**. The profiler averages filtered
device events from eager Python launches; graph timing measures repeated
device launches with HIP events. These produce different dispatch conditions.
The difference is not simply Python launch time added to a device event,
and the saved run has insufficient evidence to attribute its near-parity to
one specific clock, cache, or profiler effect.

A new comparison loads the kernel source from each commit and runs it on
the **same prepared tensors per scale mode**, with launch overrides removed
and the shared CSV settings asserted. All three versions produce bit-exact
GEMM1 payload bytes, intermediate scale bytes, and GEMM2 outputs; unwritten
padding is zeroed before comparison. Seven randomized interleaved graph
rounds and five eager-profiler rounds give:

| Kernel revision | Scale32 graph G1/G2 (µs) | Scale16 graph G1/G2 (µs) | Scale16 summed gap | NVFP4 graph G1/G2 (µs) |
|---|---:|---:|---:|---:|
| Original `db7bd5621` | 150.38 / 79.37 | 158.02 / 82.56 | +4.71% | 159.34 / 82.51 |
| Read-count correction `9c7e589e9` | 150.25 / 78.82 | 155.59 / 83.07 | +4.19% | 158.57 / 82.20 |
| Packed stores `bb5d1f80f` | 148.95 / 78.59 | 155.48 / 82.22 | +4.47% | 158.10 / 83.12 |

In this same-input run, original/current scale16 profiler summed gaps are
+3.34%/+2.43%. Reverting the scheduling and packed-store changes does not
recover fixed-tile parity. The current scale16 graph sum is 237.71 µs versus
240.58 µs for the original; this comparison shows no scale16 regression from
the retained changes. Raw samples, full revision IDs, historical results, and
the relevant JIT metadata are in
[grouped_gemm_fp4_version_results.json](grouped_gemm_fp4_version_results.json).

## Larger token counts

The 2,048 and 16,384-token cases also use **64 × 512 × 128** for both
GEMMs, as specified by their existing a4w4 CSV rows. All three scale modes
use identical tiles, wave counts, buffers, and scheduling settings. Measurements
use the same five-round interleaved procedure as the 512-token case.

Profiler timing:

| Tokens | Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---:|---|---:|---:|---:|---:|
| 2048 | Scale32, E8M0 | 251.36 | 148.09 | 399.45 | baseline |
| 2048 | Real scale16, E8M0 | 269.72 | 146.52 | 416.24 | +4.20% |
| 2048 | NVFP4, scale16 E4M3 with float scales | 271.45 | 150.96 | 422.41 | +5.75% |
| 16384 | Scale32, E8M0 | 1469.52 | 798.32 | 2267.84 | baseline |
| 16384 | Real scale16, E8M0 | 1597.62 | 834.48 | 2432.11 | +7.24% |
| 16384 | NVFP4, scale16 E4M3 with float scales | 1625.24 | 845.32 | 2470.55 | +8.94% |

Graph/event timing:

| Tokens | Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---:|---|---:|---:|---:|---:|
| 2048 | Scale32, E8M0 | 256.16 | 140.86 | 397.02 | baseline |
| 2048 | Real scale16, E8M0 | 273.86 | 148.21 | 422.07 | +6.31% |
| 2048 | NVFP4, scale16 E4M3 with float scales | 280.53 | 148.41 | 428.94 | +8.04% |
| 16384 | Scale32, E8M0 | 1471.29 | 803.14 | 2274.43 | baseline |
| 16384 | Real scale16, E8M0 | 1599.03 | 838.20 | 2437.23 | +7.16% |
| 16384 | NVFP4, scale16 E4M3 with float scales | 1629.02 | 848.98 | 2478.00 | +8.95% |

At 16,384 tokens, profiler and graph timing agree closely. Scale16 GEMM1
has about 8.7% overhead and GEMM2 about 4.4%; NVFP4 has about 10.7% and
5.7%, respectively. The 512-token profiler alignment does not persist at larger
token counts with these fixed tiles.

All six larger cases pass correctness. Scale32 `logits_diff` is about
3.40e-6, scale16 at most 3.14e-11, and NVFP4 at most 1.78e-5, versus
the existing 0.01 gate. NVFP4 uses the same float scales as the 512-token
comparison: A1=0.7, A2=1.3, W1=0.8, W2=1.1.

Raw samples and launch settings:
[2,048-token results](grouped_gemm_fp4_scaling_tokens2048_results.json) and
[16,384-token results](grouped_gemm_fp4_scaling_tokens16384_results.json).

Deeper ablations, ISA resource checks, and scale preload/LDS experiments are
in [the NVFP4 analysis](grouped_gemm_nvfp4_analysis.md) and
[the decoded trace/counter investigation](grouped_gemm_fp4_trace.md). Packing
output scales removes a measured store wait; exact scale16/scale32 parity is not
achieved with the fixed CSV tiles.

The latest [assembly/access-pattern probes](grouped_gemm_fp4_access.md) retain
native E4M3 encoding/decoding in the output quantizer, with unchanged CSV tiles.
They reduce GEMM1's sampled epilogue and save about 0.35% at 16k tokens; the
scale16 transport gap remains. That report contains updated 11-round timings.

## Correctness and representation

Scale16 stores **eight independent scale bytes per K128 row** and loads them
as a 64-bit WMMA scale operand. It emits
`v_wmma_scale16_f32_32x16x128_f4`. Adjacent K16 scales can differ; the
implementation does not duplicate each K32 scale.

Scale format 0 selects E8M0, 1 selects unsigned E5M3, and 2 selects E4M3.
The ISA matrix A format uses NEG and matrix B uses NEG_HI. The kernel's
transposed WMMA order maps logical weights to ISA matrix A and logical
activations to ISA matrix B, so it swaps the logical A/B format parameters
when emitting WMMA. Disassembly confirms SCALE16 and E4M3 on both operands
for NVFP4.

Each operand also has a float tensor scale. GEMM1 multiplies its accumulator
by `global_scale_a1 * global_scale_w1` before bias and activation; GEMM2 uses
`global_scale_a2 * global_scale_w2`. NVFP4 quantization normalizes in f32
before FP4 packing so E4M3 mantissas and float scales are honored.

| Mode | logits_diff vs reference |
|---|---:|
| Scale32 E8M0 | 3.40e-6 |
| Scale16 E8M0 | 4.54e-12 |
| NVFP4 with float scales | 1.79e-5 |

All pass the existing `logits_diff < 0.01` gate. All 25 scale tests pass,
covering mixed A/B formats, bias, exact quantized scale/payload bytes, zero inputs, and a
bit-exact scale32/scale16 GEMM comparison with deliberately repeated scales.
Existing FP8 activation and SiLU/Swiglu/SiTUv2 regressions pass. Custom FP4
scales currently apply to the gfx1250 grouped INTERLEAVE path. AOT wiring
covers auxiliary kernels; the existing GEMM AOT wiring remains incomplete.

## Reproduce

Run the original command once per scale block size:

```bash
for block in 32 16; do
  AITER_MOE_EXPERT_BALANCE=true python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario kernel --data-format a4w4 --experts 96 --tokens 512 --topk 6 \
    --model-dim 7168 --inter-dim 3072 --act silu --no-bias --no-check-aot-cache \
    --scale-block-size "$block"
done
```

For NVFP4, use the same command with `--nvfp4` and the tested float scales:
`--global-scale-a1 0.7 --global-scale-a2 1.3 --global-scale-w1 0.8 --global-scale-w2 1.1`.
Independent formats are selectable with `--scale-format-a` and
`--scale-format-b` (`e8m0`, `e5m3`, or `e4m3`). Float scales default to 1.0.

The controlled comparison checks that all modes launch with identical settings:

```bash
python3 op_tests/flydsl_tests/benchmark_grouped_gemm_fp4_scales.py \
  --rounds 5 --output /tmp/fp4_scaling_results.json

for tokens in 2048 16384; do
  python3 op_tests/flydsl_tests/benchmark_grouped_gemm_fp4_scales.py \
    --tokens "$tokens" --rounds 5 --output "/tmp/fp4_scaling_tokens${tokens}.json"
done
```

Unset tile/scheduling environment overrides to reproduce the CSV defaults.
