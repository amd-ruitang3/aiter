# FP4 scale access patterns and fixed-tile optimization attempts

Real scale16 remains slower than scale32 with the existing CSV tiles. Several
assembly-guided probes pass exact correctness, but none closes that gap. The
retained change uses native gfx1250 FP8 instructions to encode and decode the
new E4M3 block scales produced by the quantizer. It removes software conversion
work and gives a small GEMM1 improvement at 16k tokens.

All measurements use 96 experts, topk 6, dimensions 7168/3072, balanced routing,
SiLU, no bias, and **64 × 512 × 128** tiles for both stages. Waves remain 1 × 4,
payload buffers 2, cluster 1, TDM owners 2, and next-stage prefetch 0. The
experiments change instructions or data layout, never the CSV or tile size.
Measurements are sequential on gfx1250, 256 CUs, on 2026-10-09.

## E4M3 is native in WMMA; the conversion is in the output quantizer

The main loop transfers already encoded block-scale bytes from global memory
through LDS to registers. It passes them directly to:

```text
v_wmma_scale16_f32_32x16x128_f4 ...
  matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3
  matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
```

There is no per-K software E4M3 conversion. ISA matrix A is the logical weight
matrix, and ISA matrix B is the logical activation matrix. NEG and NEG_HI select
their respective formats, as described in the earlier scaling report.

GEMM1 also produces the quantized input for GEMM2. Its epilogue computes a new
floating-point `amax`, divides by `6 * global_scale_a2`, and rounds that value
into an E4M3 scale byte. The rounded scale must then be used to normalize the
output values before packing them as FP4. Native WMMA support does not encode
a new output scale from `amax`.

Previously, `emit_fp4_block_scale` encoded and decoded E4M3 using integer
bit operations and floating-point subnormal handling. The retained path uses
`v_cvt_pk_fp8_f32` to encode and `v_cvt_f32_fp8` to decode. Zero retains the
existing safe divisor of 1, and saturation remains 448. E5M3 keeps its existing
unsigned ISA representation and encoder.

A separate instruction control checks why the f32 normalization is retained.
For six scale values, 64 packed words per scale compare passing the divisor
directly to `v_cvt_scalef32_pk8_fp4_f32` against explicit f32 normalization:

| Scale | Different FP4 words |
|---:|---:|
| 1.0 | 0 / 64 |
| 1.25 | 38 / 64 |
| 1.5 | 51 / 64 |
| 1.75 | 52 / 64 |
| 2.0 | 0 / 64 |
| 0.75 | 51 / 64 |

Passing each scale directly matches passing `2^floor(log2(scale))` in all
384 words. On this GPU the pack's scale operand behaves as an exponent scale;
it does not provide general E4M3-mantissa normalization. This is distinct from
WMMA's native E4M3 block-scale operand. Using the complete block and tensor
scale to normalize in f32 preserves the required quantization behavior.

The native encoder/decoder matches PyTorch exactly on 99,837 values across
three tensor scales (1.0, 0.7, 1.3): all positive finite E4M3 levels, every
midpoint and its adjacent f32 values, subnormals, saturation, and deterministic
random values over a broad exponent range. GEMM1 payload bytes, intermediate
scale bytes, and GEMM2 output also match the software helper exactly for all
three token counts.

## Actual global and LDS access patterns

The preshuffled weight-scale layout is:

```text
[E, N32, K128, lane32, scale_bytes]
scale_bytes = 4 for scale32, 8 for scale16
```

For a K128 step, the N512 tile reads 16 N32 groups. Each group is contiguous,
but the groups are separated by the complete K extent:

| Stage | Scale32 group slice / row stride | Scale16 group slice / row stride |
|---|---:|---:|
| GEMM1, K7168 | 128 B / 7168 B | 256 B / 14336 B |
| GEMM2, K3072 | 128 B / 3072 B | 256 B / 6144 B |

These strides are **bytes**; the TDM descriptor expresses them in dwords.
Weight payload N16 groups have separate strides of 57,344 B and 24,576 B.
Adjacent K steps revisit adjacent slices of each group. The original layout
already consumes complete aligned scale slices, rather than issuing one tiny
request for each scale byte.

Each scale LDS operand uses address `base + lane * 4` or `base + lane * 8`.
A full wave spans 128 B or 256 B contiguously per N32 group. The compiler merges
six logical scale reads into three `ds_load_2addr_b32` / `ds_load_2addr_b64`
instructions. Payload accesses remain 24 `ds_load_b128`, and arithmetic remains
16 WMMA instructions per K128 per wave. Wider scales double transferred data
without doubling the scale instruction count.

Natural contiguous addressing is not itself evidence of a bank conflict.
The earlier full-kernel ATT captures show close LDS wait durations, and the
new software/native E4M3 comparison also gives about 8110 / 8076 LDS-wait cycles
per complete sampled wave. No supported gfx1250 LDS bank-conflict counter is
exposed by this SDK. Invalid zero SQ counters are not used to claim zero
conflicts. Two-address load service and contention with payloads remain possible,
but the evidence does not identify a large new scale16 bank penalty.

The common exposed dependency is:

```text
wait for current TDM jobs -> workgroup ready barrier
LDS reads -> s_wait_dscnt thresholds -> WMMA
workgroup reuse barrier -> payload and scale TDM jobs for k+2
```

The current assembly retains the buffer until all 16 WMMA instructions have
issued, even though the operands reside in registers. Releasing it after the
LDS reads is a plausible overlap opportunity. The probes below test it directly.
The earlier memory counters still show roughly 5.9% more L2-to-memory read bytes
for scale16; this transport cost remains after quantizer optimizations.

## Measured attempts

Seven randomized interleaved graph rounds use the same prepared tensors per
mode. Each round captures 16 launches, warms up three replays, and takes the
median of five event measurements. Every probe below passes exact G1 payload,
scale-byte, and G2-output comparisons at 512 and 16,384 tokens. Unwritten
padding is zeroed before comparison. Each experiment has its own baseline;
compare changes within a row, rather than absolute timings across experiments.

The table gives the change in the sum of GEMM1/GEMM2 medians; negative is faster.

| Probe | Scale16, 512 | NVFP4, 512 | Scale16, 16k | NVFP4, 16k |
|---|---:|---:|---:|---:|
| Batch two K128 scale slices per transfer | +1.67% | +0.16% | +1.96% | +2.22% |
| Batch four slices | +1.18% | +0.22% | +1.48% | +1.70% |
| K-major contiguous weight-scale layout | +0.46% | −0.68% | −0.05% | −0.08% |
| Balanced `amax` reduction tree | +0.15% | +0.22% | +0.00% | +0.05% |
| Recycle LDS before all WMMA | −0.58% | +0.20% | −0.30% | −0.35% |
| Recycle after eight WMMA | +0.09% | +0.65% | +0.21% | +0.24% |
| Skip each entirely padded M16 fragment | −1.59% | −2.14% | +0.87% | +0.96% |
| Skip entirely padded back-half fragments | −2.82% | −2.64% | +2.52% | +2.88% |
| Pack weight scales beside weight payload | +0.62% | −0.37% | −0.44% | −0.44% |

Batching reduces scale TDM job count but raises scale16 LDS from 84,992 B to
94,208 B / 112,640 B and adds indexing and wait dependencies. It is slower.
K-major packing changes the layout to `[E, K128, N32, lane32, scale_bytes]`
and makes each N512 scale transfer contiguous without extra LDS. It brings no
consistent scale16 improvement and makes scale32's 16k sum about 2.02% slower.
Both layout conversions occur during preparation and are excluded from timing;
production adoption would need an explicit compatible packing API.

The reduction-tree probe does not simplify the resulting ISA. The original
serial source already becomes `v_max3_num_f32` groups interleaved with SiLU.
The tree reduces the number of max3 instructions and adds two-input max
instructions, increasing the static GEMM1 instruction count by 18. It is not
retained.

Early reuse changes the assembly as intended: the next TDM loads issue before
0 / 8 / 16 completed WMMA instructions for recycle0 / recycle8 / baseline.
The barrier enforces completion of the LDS reads before overwriting the buffer.
Full early reuse introduces an immediate `s_wait_dscnt 0`; it trades the
baseline's LDS/WMMA overlap for earlier transfers. Its 16k gain is small and
also helps scale32 by 0.43%, so it does not close the relative gap.

At 512 tokens there are 32 valid routes per expert in an M64 tile. Skipping
padding avoids half of the matrix work while retaining the tile. The gain is
about 2–3% for every scale mode, with almost the same relative scale gap.
Runtime control changes register allocation and instruction scheduling;
full tiles at 16k regress. These probes are not retained.

Joint weight/scale packing replaces separate B/SB transfers with one TDM job
per weight owner. Each N32/K128 record is 2048 payload bytes plus 128/256 scale
bytes. Activation owners still issue payload and scale separately, and waits
account for the unequal per-wave job counts. This preserves payload bytes and
keeps the same LDS allocation, but the new LDS offsets inhibit some merging:
three two-address scale reads become four single b64 reads plus one two-address
b64 read. The measured gain is only 0.44% at 16k, including about 0.43% for
scale32. It needs a new weight format and is not retained.

## Retained native conversion: assembly, trace, and final comparison

The native E4M3 path removes 350 static instructions from the fixed GEMM1 ELF
(4163 to 3813 instruction lines), replacing software rounding and decode with
16 FP8 encodes and 16 FP8 decodes across the compiled epilogue paths. It keeps
376/374 VGPRs, no spills, and the same 84,992 B LDS for scale16/NVFP4.

A new ATT capture contains three eager GEMM1 dispatches per revision, on CU1,
SIMD1, shader-engine mask3. The retained sample has 30 software and 27 native
waves, each with the complete 896 WMMA instructions. Its median epilogue span
falls from **3906.5 to 3675 cycles**, and epilogue instruction count from
**1255 to 1151**. The packed scale-store path is unchanged. Tensor/barrier waits
vary substantially across this WGP sample, so the epilogue reduction is evidence
of less quantizer work, not a prediction of global kernel latency.

The final source is compared with the software helper over 11 randomized
interleaved graph rounds. Scale32/E8M0 paths are unchanged instruction-wise;
small differences in their timings indicate measurement variation.

| Tokens | Software NVFP4 G1/G2 (µs) | Native NVFP4 G1/G2 (µs) | NVFP4 sum change |
|---:|---:|---:|---:|
| 512 | 157.04 / 82.39 | 158.03 / 81.94 | +0.22% |
| 2048 | 278.68 / 147.58 | 276.98 / 148.09 | −0.28% |
| 16384 | 1631.45 / 849.09 | 1625.79 / 848.90 | −0.24% |

At 16k, the new GEMM1 median is about 5.67 µs faster (0.35%). The small-token
comparison does not establish a speedup. The native path is retained because
it replaces unnecessary software E4M3 encoding/decoding with native operations,
passes rounding and exact-output checks, and reduces the measured epilogue.
It does not establish fixed-tile parity.

Final same-run graph medians:

| Tokens | Scale32 G1/G2 (µs) | Scale16 G1/G2 (µs) | Native NVFP4 G1/G2 (µs) | Scale16 / NVFP4 summed gaps |
|---:|---:|---:|---:|---:|
| 512 | 149.68 / 78.55 | 155.87 / 82.59 | 158.03 / 81.94 | +4.48% / +5.14% |
| 2048 | 254.79 / 140.77 | 274.05 / 147.05 | 276.98 / 148.09 | +6.46% / +7.46% |
| 16384 | 1472.86 / 804.38 | 1603.33 / 840.71 | 1625.79 / 848.90 | +7.32% / +8.67% |

The useful remaining direction is overlap that preserves the baseline's
LDS/WMMA schedule while advancing scale transfer completion. Simply preloading
more scales, flattening their layout, or moving the reuse barrier is insufficient
in these measurements. A larger reduction would need to reduce scale transport
or exploit reuse without changing numerical block scales. At 512, unique
weights are streamed once per expert; a warm-cache parity expectation is not
supported by that access pattern.

## Validation and artifacts

The 25 existing scale cases and three new native rounding cases pass with
caching disabled. Six relevant activation/FP8 JIT regressions also pass when
run in the requested no-AOT-cache mode, with the data format selected per case.
AOT-only checks cannot run against the missing caches in this workspace.
Ruff and formatting checks pass for the changed Python sources.

[grouped_gemm_fp4_access_results.json](grouped_gemm_fp4_access_results.json)
contains raw samples, launch settings, source hashes, reconstructable probe
diffs against `4ed1a5fdd`, resource metadata, the FP4 divisor control, and the
verified ATT aggregates. Temporary prototypes are not part of the production
kernel. Older trace/counter evidence and capture commands remain in
[grouped_gemm_fp4_trace.md](grouped_gemm_fp4_trace.md).

A helper-only edit can leave an old nested-kernel disk artifact in this FlyDSL
installation. The final capture verifies the native conversion opcodes in the
actual binary, and the rounding tests were run with
`FLYDSL_RUNTIME_ENABLE_CACHE=0`. An initial capture containing the old conversion
opcodes was discarded. Production quantizer source changes also invalidate
its existing artifact; no stale timing is reported as native performance.
