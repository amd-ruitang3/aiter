# 512-token FP4 trace, counters, and assembly on gfx1250

The main-loop traces point to memory transfer completion and workgroup
synchronization as the dominant exposed waits. They do **not** show a large
additional LDS-read or WMMA stall for scale16. The traces also identify a
separate GEMM1 epilogue problem: individual output-scale byte stores leave a
long `s_wait_storecnt 0`. Packing adjacent scale bytes removes most of that
wait. This fix is retained, but fixed-tile scale16/scale32 parity remains open.

All captures use 96 experts, 512 tokens, topk 6, K=7168, intermediate width
3072, balanced routing, SiLU, and no bias. Both GEMMs keep the CSV
**64 × 512 × 128**, waves 1 × 4, buffers 2, cluster 1, two TDM owners per
tensor, and next-stage prefetch 0. No tile tuning or CSV changes are involved.
GEMM1 includes its fused activation and intermediate FP4 quantizer; these
isolated measurements exclude input quantization, routing, and gather/reduce.

## Memory counters establish extra traffic, not a complete latency explanation

Eight graph launches per mode were collected in separate counter passes.
The table reports medians. Read bytes use the documented sizes of
`GL2C_EA_RDREQ_{32B,64B,128B,256B}_sum`; request latency uses
`GL2C_EA_RDREQ_LEVEL / GL2C_EA_RDREQ` as defined by the SDK.

| Metric | GEMM1 scale32 | GEMM1 scale16 | GEMM2 scale32 | GEMM2 scale16 |
|---|---:|---:|---:|---:|
| L2-to-memory read-request bytes | 2.169 GB | 2.297 GB | 1.083 GB | 1.147 GB |
| Change in read bytes | — | +5.92% | — | +5.90% |
| Read-request bandwidth during that counter pass | 13.39 TB/s | 13.61 TB/s | 13.42 TB/s | 12.79 TB/s |
| Mean read-request latency | 883.5 cycles | 905.5 cycles | 836.0 cycles | 871.9 cycles |
| L2 hit count | 1,550,852 | 1,055,462 | 757,343 | 498,422 |
| L2 miss count | 8,491,816 | 8,991,997 | 4,396,812 | 4,645,846 |

NVFP4 E4M3 with unit/non-unit float scales has approximately the same read
bytes as E8M0 scale16. Format selection does not add another matrix read.
The undocumented units of `GL2C_EA_RDREQ_DRAM` are preserved in the JSON
but are not converted to bytes. FIFO-full counts do not increase consistently;
they do not identify a new queue-full bottleneck.

The scale bytes are significant for FP4: a weight row has 16 payload bytes
and one scale byte per K32. Scale16 doubles that scale portion. For one
64 × 512 × 128 tile, payload bytes are 36,864 and scale bytes grow from
2304 to 4608, increasing total input bytes by 5.88%. At 512 tokens there
are only 32 valid routes per expert, so each expert still needs its full
weights; unique packed GEMM1 weights alone occupy 2.114 GB. There is little
reuse across M tiles in this case. The measured byte increase is therefore
substantial even though the additional data are called scales.

However, byte growth alone cannot explain a particular 6.11% timing result.
Counter collection changes dispatch timing, and the GEMM has overlapping
requests, barrier tails, and an output quantizer. The bandwidth above comes
from the **same counter pass**, not from dividing counters by a faster
unprofiled timing. Request latency and the decoded waits below provide the
additional evidence that this investigation originally lacked.

## Decoded thread traces: TDM/barriers dominate; LDS waits stay close

ATT captured three launches of every mode on target CU/WGP 1, SIMD 1,
shader-engine mask 3. The decoder emitted active waves for SE0; the second
engine files had no decoded instructions. Waves with no WMMA are padding
expert waves and are excluded. Every retained GEMM1 wave executes 896 WMMA
instructions (56 K128 tiles × 16); GEMM2 executes 384 (24 × 16).

The main-loop interval starts at the first tensor wait and ends at the last
WMMA. Entries below are **mean accumulated instruction durations per active
sampled wave**, in cycles, before the output-store fix. They are not global
kernel times. Workgroup waves divide TDM ownership, so time can appear in
either the owner's tensor wait or another wave's barrier wait.

| Stage/mode | Active waves | Tensor + barrier waits | LDS waits | WMMA |
|---|---:|---:|---:|---:|
| GEMM1 scale32 | 25 | 60,479 | 8205 | 9016 |
| GEMM1 scale16 | 27 | 63,069 | 8106 | 8927 |
| GEMM1 E4M3 unit floats | 28 | 58,397 | 8171 | 8937 |
| GEMM1 E4M3 non-unit floats | 26 | 66,561 | 8089 | 8944 |
| GEMM2 scale32 | 33 | 24,854 | 3490 | 3876 |
| GEMM2 scale16 | 33 | 25,450 | 3458 | 3852 |
| GEMM2 E4M3 unit floats | 31 | 26,640 | 3480 | 3865 |
| GEMM2 E4M3 non-unit floats | 35 | 26,449 | 3463 | 3876 |

The two earlier single-dispatch captures on SIMD0 and SIMD1 also show
similar LDS waits/WMMA durations and large tensor/barrier waits. The
per-dispatch TDM variation is too large to assign the entire overhead from
this small WGP sample. In particular, E4M3 unit-float GEMM1 having a lower
wait total does not prove that E4M3 makes memory faster.

There are three merged scale LDS instructions per K128 in both modes:
`ds_load_2addr_b32` becomes `ds_load_2addr_b64`. Scale16 returns twice the
data, while the number of scale instructions remains 168/72 per sampled
GEMM1/GEMM2 wave. A load instruction's short decoded duration measures
issue-related work; asynchronous completion appears in `s_wait_dscnt`.
Comparing those waits gives no evidence of a large new scale16 LDS penalty.
It cannot exclude all bank contention or register-bank interactions.

The profiler's exposed gfx1250 event catalog has no LDS bank-conflict event.
SQ instruction and wait counters return zero for both known LDS and WMMA
work, including an eager non-graph control. Those zero values are invalid
for this attribution; they do **not** mean zero conflicts. `SQ_WAVES` and
`SQ_BUSY_CYCLES` return nonzero values. No unsupported event from another
architecture was substituted.

## What the main-loop assembly actually waits on

The fixed K128 pipeline primes two LDS buffers. The steady loop is:

```text
s_wait_tensorcnt 2
s_barrier_signal -1 / s_barrier_wait 0xffff
24 × ds_load_b128 payload reads + 3 × two-address scale reads
s_wait_dscnt ... / 16 × v_wmma_scale[16]_f32_32x16x128_f4
s_barrier_signal -1 / s_barrier_wait 0xffff
tensor_load_to_lds payload for tile k+2
tensor_load_to_lds scales for tile k+2
```

The next payload and scale jobs issue after the current compute finishes and
the reuse barrier releases its buffer. They overlap the other buffered
tile, but the next tile cannot consume LDS until **both** payload and scales
have completed across the four workgroup waves. Scale traffic can therefore
extend the tile's completion tail; being memory bound does not guarantee
that an extra transfer is hidden.

In the final GEMM2 ELF, scale32's first readiness wait is at `0x28f0`
(`s_wait_dscnt 0x13`), followed by WMMA at `0x28f4`. Scale16 uses
`0x2904` (`s_wait_dscnt 0x12`), followed by WMMA at `0x2908`. The threshold
changes because the needed scale operands occur at different points in the
outstanding read sequence. It does not itself imply a bank conflict or
give a cycle count. Payload addresses and accumulator allocation are the
same; scale16 uses VGPR pairs for eight encoded bytes per row.

The final binaries have 376/374 VGPRs for GEMM1/GEMM2 in every mode, no
VGPR/SGPR spills, and LDS 80,896 B for scale32 versus 84,992 B for scale16.
This rules out a mode-specific register-count cliff. The wider operands
still could interact with register banks, but neither ATT WMMA durations
nor the register-resident instruction probe shows a large intrinsic cost.

Two targeted TDM probes kept the tile/launch settings fixed: issue scales
before payloads, and move activation-scale ownership to the weight waves
while moving weight-scale ownership to the activation waves. Both pass
correctness. The scales-first probe shows no consistent GEMM1 benefit at
512 or 16k; changing ownership also fails to close the gap. Neither is
retained. Whole-stage and direct-global scale preloads from the earlier
investigation also remained slower.

## A confirmed output-scale store issue and the retained fix

Before the fix, each valid GEMM1 row stores each block-scale byte separately.
With this tile, a scale32 wave emits four byte stores over its two valid
M16 rows; scale16 emits eight. Their lane addresses have four/eight-byte
stride, with adjacent bytes of the same row written by separate instructions.
The store wait is explicitly after the TDM payload store, so it remains an
exposed tail at kernel exit.

The fix packs adjacent bytes within one existing preshuffled scale group.
Scale32 emits two 16-bit stores; scale16 emits two 32-bit stores. Smaller
legal N tiles retain a two-byte or one-byte fallback. Row masking, bytes,
tensor-scale math, and the CSV selection stay the same.

| GEMM1 mode | Mean store wait before/after (cycles) | Median epilogue span before/after (cycles) |
|---|---:|---:|
| Scale32 | 127 / 114 | 2720 / 2727 |
| Scale16 E8M0 | 692 / 116 | 3727 / 3072 |
| E4M3 unit floats | 660 / 100 | 4402 / 3818 |
| E4M3 non-unit floats | 711 / 116 | 4619 / 3867 |

The earlier SIMD0/SIMD1 captures independently show the same long scale16
store wait before the fix (660–934 cycles versus 119–127 for scale32).
After packing, the assembly uses `global_store_b32` with `scale_offset`
addressing for scale16. A same-input comparison against the previous kernel,
after zero-initializing padding, verifies bit-exact GEMM1 FP4 payload and
intermediate scale bytes for scale32, scale16, and NVFP4.

A final three-dispatch capture using the committed profiling driver repeats
the result: GEMM1 scale16/E4M3 store waits are 108–120 cycles, and its LDS
waits remain 8084–8104 versus 8209 cycles for scale32. GEMM2 LDS waits are
3430–3475 versus 3502 cycles. This capture is also in the compact JSON.

Seven interleaved graph timing rounds show the practical size of this fix:

| Tokens/mode | GEMM1 before/after (µs) |
|---|---:|
| 512 scale16 | 155.33 / 156.52 |
| 512 NVFP4 + float | 158.58 / 157.64 |
| 16384 scale16 | 1605.57 / 1603.64 |
| 16384 NVFP4 + float | 1636.61 / 1632.33 |

The 512 timing effect is small and noisy, despite the clear sampled store
wait reduction. This is an epilogue improvement, not an explanation or fix
for the full memory-pipeline gap. Final production timing is recorded in
[the scaling report](grouped_gemm_fp4_scaling.md).

## E4M3 and the final float scale

A separate validated 512-token format ablation gives E8M0 scale16 at
155.40/82.28 µs, E4M3 unit globals at 156.44/82.09 µs, and E4M3 non-unit
globals at 157.38/81.71 µs for GEMM1/GEMM2. This rebuilds correctly encoded
inputs for each mode; it does not reinterpret the same bytes under another
format.

ATT shows GEMM1's E4M3 epilogue executing roughly 270 additional non-store,
non-wait, non-conversion instructions per active sampled wave versus E8M0
scale16, and another 94 for non-unit globals. It rounds/encodes the block
scale, decodes it, normalizes in f32, and packs FP4. GEMM2's non-unit path
adds roughly 128 accumulator multiplies plus control instructions per wave;
its epilogue span changes by only about 80 cycles in this sample. The final
tensor product is paid after WMMA, once per accumulator, not per K tile.
These observations agree with the larger-token format and float-only
ablations: scale transport dominates the common scale16 cost; E4M3 output
quantization adds a smaller GEMM1 cost, and the final float multiply is small.

## Reproduce and inspect

The profiling driver validates every prepared mode and asserts identical
CSV launch parameters. For graph traces/counters use `--graph`; for
repeated ATT captures omit it so every eager launch gets a separate ROCTx
region. A single range containing several eager launches only captures its
first dispatch with this SDK's selected-region ATT mode.

This machine's AQL library is installed as `.so.1`, while the profiler opens
the unversioned name. Temporary aliases make both the counter loader and
ATT decoder available without changing the SDK installation:

```bash
SDK_LIB=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib
mkdir -p /tmp/aiter-profile-libs
ln -sf "$SDK_LIB/libhsa-amd-aqlprofile64.so.1" /tmp/aiter-profile-libs/libhsa-amd-aqlprofile64.so
ln -sf "$SDK_LIB/librocprof-trace-decoder.so.0.2" /tmp/aiter-profile-libs/librocprof-trace-decoder.so
export LD_LIBRARY_PATH="/tmp/aiter-profile-libs:$SDK_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

timeout --signal=KILL 90s /opt/rocm/bin/rocprofv3 \
  --sdk-soversion 1 --disable-signal-handlers true --selected-regions true \
  --kernel-trace --marker-trace --hip-runtime-trace --output-format csv pftrace \
  -d /tmp/fp4-timeline -- \
  python3 op_tests/flydsl_tests/profile_grouped_gemm_fp4_scales.py --graph --repeat 32

timeout --signal=KILL 90s /opt/rocm/bin/rocprofv3 \
  --sdk-soversion 1 --disable-signal-handlers true --selected-regions true \
  --kernel-trace --pmc GL2C_EA_RDREQ_32B_sum GL2C_EA_RDREQ_64B_sum \
  GL2C_EA_RDREQ_128B_sum GL2C_EA_RDREQ_256B_sum --output-format csv \
  -d /tmp/fp4-counters/bytes -- \
  python3 op_tests/flydsl_tests/profile_grouped_gemm_fp4_scales.py --graph --repeat 8

timeout --signal=KILL 90s /opt/rocm/bin/rocprofv3 \
  --sdk-soversion 1 --disable-signal-handlers true --selected-regions true \
  --att --att-target-cu 1 --att-simd-select 1 --att-shader-engine-mask 3 \
  --att-buffer-size 33554432 --att-library-path /tmp/aiter-profile-libs \
  --kernel-include-regex a8w4_tdm_fp4 --output-format csv -d /tmp/fp4-att -- \
  python3 op_tests/flydsl_tests/profile_grouped_gemm_fp4_scales.py --repeat 3

python3 op_tests/flydsl_tests/summarize_grouped_gemm_fp4_profile.py \
  --att /tmp/fp4-att --counters /tmp/fp4-counters --output /tmp/fp4-summary.json
```

Additional independent passes use `GL2C_HIT GL2C_MISS GL2C_READ_SECTORS
GL2C_EA_RDREQ_DRAM` and `GL2C_EA_RDREQ_LEVEL GL2C_EA_RDREQ
GL2C_LATENCY_FIFO_FULL GL2C_SRC_FIFO_FULL`. Run benchmarks and profiler
captures sequentially on an otherwise idle GPU; profiling perturbs timing.

The committed [compact results](grouped_gemm_fp4_profile_results.json) contain
counter medians, decoded instruction aggregates, per-dispatch wave counts,
and raw interleaved timing samples for the targeted probes. Local original
artifacts are `/tmp/aiter-512-trace`, `/tmp/aiter-512-counters`,
`/tmp/aiter-512-att-repeated`, `/tmp/aiter-512-att-packed`, and
`/tmp/aiter-512-att-final`, with a final Perfetto timeline in
`/tmp/aiter-512-packed-trace-clean`. The timeline contains the retained kernel.
ELF/ISA exports are in
`/tmp/aiter-fp4-final-binaries`. Raw ATT and code objects are kept outside
Git because the captures are large.

Validation after the store fix: 25 scale tests, eight JIT activation/FP8
regressions, all nine 512/2048/16384-token production comparisons, bit-exact
same-input intermediate buffers, lint/format, and final ELF resource checks.
