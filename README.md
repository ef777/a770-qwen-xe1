# Qwen3.8-27B on an Intel Arc A770 (16 GB): ~2.5x decode with custom SYCL/ESIMD kernels + MTP

Notes, patches and launch settings from a few weeks of tuning llama.cpp's SYCL backend for one
specific setup: **Intel Arc A770 16 GB, Windows 11, Qwen3.8-27B (hybrid Gated DeltaNet + attention)**,
used as a local coding agent in OpenCode. Two GGUFs are covered:

- **Qwen3.8-27B Q3_K_S**: almost all K-quants (mostly Q3_K).
- **Huihui-Qwen3.8-27B-abliterated UD-Q3_K_XL**: an Unsloth dynamic quant that is ~60% i-quants
  (IQ4_XS, IQ3_S, IQ3_XXS, IQ2_*, IQ4_NL). Before this work, those weights ran on SYCL's slow
  generic paths, so this model was ~30% slower than the Q3_K_S one. It is now on par.

The short version: decode speed went from ~20 t/s to 40-50 t/s, mostly because MTP speculative
decoding became usable on this card. Prefill improved less (the GEMM library is the ceiling).

This is a personal, hardware-specific fork, not a polished upstream PR. Several kernels are tuned
for Xe-HPG (DG2) and these models' shapes. Use at your own risk.

## Results

All numbers are from an A770 16 GB, Windows 11, oneAPI 2026.1, q8_0 KV cache, temperature 0
code generation after a filler context unless noted.

**Qwen3.8-27B Q3_K_S**

| Workload | Before | After |
|---|---|---|
| Code generation, 8k ctx, no speculation | 19.7 t/s | ~20 t/s (bandwidth bound) |
| Code generation, short ctx, MTP 3 drafts | MTP not worth using | **~50 t/s** |
| 16k server (ub768, MTP 3): tg at 1k / 8k / 15k ctx | - | **49.6 / 44.7 / 43.3 t/s** |
| 24k server (ub512, MTP 2): tg at 1k / 8k / 15k / 22k ctx | - | **43.0 / 42.4 / 41.3 / 39.0 t/s** |
| Prefill, 12.7k prompt (16k MTP server) | ~413 t/s | **~610-690 t/s** |
| OpenCode agent task (fix a module until 10 tests pass) | 73 s | **37-41 s** |
| 40-token appended prompt (agent tool output) at 12.7k ctx | ~860 ms | **~650 ms** |

**Huihui-Qwen3.8-27B-abliterated UD-Q3_K_XL (20k ctx, MTP 3 drafts)**

| Context depth | tg before | tg after | prefill before | prefill after (ub1024) |
|---|---|---|---|---|
| 1k | 30.2 | **40.9** | 279 | **581** |
| 8k | 29.3 | **41.5** | 384 | **687** |
| 15k | 28.6 | **39.1** | 390 | **666** |
| 19k | 28.0 | **39.5** | 383 | **648** |

The "before" columns for this model are this fork before the i-quant kernels (MTP 2 drafts, ub512).
The OpenCode agent task takes 40-41 s on it with all 10 tests passing.

"Before" for the Q3_K_S table means the same llama.cpp base with the open SYCL PRs listed below and
stock kernels.

For scale: the 27B Q3 model is ~11.5-12 GB of weights read per decode step, about 31 ms at the
A770's ~374 GB/s effective read rate. So without speculation, ~20 t/s is the hard ceiling. Most of
the decode gain comes from making the speculative-decoding verify step (3-4 tokens at once) cost
close to a single token, plus making the draft step itself cheap.

## What changed

All in `ggml/src/ggml-sycl` unless noted. ESIMD = Intel explicit SIMD (`[[intel::sycl_explicit_simd]]`).

### i-quant and K-quant matvec for the verify step (new, 2026-10-11)

A new kernel family in `dmmv.cpp` (`iqmv_*`) for the plain (not reordered) block layout of
**IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, Q4_K and Q5_K** (Q6_K is implemented but
off by default, since the existing reordered Q6_K kernel is just as fast):

- One thread per weight row, work-groups of 32 rows, 4 super-blocks per step, loaded with aligned
  block loads.
- The type's lookup grid (IQ4 values, IQ3/IQ2 grids) is copied to SLM once per work-group.
  IQ4 uses a 256-entry 16-bit table that decodes both nibbles of a byte with one gather.
- Activations are quantized to int8 per 32 values in SLM, shared by the work-group's 32 rows, and
  the dot products use `dp4a`. Signs of the IQ2/IQ3 grids are applied with a branch-free per-byte
  SWAR mask (`(g ^ m) + (m & 0x01010101)`). The Q4_K/Q5_K minimum term is subtracted using the
  activation group sums.
- 1-8 columns per call, so one weight read serves all the tokens of an MTP verify step.

At 3 columns (MTP verify), against the previous SYCL paths:

| Type | Before | After |
|---|---|---|
| IQ2_XXS / IQ2_XS / IQ2_S | | ~3x faster |
| IQ4_NL | | ~7x faster |
| IQ4_XS (17408x5120) | 245 us | 205 us |
| IQ3_S (17408x5120) | 211 us | 169 us |
| Q4_K (5120x17408) | 226 us | 148 us (~340 GB/s) |
| Q5_K | 119 us | 104 us |

The same block decoders also back a **dequantize-to-fp16 kernel** for prefill: weights are expanded
for the oneDNN GEMM through a hook in `ggml_get_to_fp16_sycl`. The generic converters were the main
reason the Huihui model's prefill was ~30% behind (380 -> 540 t/s at ub512, ~650-690 at ub1024).

Small Q8_0 matrices (the 48x5120 DeltaNet alpha/beta projections in the Huihui GGUF) now run on the
small matvec kernel as well.

Accuracy: perplexity at 16 chunks is unchanged with batched evaluation (2.5961 for both).
With 4-token batches, which go through the int8-activation path, it moves 2.5967 -> 2.6003
(the error bar is ±0.077).

What limits these kernels now is the number of SLM gathers: removing them (wrong results, timing
only) takes IQ4_XS from 188 to 129 us, and halving them gives 153 us, which is linear in the gather
count. See "Things that did not work" for the attempts to get around this.

### Decode / MTP verify (earlier rounds)

- **GQA flash-decode kernel** (`fattn-gqa-dec.cpp`) for 1-8 query tokens, q8_0/f16 KV, head size
  64/128/256. The old vector FA kernel re-read the KV cache once per query head (24 q heads / 4 kv
  heads). There is an ESIMD variant for q8_0 D=256 with batched K/V tile loads, and a work-group
  variant that dequantizes each K/V tile once into SLM for 2+ tokens: 12k n4 762 -> 404 us,
  24k n4 1302 -> 714 us. A merged partial-result reduction and a tuned chunk size for single tokens
  cut another 20% at 24k (n1 520 -> 410 us, n4 691 -> 583 us).
- **Multi-column dequantize-once matvec** for reordered K-quants (`dmmv.cpp`): one weight read
  serves up to 8 activation columns.
- **Q3_K matvec kernels**:
  - A wide-load fp32 kernel for 1-2 columns, at ~340 GB/s (91% of the measured read peak).
  - A dp4a kernel for 3-8 columns: SWAR dword dequant feeds the 3-bit values straight into dp4a,
    and the -4 offset is handled with 16-value activation sums in SLM.
- **Gated DeltaNet ESIMD kernel** for up to 8 tokens (`gated_delta_net.cpp`, decode), and the
  skipped state copy through `GET_ROWS`: ~+3% tg.
- **Compact MTP draft head** (`src/llama-model.cpp`, `src/models/qwen35.cpp`):
  - The draft head reads only the K most frequent vocabulary rows, copied once at load time.
    Draft step 5.1 -> 2.6 ms.
  - Set `LLAMA_MTP_HEAD_ORDER` to a file of int32 token ids, most frequent first, and
    `LLAMA_MTP_HEAD_K` to the row count (16384 works well).
  - The target model still verifies every token, so the output is unchanged; only the acceptance
    rate depends on the list.
  - Build the list from text that resembles your use: code, and your own language.
    Ordering by token id instead of frequency broke non-English text.
- **Scheduler**: host inputs are uploaded with `set_tensor_async` and one synchronization
  (`ggml-backend.cpp`, `GGML_SCHED_ASYNC_INPUT`): graph overhead 0.72 -> 0.25 ms per graph, ~+3% tg.
- **Small f32 matvec** for the 48x5120 DeltaNet alpha/beta projections: 21.5 -> 8.9 us at 4 columns.
  The GEMM library path had a large fixed cost.
- **VRAM**:
  - The legacy pool trims superseded buffers.
  - Q4_0 batches of 9-64 columns (the MTP layer) run as 8-column matvec slices instead of making a
    180 MB fp16 dequantized copy.

### Prefill / short prompts

- **Gated DeltaNet ESIMD kernel for prefill** too (2 state columns per thread, next-token preload):
  1024 tokens 4.3 -> 0.79 ms, 12.7k prompt 629 -> 685 t/s.
- **`LLAMA_MTP_PREFILL_TAIL=N`** (`common/speculative.cpp`): in prompt batches longer than 64 tokens,
  only the last N tokens go through the MTP head. With N=256, a 12k prefill on the MTP server goes
  473 -> 570 t/s. Code tg is unchanged and prose tg drops ~3%.
- **Fused-dequant XMX GEMM for Q3_K** (`ggml_sycl_q3k_gemm_xmx`), used for 12-192 columns:
  - Weights are read as Q3_K and expanded per 32 values to the exact u8 form
    `(q-4)*(scale-32)+124`.
  - Activations are quantized to int8 per 256 values in the DPAS VNNI layout. The +124 offset is
    removed using column sums.
  - B tiles are shared through SLM.
  - In that range it is 1.5-2.9x faster than dequantize + oneDNN (17408x5120 at 64 columns:
    2.01 -> 0.70 ms).
- Q3_K batches of 9-11 columns run as 8-column ESIMD slices.

### Server (`tools/server`, `src`)

- Skip re-saving a context checkpoint identical to an existing one. With a recurrent model, every
  request restored a ~176 MB checkpoint and then saved the very same state again.
- Checkpoint save/restore as async per-tensor copies with one synchronization
  (`LLAMA_STATE_ASYNC`, default on): restore 50 -> 30 ms, save 73 -> 56 ms.

### Measurement tools (env switches)

- `GGML_SYCL_OP_PROFILE=1` (with graphs off): per-op time by type, shape and column count, written
  to `GGML_SYCL_OP_PROFILE_OUT`. It adds ~80 us per op, so subtract that for small ops.
- `GGML_SYCL_GRAPH_PROF`, `LLAMA_DECODE_PROF`, `GGML_SCHED_PROF`, `LLAMA_MTP_PROF`: host-side
  timings of graph replay, decode preparation, the scheduler and the MTP draft loop.
- `test-backend-ops` has extra eval and perf cases with these models' shapes, for every type above.

## Things that did not work

Kept here so nobody repeats them. Each was measured and reverted.

**i-quant matvec (SLM gather bound)**

| Attempt | Result |
|---|---|
| Lookup table replicated per gather lane in SLM (16 KB, bank-conflict free) | IQ4 ~8% slower. Only IQ2_XXS gained, so bank conflicts are not the limit. |
| Half of the lookups from global memory (L1) instead of SLM | Much slower (188 -> 227 us) |
| Register lookup with `iselect` | ~2x slower |
| 32-bit IQ4 table instead of 16-bit | ~3% slower |
| Activations quantized once by a separate kernel instead of per work-group | +10 us per call, slower |
| Exact arithmetic replacement for the 16 IQ4 values | None exists. A piecewise cubic exists but costs more than the table. |
| LSC prefetch, 256 GRF | No gain |

**Other**

- Q3_K: multi-row-per-thread matvec (register spills), pre-quantized activations without SLM.
- An 8-row XMX matvec: slower than dp4a at 1-8 columns.
- The XMX N8 mmvq path for 9-80 columns: slower than dequantize + GEMM on this card, so disable it
  (see below).
- An XMX flash-attention kernel: correct but slower (SLM/barrier stalls). Kept off
  (`GGML_SYCL_FA_XMX`).
- MTP:
  - `LLAMA_MTP_UBATCH=64` makes 12k prefill 27 s -> 204 s.
  - 4-5 drafts and `--spec-draft-p-min`: the changing verify shape defeats graph reuse.
  - Splitting MTP processing per token is slower.

## Applying the patches

Base: upstream llama.cpp `889edf43` (2026-10-03) plus three open SYCL PRs:
[#28725](https://github.com/ggml-org/llama.cpp/pull/28725) (SYCL graph record/replay),
[#29107](https://github.com/ggml-org/llama.cpp/pull/29107) (IQ3 reorder),
[#29864](https://github.com/ggml-org/llama.cpp/pull/29864) (multi-column XMX). Credit for those
goes to their authors.

- `patches/full-vs-upstream-889edf43.patch`: everything (the three PRs + this work), applies cleanly
  to `889edf43`.
- `patches/xe1-only-on-top-of-prs.patch`: only this work, on top of the three PRs.

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 889edf43
git apply path\to\full-vs-upstream-889edf43.patch
```

Build: see `scripts/build.bat` (oneAPI 2026.1, Ninja, `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`,
`GGML_SYCL_GRAPH=ON`). Kernels are JIT compiled on first use, so the first request after a new build
can be slow for a few minutes. Keep `SYCL_CACHE_PERSISTENT=1`. Compiling the many ESIMD/XMX
kernels can briefly take a lot of host RAM, so close other programs when building.

## Launch settings

See `scripts/run-server-16k-mtp.bat` (Q3_K_S) and `scripts/run-server-20k-huihui-mtp.bat` (the
i-quant Huihui GGUF). The important parts:

```
set GGML_SYCL_ENABLE_VMM=0        & rem VMM pool caused fragmentation / spills here
set GGML_SYCL_ENABLE_GRAPH=1
set GGML_SYCL_FA_WG_PER_CU=8
set GGML_SYCL_ENABLE_XMX=0        & rem the XMX mmvq path is slower than dequant+GEMM for 9-80 cols on DG2
set LLAMA_MTP_UBATCH=8            & rem MTP layer batches; larger values blow up VRAM/prefill
set LLAMA_MTP_PREFILL_TAIL=256    & rem only the prompt tail goes through the MTP head
set LLAMA_MTP_HEAD_ORDER=path\to\vocab_order.bin   & rem optional compact draft head
set LLAMA_MTP_HEAD_K=16384
set SYCL_CACHE_PERSISTENT=1

llama-server -m Qwen3.8-27B-Q3_K_S.gguf -c 16384 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 ^
  -b 2048 -ub 768 --parallel 1 --jinja ^
  --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 ^
  --cache-ram 4096 --ctx-checkpoints 8
```

Configs that fit 16 GB (with ~650 MB of desktop/driver VRAM use):

| Model | Context | ubatch | MTP drafts | tg | prefill |
|---|---|---|---|---|---|
| Q3_K_S | 16k | 768 | 3 | 43-50 | ~610 |
| Q3_K_S | 24k | 512 | 2 | 39-43 | 530-555 |
| Huihui UD-Q3_K_XL | 16k (browser chat) | 768 | 3 | 39-41 | ~600 |
| Huihui UD-Q3_K_XL | 20k | 1024 | 3 | 39-41 | 580-690 |

What I measured when trying bigger settings:

- **Q3_K_S at 24k:** ub1024 spills ~930 MB into shared memory and prefill collapses to ~240 t/s.
  3 drafts are +5-9% up to 15k but -3% at 22k, so 2 drafts it is.
- **Huihui at 20k:** ub2048 gives 711-788 t/s prefill, but tg drops ~3% and it spills more.

**Watch "Shared GPU memory" in Task Manager.** On Windows the driver silently spills into system RAM
when VRAM runs out. Up to ~700 MB of shared usage was fine here; at ~900 MB prefill dropped from
hundreds of t/s to 40-250. Browsers (hardware acceleration), Remote Desktop and the desktop
compositor all take VRAM. Some virtual display drivers make the compositor's VRAM grow over hours,
and a reboot resets it.

## OpenCode tip

OpenCode's default system prompt plus tool definitions is ~12.5k tokens. It also lists every
Claude skill it finds in `~/.claude/skills` (~4k tokens). On a 16k context that leaves very little
room, and the session ends up in a compaction loop. Disabling only the skill tool keeps all real
tools and brings the first request down to ~7.3k tokens:

```json
{ "tools": { "skill": false } }
```

See `scripts/opencode.example.json`.

## Kernel knobs

Most new code paths have an environment switch, mainly for A/B testing:

| Variable | Default | Meaning |
|---|---|---|
| `GGML_SYCL_IQ_MV` | 511 | i-quant/K-quant matvec, bit mask: 1 IQ4_XS, 2 IQ3_S, 4 IQ3_XXS, 8 IQ2_XXS, 16 IQ2_XS, 32 IQ2_S, 64 IQ4_NL, 128 Q4_K, 256 Q5_K, 512 Q6_K |
| `GGML_SYCL_IQ_F16` | 511 | same mask, for the dequantize-to-fp16 (prefill) kernels |
| `GGML_SYCL_Q8_SMALL` | 1 | small Q8_0 matrices (<=256 rows) on the small matvec kernel |
| `GGML_SYCL_Q3K_DP4A` | 1 | dp4a Q3_K matvec (except 2 columns); 0 = fp32 kernel only |
| `GGML_SYCL_Q3K_SPLIT` | 48 | Q3_K batches up to N columns as 8-column ESIMD slices |
| `GGML_SYCL_Q3K_GEMM` / `_MAX` | 12 / 192 | column range of the fused XMX Q3_K GEMM |
| `GGML_SYCL_GDN_ESIMD` | 1 | ESIMD Gated DeltaNet kernel (decode and prefill) |
| `GGML_SYCL_GDN_GATHER` | 1 | skip the separate DeltaNet state copy |
| `GGML_SYCL_GDN2` | 1 | Gated DeltaNet prefill variant (non-ESIMD path) |
| `GGML_SYCL_F32_SMALL` | 1 | small f32 matvec (<=1024 rows, <=8 columns) |
| `GGML_SYCL_FA_GQA_ESIMD` | 1 | ESIMD flash-decode for q8_0 D=256 |
| `GGML_SYCL_FA_ESIMD_SLM` | 1 | flash-decode work-group variant with K/V tiles in SLM (2+ tokens) |
| `GGML_SYCL_FA_MERGE3` | 1 | merged partial-result reduction for flash-decode |
| `GGML_SYCL_FA_CHUNK1` | 80 | KV chunk size for single-token flash-decode |
| `GGML_SYCL_POOL_TRIM` | 1 | free superseded pool buffers |
| `GGML_SCHED_ASYNC_INPUT` | 1 | async host input upload with one synchronization |
| `LLAMA_STATE_ASYNC` | 1 | async state copies for checkpoints |
| `LLAMA_MTP_PREFILL_TAIL` | off | only the last N prompt tokens of a long batch go through the MTP head |
| `LLAMA_MTP_HEAD_ORDER` / `_K` | off | compact MTP draft head (token frequency list, row count) |

## Known issues

- 2 of 4066 `FLASH_ATTN_EXT` backend tests fail (batch 64/67, D=256). They also fail without these
  patches, and one depends on `GGML_SYCL_ENABLE_MKL_FA`. All 1426 `MUL_MAT` and 216 `GET_ROWS`
  tests pass.
- Greedy MTP output can differ slightly from run to run (this was already the case before these
  patches). Compare log-probabilities, not text, when validating.
- Tested only on this one card and these models. Other DG2 cards should behave similarly. Battlemage
  (Xe2) has different XMX tile sizes and was not tested.

## Tooling notes

- VTune 2025.4 `gpu-hotspots` works on Windows without admin rights and was the only way to see
  pipeline utilization. IGC shader dumps (`IGC_ShaderDumpEnable`) did not produce output on Windows,
  neither at runtime nor in AOT builds.
- Raw ESIMD `dpas` on the A770 reaches ~200 int8 TOPS with 8 independent accumulators and ~37 with
  a single dependency chain.
- SLM gathers on the A770 run at roughly 10 lanes per clock per Xe core in these kernels. A table
  lookup per 2 weights costs about as much as the rest of the matvec.

## License

The patches modify llama.cpp and are under the same MIT license as llama.cpp.
