# Qwen3.8-27B on an Intel Arc A770 (16 GB): ~2x decode with custom SYCL/ESIMD kernels + MTP

Notes, patches and launch settings from a few weeks of tuning llama.cpp's SYCL backend for one
specific setup: **Intel Arc A770 16 GB, Windows 11, Qwen3.8-27B (hybrid Gated DeltaNet + attention)
at Q3_K_S**, used as a local coding agent in OpenCode.

The short version: decode speed roughly doubled (MTP speculative decoding became actually usable
on this card), and small agent turns got noticeably snappier. Prefill improved less.

This is a personal, hardware-specific fork, not a polished upstream PR. Several kernels are tuned
for Xe-HPG (DG2) and this model's shapes. Use at your own risk.

## Results

All numbers on an A770 16 GB, Windows 11, oneAPI 2026.1, Q3_K_S weights, q8_0 KV cache.

| Workload | Before | After |
|---|---|---|
| Code generation, 8k ctx, greedy, no speculation | 19.7 t/s | ~20 t/s (bandwidth bound) |
| Code generation, 8k ctx, greedy, MTP 3 drafts | MTP not worth using | **39-41 t/s** |
| OpenCode agent session (12-16k ctx, temp 0.6), decode | ~17 t/s | **~30-42 t/s** |
| Prefill, 12k prompt, no MTP, ubatch 2048 (24k ctx) | ~625 t/s @ub1024 | **~710 t/s** |
| Prefill, 12k prompt, MTP server, ubatch 1024 (16k ctx) | ~413 t/s @ub512 | **~450-480 t/s** |
| 40-token appended prompt (agent tool output) at 12.7k ctx | ~860 ms | **~650 ms** |

"Before" = the same llama.cpp base with the open SYCL PRs listed below, stock kernels.

For scale: the 27B Q3 model is ~11.5 GB, so on this card every decode step is bandwidth bound
(~31 ms of pure weight reads at the A770's ~374 GB/s effective read rate). Most of the decode gain
comes from making the speculative-decoding verify step (4 tokens at once) cost close to a single
token.

## What changed

All in `ggml/src/ggml-sycl` unless noted. ESIMD = Intel explicit SIMD (`[[intel::sycl_explicit_simd]]`).

**Decode / MTP verify**
- **GQA flash-decode kernel** (`fattn-gqa-dec.cpp`) for 1-8 query tokens, q8_0/f16 KV, head size
  64/128/256, plus an ESIMD variant for q8_0 D=256. The old vector FA kernel re-read the KV cache
  once per query head (24 q heads / 4 kv heads).
- **Multi-column dequantize-once matvec** for reordered K-quants (`dmmv.cpp`): one weight read
  serves up to 8 activation columns.
- **Q3_K matvec kernels**: a wide-load fp32 kernel for 1-2 columns (~340 GB/s, 91% of measured read
  peak) and a dp4a kernel (int8 activations in SLM, dword-mask scale unpack) for 3-8 columns.
- **Small f32 matvec** for the 48x5120 DeltaNet alpha/beta projections (21.5 -> 8.9 us at 4 columns;
  the GEMM library path had a large fixed cost).
- **MTP draft head over a row view of the reordered LM head** (vocabulary trimming via
  `LLAMA_MTP_VOCAB`; works, but off by default: it lowers acceptance in non-code text).
- **VRAM**: the legacy pool trims superseded buffers; Q4_0 batches of 9-64 columns (the MTP layer)
  run as 8-column matvec slices instead of a 180 MB fp16 dequantized copy.

**Prefill / short prompts**
- **Gated DeltaNet kernel** variant with 2 state columns per sub-group and next-token preload
  (625 -> 642 t/s at ubatch 1024).
- **Fused-dequant XMX GEMM for Q3_K** (`ggml_sycl_q3k_gemm_xmx`): weights are read as Q3_K and
  expanded per 32 to u8 `(q-4)*(scale-32)+124` (exact), activations are quantized to int8 per 256
  in the DPAS VNNI layout, the +124 offset is removed with column sums, B tiles are shared through
  SLM. Used for 12-192 columns, where it is 1.5-2.9x faster than dequantize + oneDNN
  (e.g. 17408x5120 at 64 columns: 2.01 -> 0.70 ms). For large batches it is still slower than
  oneDNN (VTune: ALU bound on the weight expansion, XMX ~17% busy), so prefill-sized batches keep
  the library path.
- Q3_K batches of 9-11 columns run as 8-column ESIMD slices.

**Server (`tools/server`, `src`)**
- Skip re-saving a context checkpoint identical to an existing one. With a recurrent model
  every request restored a ~176 MB checkpoint and then saved the very same state again.
- Checkpoint save/restore as async per-tensor copies with one synchronization
  (`LLAMA_STATE_ASYNC`, default on): restore 50 -> 30 ms, save 73 -> 56 ms.

**Things that did not work** (kept here so nobody repeats them): multi-row-per-thread Q3_K
matvec (register spills), LSC prefetch, pre-quantized activations without SLM, an 8-row XMX matvec
(slower than dp4a at 1-8 columns), the XMX N8 mmvq path for 9-80 columns (slower than
dequantize + GEMM on this card: disable it, see below), `LLAMA_MTP_UBATCH=64` (12k prefill
27 s -> 204 s).

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
`GGML_SYCL_GRAPH=ON`). Kernels are JIT compiled on first use; keep `SYCL_CACHE_PERSISTENT=1`.

## Launch settings

See `scripts/run-server-16k-mtp.bat`. The important parts:

```
set GGML_SYCL_ENABLE_VMM=0        & rem VMM pool caused fragmentation / spills here
set GGML_SYCL_ENABLE_GRAPH=1
set GGML_SYCL_FA_WG_PER_CU=8
set GGML_SYCL_ENABLE_XMX=0        & rem the XMX mmvq path is slower than dequant+GEMM for 9-80 cols on DG2
set LLAMA_MTP_UBATCH=8            & rem MTP layer batches; larger values blow up VRAM/prefill
set SYCL_CACHE_PERSISTENT=1

llama-server -m Qwen3.8-27B-Q3_K_S.gguf -c 16384 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 ^
  -b 2048 -ub 1024 --parallel 1 --jinja ^
  --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 ^
  --cache-ram 4096 --ctx-checkpoints 8
```

Configs that fit 16 GB without spilling into shared memory on a desktop with a browser open:

| Context | ubatch | MTP drafts | Notes |
|---|---|---|---|
| 16k | 1024 | 3 | fastest; best for agent work |
| 20k | 1024 | 2 | |
| 24k | 512 | 2 | borderline: spills if the desktop uses more VRAM |
| 24k, no MTP | 2048 | - | best prefill (~710 t/s) |

**Watch "Shared GPU memory" in Task Manager.** On Windows the driver silently spills into system RAM
when VRAM runs out. Above ~400 MB of shared usage, prefill drops from hundreds of t/s to 40-100.
Browsers, Remote Desktop and the desktop compositor all take VRAM.

## OpenCode tip

OpenCode's default system prompt plus tool definitions is ~12.5k tokens, and it also lists every
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
| `GGML_SYCL_Q3K_DP4A` | 1 | dp4a Q3_K matvec (except 2 columns); 0 = fp32 kernel only |
| `GGML_SYCL_Q3K_SPLIT` | 48 | Q3_K batches up to N columns as 8-column ESIMD slices |
| `GGML_SYCL_Q3K_GEMM` / `_MAX` | 12 / 192 | column range of the fused XMX Q3_K GEMM |
| `GGML_SYCL_GDN2` | 1 | Gated DeltaNet prefill variant |
| `GGML_SYCL_F32_SMALL` | 1 | small f32 matvec (<=1024 rows, <=8 columns) |
| `GGML_SYCL_FA_GQA_ESIMD` | 1 | ESIMD flash-decode for q8_0 D=256 |
| `GGML_SYCL_POOL_TRIM` | 1 | free superseded pool buffers |
| `LLAMA_STATE_ASYNC` | 1 | async state copies for checkpoints |
| `LLAMA_MTP_VOCAB` | 0 | MTP draft head over the first N vocabulary rows |

## Known issues

- 2 of 4066 `FLASH_ATTN_EXT` backend tests fail (batch 64/67, D=256). They also fail without these
  patches, and one depends on `GGML_SYCL_ENABLE_MKL_FA`.
- The first use of a new ESIMD kernel can take a long time to JIT compile; keep the persistent cache.
- Tested only on this one card and model. Other DG2 cards should behave similarly; Battlemage
  (Xe2) has different XMX tile sizes and was not tested.

## Tooling notes

- VTune 2025.4 `gpu-hotspots` works on Windows without admin rights and was the only way to see
  pipeline utilization. IGC shader dumps (`IGC_ShaderDumpEnable`) did not produce output on Windows,
  neither at runtime nor in AOT builds.
- Raw ESIMD `dpas` on the A770 reaches ~200 int8 TOPS with 8 independent accumulators and ~37 with
  a single dependency chain.

## License

The patches modify llama.cpp and are under the same MIT license as llama.cpp.
