@echo off
rem Qwen3.8-27B Q3_K_S, 16k context, MTP speculative decoding (3 drafts), A770 16 GB.
rem Edit the two paths below.
set "BIN=C:\path\to\llama.cpp\build\bin"
set "MODEL=C:\path\to\Qwen3.8-27B-Q3_K_S.gguf"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul
set GGML_SYCL_ENABLE_VMM=0
set GGML_SYCL_ENABLE_GRAPH=1
set GGML_SYCL_FA_WG_PER_CU=8
set GGML_SYCL_ENABLE_XMX=0
set LLAMA_MTP_UBATCH=8
set LLAMA_MTP_PREFILL_TAIL=256
rem Optional compact MTP draft head (int32 token ids, most frequent first):
rem set LLAMA_MTP_HEAD_ORDER=C:\path\to\vocab_order.bin
rem set LLAMA_MTP_HEAD_K=16384
set SYCL_CACHE_PERSISTENT=1

"%BIN%\llama-server.exe" -m "%MODEL%" --alias qwen3.8-27b-16k -c 16384 ^
  --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 768 --parallel 1 ^
  -ngl 99 -fa on --jinja --reasoning-effort medium --reasoning-budget 2048 ^
  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8001 ^
  --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 ^
  -lm none --cache-ram 4096 --ctx-checkpoints 8
