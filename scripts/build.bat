@echo off
rem Build llama.cpp with the SYCL backend (oneAPI 2026.1, Visual Studio 2022 build tools, Ninja).
rem Usage: build.bat <path-to-llama.cpp>
set "SRC=%~1"
if "%SRC%"=="" set "SRC=%CD%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul
cmake -S "%SRC%" -B "%SRC%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=/EHsc ^
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx ^
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON ^
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON
cmake --build "%SRC%\build" --config Release -j 12 --target llama-server llama-bench test-backend-ops
