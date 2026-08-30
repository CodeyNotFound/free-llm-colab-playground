#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PWD/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BUILD_JOBS="${BUILD_JOBS:-2}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
else
  git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
  git -C "$LLAMA_CPP_DIR" checkout --detach FETCH_HEAD
fi

CUDA_FLAG="OFF"
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  CUDA_FLAG="ON"
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CUDA="$CUDA_FLAG" \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$BUILD_JOBS" --target llama-server

SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
if [[ ! -x "$SERVER_PATH" ]]; then
  echo "llama-server build completed but binary was not found at $SERVER_PATH" >&2
  exit 1
fi

echo "llama-server ready: $SERVER_PATH"
echo "export LLAMA_SERVER_PATH='$SERVER_PATH'"

