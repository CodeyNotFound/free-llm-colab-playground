#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PWD/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BUILD_JOBS="${BUILD_JOBS:-2}"
RUNTIME_DIR="${PLAYGROUND_RUNTIME_DIR:-$PWD/.runtime}"
PATH_FILE="$RUNTIME_DIR/llama_server_path.txt"
PREBUILT_BASE_URL="${PLAYGROUND_PREBUILT_BASE_URL:-https://github.com/CodeyNotFound/free-llm-colab-playground/releases/download/runtime-t4-v1}"

stage() {
  echo "::playground-stage::$1"
}

mkdir -p "$RUNTIME_DIR"

CUDA_FLAG="OFF"
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  CUDA_FLAG="ON"
fi

PREBUILT_SERVER="$RUNTIME_DIR/llama-server"
if [[ -x "$PREBUILT_SERVER" ]] && "$PREBUILT_SERVER" --version >/dev/null 2>&1; then
  stage "Using cached llama.cpp runtime"
  printf '%s\n' "$PREBUILT_SERVER" > "$PATH_FILE"
  echo "llama-server ready: $PREBUILT_SERVER"
  exit 0
fi

if [[ "$CUDA_FLAG" == "ON" && "$(uname -m)" == "x86_64" && "${PLAYGROUND_SKIP_PREBUILT:-0}" != "1" ]]; then
  stage "Downloading the precompiled T4 runtime (usually 1–3 minutes)"
  TEMP_DIR="$(mktemp -d)"
  ARCHIVE="$TEMP_DIR/llama-server-colab-t4-cuda12.tar.gz"
  CHECKSUM="$ARCHIVE.sha256"
  if curl --fail --location --silent --show-error \
      "$PREBUILT_BASE_URL/llama-server-colab-t4-cuda12.tar.gz" --output "$ARCHIVE" && \
      curl --fail --location --silent --show-error \
      "$PREBUILT_BASE_URL/llama-server-colab-t4-cuda12.tar.gz.sha256" --output "$CHECKSUM" && \
      (cd "$TEMP_DIR" && sha256sum --check "$(basename "$CHECKSUM")") && \
      tar -xzf "$ARCHIVE" -C "$RUNTIME_DIR" && \
      chmod +x "$PREBUILT_SERVER" && \
      "$PREBUILT_SERVER" --version >/dev/null 2>&1; then
    printf '%s\n' "$PREBUILT_SERVER" > "$PATH_FILE"
    rm -rf "$TEMP_DIR"
    stage "Precompiled T4 runtime verified"
    echo "llama-server ready: $PREBUILT_SERVER"
    exit 0
  fi
  rm -rf "$TEMP_DIR"
  echo "Precompiled runtime unavailable or incompatible; falling back to a source build." >&2
fi

stage "Downloading llama.cpp source"
if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
else
  git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
  git -C "$LLAMA_CPP_DIR" checkout --detach FETCH_HEAD
fi

stage "Configuring the CUDA build"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CUDA="$CUDA_FLAG" \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
stage "Compiling llama-server (first run can take 15–30 minutes)"
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$BUILD_JOBS" --target llama-server

SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
if [[ ! -x "$SERVER_PATH" ]]; then
  echo "llama-server build completed but binary was not found at $SERVER_PATH" >&2
  exit 1
fi

printf '%s\n' "$SERVER_PATH" > "$PATH_FILE"
echo "llama-server ready: $SERVER_PATH"
echo "export LLAMA_SERVER_PATH='$SERVER_PATH'"
