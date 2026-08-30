#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates cmake git ninja-build

rm -rf /tmp/llama.cpp
git init /tmp/llama.cpp
git -C /tmp/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
git -C /tmp/llama.cpp fetch --depth 1 origin "${LLAMA_REF:-master}"
git -C /tmp/llama.cpp checkout --detach FETCH_HEAD
cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama.cpp/build --target llama-server --parallel 2

mkdir -p /workspace/dist /tmp/playground-runtime
cp /tmp/llama.cpp/build/bin/llama-server /tmp/playground-runtime/llama-server
tar -C /tmp/playground-runtime -czf /workspace/dist/llama-server-colab-t4-cuda12.tar.gz llama-server
cd /workspace/dist
sha256sum llama-server-colab-t4-cuda12.tar.gz > llama-server-colab-t4-cuda12.tar.gz.sha256
