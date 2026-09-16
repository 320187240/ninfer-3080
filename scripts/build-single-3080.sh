#!/usr/bin/env bash
# Build NInfer for a single RTX 3080 (sm_89, Ada).
#
# The repo's top-level CMakeLists.txt only accepts CMAKE_CUDA_ARCHITECTURES of
# 86 and/or 89. For an RTX 3080 pass 89. The sm_120a nvfp4 TMA kernels are
# conditionally compiled out when the arch is not 120a, so no source changes
# are needed to target the 3080.
#
# Prereqs: CUDA >= 12.8, CMake >= 3.28, Ninja, a C++20 compiler, FFmpeg dev
# libs (libavformat/avcodec/avutil/swscale), libcurl >= 7.85.
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${1:-build}"

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"

echo "Build complete: ${BUILD_DIR}/apps/ninfer-serve"