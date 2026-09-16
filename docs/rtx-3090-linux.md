# Build NInfer on Linux (RTX 3080, sm_86)

This guide builds the `sm_86` runtime for one NVIDIA GeForce RTX 3080 20 GB (Ampere).
The project does not publish a qualified Linux binary release; the production host builds
natively and serves through a user systemd unit.

Do not change `CMAKE_CUDA_ARCHITECTURES` to `89`.
The RTX 4090 fork uses Ada-specific schedules that do not apply to sm_86 Ampere cards.

## Native Ubuntu 24.04 build

Install the CUDA Toolkit 12.8 or newer from NVIDIA.
Then install the host compiler and media dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

Select GCC 13 for host and CUDA compilation:

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13
```

Configure and build the Linux applications:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build --parallel 2
```

The build creates these applications:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

## Optional vcpkg dependencies

Linux can use the pinned `vcpkg.json` manifest instead of system FFmpeg and curl packages.
Bootstrap the pinned vcpkg revision:

```bash
git clone https://github.com/microsoft/vcpkg.git "$HOME/.local/share/vcpkg"
git -C "$HOME/.local/share/vcpkg" checkout 4bca8fd8654e5ba76f92661db7bfe954768ad8ef
"$HOME/.local/share/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
```

Add these options to the native CMake command:

```text
-DCMAKE_TOOLCHAIN_FILE=$HOME/.local/share/vcpkg/scripts/buildsystems/vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-linux
```

## Bash scripts

`scripts/` contains the local build and model-download helpers:

```bash
./scripts/build-single-3080.sh   # cmake + ninja, sm_86 (RTX 3080 Ampere)
./scripts/download-qwen38.sh     # fetch the Qwen3.8-27B .ninfer artifact
```

Set `NINFER_MODEL_DIR` to use another model directory (default: `scripts/models`). The production
host keeps the model at `models/qwen3_8_27b.ninfer` and serves it through the user unit
`~/.config/systemd/user/ninfer.service`; build output is `build/apps/ninfer-serve`.

## Deploying the user systemd unit

A minimal production unit (single GPU, 76,800-token `rk8v4` context, MTP3):

```ini
# ~/.config/systemd/user/ninfer.service
[Unit]
Description=NInfer Qwen3.8-27B Inference Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/ninfer-3080
ExecStart=/path/to/ninfer-3080/build/apps/ninfer-serve /path/to/ninfer-3080/models/qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 --api-key <your key> --max-context 76800 --kv-capacity 76800 --kv-dtype rk8v4 --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft --no-thinking
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Enable it for boot (requires `loginctl enable-linger $USER` so user units start without a
login session):

```bash
systemctl --user daemon-reload
systemctl --user enable --now ninfer.service
journalctl --user -u ninfer.service -n 50 --no-pager
```

Check the server answers:

```bash
curl -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer <your key>"
```

## Validation

Make sure that the applications start:

```bash
./build/apps/ninfer --help
./build/apps/ninfer-serve --help
```

Run one short generation with the real Qwen3.8 artifact:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in two sentences." \
  --max-context 8192 --max-new 32 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

A successful compile does not qualify Linux performance.
Record the GPU, driver, CUDA Toolkit, compiler, workload, and result before you publish Linux measurements.
