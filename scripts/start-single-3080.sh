#!/usr/bin/env bash
# Single RTX 3080 startup script for jimchan3301/ninfer-3080duo (non-TP mode)
set -euo pipefail

cd "$(dirname "$0")/.."

# Stop llama.cpp
echo "Stopping llama.cpp..."
systemctl --user stop hermes-local-model.service || true

# Wait for GPU and port to be free
for i in {1..30}; do
  free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if (( free_vram >= 19000 )) && ! ss -tln | grep -q ':8080 '; then
    echo "GPU free: ${free_vram} MiB, port 8080 released"
    break
  fi
  sleep 2
done

# Start NInfer (single GPU, no --tp)
echo "Starting Qwen3.8-27B on single RTX 3080..."
exec build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --api-key sk-litellm-REDACTED \
  --max-context 76800 --kv-capacity 76800 \
  --kv-dtype rk8v4 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --no-thinking
