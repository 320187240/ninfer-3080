# NInfer-3080duo

NInfer-3080duo is a fork of the NInfer-3090 C++20/CUDA inference engine that runs **Qwen3.8-27B
tensor-parallel across two consumer GPUs in one Windows process** (developed on
3080 20 GB + 4080 16 GB, now running on 2× RTX 3080 20 GB). The 16.96 GiB groupwise
`.ninfer` artifact is split along group-safe axes to ~8.4 GiB of weights per rank,
per-rank CUDA Graphs execute the decode rounds, and a mapped-pinned spin transport
(`TpLink`) carries the ~150 allreduce/argmax exchanges per round. Neither card can hold
the model alone; together they serve the full OpenAI/Anthropic surface with a
**262,144-token INT8 context**.

Inherited single-GPU capabilities (paged KV, prefix reuse, CUDA Graphs, MTP speculative
decoding, reasoning-effort control, ReplaySSM, bounded concurrency, vision) are unchanged
and documented below. See [TP2 architecture](docs/maintainer/tp-architecture.md) for the
split plan, transport measurements, and gate evidence.

Community project, maintained on a best-effort basis. Issues and PRs are very welcome, but support
and feature requests are not guaranteed.

## TP2 quick start

Both GPUs must be visible to CUDA (the engine addresses CUDA dev0/dev1 in order; on the
current host both are RTX 3080 20 GB).
`--tp` enables the two-rank path; it is greedy-only (requests with temperature > 0 are
rejected with HTTP 503 and a clear message). `--tp --vision` enables image/video input: the
vision tower runs on rank 0 and its embeddings are broadcast to rank 1 through the TpLink
(see [TP2 architecture](docs/maintainer/tp-architecture.md)), so vision works at moderate
contexts — the 262,144-token profile does not fit with vision loaded (startup reports the
exact shortfall; the 8192-token vision profile starts both ranks with ~5.3 GiB free).

CLI, one greedy generation inside the full 262K context:

```bat
ninfer.exe qwen3_8_27b.ninfer --tp --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --greedy --max-new 128 --no-thinking --prompt "What is the boiling point of water at sea level? Answer in one sentence."
```

CLI, image understanding across both GPUs (`--messages` carries image parts exactly like the
single-GPU vision profile):

```bat
ninfer.exe qwen3_8_27b.ninfer --tp --vision --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --greedy --max-new 64 --no-thinking --messages messages.json
```

Server with the same 262K profile (send `"temperature": 0` from clients, or add `--greedy`).
TP serves the same bounded multi-request concurrency as single-GPU (`--max-concurrency 1-8`).
Compatible-prefix reuse is on by default in TP mode; `--no-tp-prefix-reuse` opts out.

```bat
ninfer-serve.exe qwen3_8_27b.ninfer --tp --host 127.0.0.1 --port 8117 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 8 --spec mtp --draft-tokens 3 --lm-head-draft
```

```bash
curl http://127.0.0.1:8117/v1/chat/completions -H "Content-Type: application/json" -d \
  '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is the boiling point of water at sea level?"}],"max_tokens":64,"temperature":0}'
```

## TP2 measured results (Qwen3.8-27B, 2× RTX 3080 20 GB)

| Gate | Result |
|---|---|
| 262,144-token INT8 startup validation | passes on both ranks, no trimming (planned slack 1.64 GiB, graphs 12 of 86 MiB) |
| CLI MTP3 greedy decode, RDP host (code-content 1024-token generation) | 75.1 tok/s @ 86.6% MTP acceptance; ~53 tok/s on essay-content prompts @ ~61% — decode is GEMM-bound at MTP3's T≤4 shapes, so tok/s tracks content acceptance |
| Served agentic session (Bun harness hard task, chunk 128, prefix reuse) | decode 78.4-80.3 tok/s aggregate, MTP 2.86/3 (95%), mean TTFT ~1.0-1.5 s |
| Served agentic session prefix-cache hit rate | 91-95% of prompt tokens (was 0% before TP prefix reuse) |
| Multi-request concurrency (`--max-concurrency 8`, MTP3, decode-saturation sweep, 2048-token waves) | aggregate steady decode 57.1 (C1) → 88.3 (C2) → 135.3 (C4) → 217.7 tok/s (C8) = 3.81×; full batch in every steady interval; `ninfer_qwen3_6_27b_tp_concurrent_test` requires token-identical greedy output across C1, C8 concurrent, and warm sequential phases |
| Correctness spot gate ("The capital of France is") | `760 6511 314 9338 369 2972 57590 159034 248046` — token-identical with and without MTP3 |
| Vision gate (512x512 red-square image, 8192-token profile) | answers "I see a red square." — identical to the single-GPU answer on device 1 |

Previous host (3080 20 GB + 4080 16 GB) reference points: quiet-window single-generation
decode 82.7-85.3 tok/s on code-like content; with a physical display attached under ambient
desktop load, 39-42 tok/s (the clock-parking caveat below). The current 2×3080 host runs
headless over RDP, where display-driven clock parking does not occur and the decode-round
time is set by small-batch GEMM efficiency (see the decode anatomy record in the harness
FINDINGS).

### Ambient-load caveat and clock locking

This caveat applies when a physical display is attached to rank 0 (the current host is
headless/RDP and does not exhibit it). Under Windows WDDM desktop load the display GPU's SM
clock parks (measured 780 MHz with ~58% background utilization while the peer held
2820 MHz), and because every TP step synchronizes at each allreduce, decode in those windows
drops to roughly half of the quiet-window rate. The original measurement host had no
administrator access, so clocks could not be locked. In an elevated shell, locking both
cards below max boost removes most of this variance — suggested starting points, not
requirements:

```bat
nvidia-smi -i <rank 0 index> -lgc 1750,1750
nvidia-smi -i <rank 1 index> -lgc 1750,1750
```

---

The sections below describe the inherited single-GPU NInfer-3090 behavior, measurements, and
packaging; on this fork they apply to the non-`--tp` path on a single installed GPU.



On an RTX 3090, Qwen3.8-27B supports a measured **171K-token INT8 context** with the standard
1 GiB safety headroom. The optional RotorQuant `rk8v4` cache raises this to **226K tokens with
1 GiB headroom**, or **247,872 tokens (about 248K)** in a tightly packed 300 MiB-headroom profile.
RotorQuant is an opt-in feature; INT8 remains the default because `rk8v4` is lossy.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable. 

The goal is the make the utmost rippin Qwen inference stack for the 3000 series. Gladly taking PR's, all help much appreciated. 

Release notes for this branch: [v0.6.1](RELEASE_NOTES_0.6.1.md).

## Choose a platform

### Linux (RTX 3080, sm_86)

The single-GPU path on this host is the production setup: build with `scripts/build-single-3080.sh`
and run through the user service `~/.config/systemd/user/ninfer.service` (76,800-token context,
rk8v4 KV, MTP3, single request). The Linux build guide covers the native Ubuntu build and model
download (`scripts/download-qwen38.sh`, interrupted downloads resume).

## Qwen3.8-27B support and RTX 3090 results

Qwen3.8-27B is validated from one through eight simultaneous users. ReplaySSM cuts the memory cost
of speculative decoding, allowing the faster MTP3 mode to remain enabled at C8. The table below is
the new sustained test: every request generated 1,024 tokens with CUDA Graphs enabled.

The prompts were **29-34 input tokens** and the server's maximum context window was **8,192 tokens
per request**. Each measured sequence therefore reached roughly 1,053-1,058 tokens including its
generated output. This is a long-output/decode benchmark, not an 8K-prompt or long-prefill test.
C1-C4 used an 8,192-token shared KV pool; C8 used 16,384 tokens so all eight requested outputs
could be admitted simultaneously.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **70.19 tok/s** | **71.00 tok/s** | 61.13% | 149 ms | 19,641 MiB |
| C2 | 2,048 tokens | **89.43 tok/s** | **90.66 tok/s** | 59.66% | 262 ms | 20,022 MiB |
| C4 | 4,096 tokens | **97.89 tok/s** | **100.28 tok/s** | 59.63% | 538 ms | 20,641 MiB |
| C8 | 8,192 tokens | **161.28 tok/s** | **165.33 tok/s** | 56.84% | 1,215 ms | 22,138 MiB |

C1 is the responsive choice for a single user. C8 delivers **2.3x the total throughput** when
several requests are active. The C8 long-output test uses a 16K shared KV pool so all eight
1,024-token responses can be admitted together.

### Prompt-processing speed

Prompt processing was tested separately with **4,362 fresh input tokens per request**, an 8,192-token
per-request context window, 512-token prefill chunks, INT8 KV, ReplaySSM/MTP3, CUDA Graphs, and
prefix reuse disabled. Each request generated only 16 tokens so the run measures prefill rather
than long decode.

| Cohort | Total fresh input | Aggregate prefill | Active-prefill speed | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| C1 | 4,362 tokens | **861.51 tok/s** | 893.98 tok/s | 4,893 ms | 19,114 MiB |
| C2 | 8,724 tokens | **853.86 tok/s** | 883.95 tok/s | 7,478 ms | 19,697 MiB |
| C4 | 17,448 tokens | **847.26 tok/s** | 874.49 tok/s | 12,692 ms | 20,894 MiB |
| C8 | 34,896 tokens | **844.10 tok/s** | 870.94 tok/s | 23,028 ms | 23,207 MiB |

`Aggregate prefill` is total fresh input tokens divided by the complete request-wave time, so it is
the user-facing throughput number. NInfer currently processes one long prefill at a time; cohort
batching accelerates decode, but does not multiply prompt ingestion. Consequently C1-C8 remain near
844-862 input tok/s while queued requests increase mean TTFT. `Active-prefill speed` excludes queue
waiting and measures only the server's recorded prefill phase.

### Optional RotorQuant KV for longer context

`rk8v4` is an **experimental, opt-in** KV-cache mode for Qwen3.8-27B. INT8 remains the default and
the recommended quality setting. RotorQuant applies the same normalized transform to queries and
keys, rotates values before four-bit storage, and reverses the value transform after attention.
This reduces the V-cache footprint while keeping keys at eight bits.

Add `--kv-dtype rk8v4` to either `ninfer.exe` or `ninfer-serve.exe`. For example, from Command
Prompt:

```bat
ninfer-serve.exe qwen3_8_27b.ninfer --max-context 131072 --kv-capacity auto --max-concurrency 1 --prefill-chunk 1024 --kv-dtype rk8v4 --spec mtp --draft-tokens 3 --lm-head-draft
```

On the development RTX 3090, C1 with MTP and CUDA Graphs disabled fit a **226,560-token** logical
context with the normal 1 GiB automatic-sizing headroom; 226,624 was rejected. The comparable INT8
boundary was 171,648 tokens, so `rk8v4` increased measured allocatable context by about **32%**.
This is an allocation plus short-execution boundary, not a full 226K prefill quality claim.

A second explicit-reservation test reduced operational headroom to approximately 300 MiB.
`rk8v4` successfully started and generated at **247,872 tokens**, leaving **302.97 MiB** physically
free after startup. The next 64-token page boundary, 247,936, left only 274.58 MiB because it crossed
a CUDA allocation granularity. Therefore 247,872 is the measured maximum that retained at least
300 MiB—not 2x INT8 capacity, but about **44% more** than the 171,648-token INT8/1 GiB baseline.
This tight profile leaves little tolerance for other GPU users and is not the recommended default.

The matched 1,024-token hard coding test reduced 4K KV payload from 140.38 MiB to 106.35 MiB and
decoded at 84.07 tok/s versus 85.72 tok/s for INT8. Quality was not equivalent: the RotorQuant
answer introduced a faulty nested-rollback design that the INT8 answer avoided, and both answers
hit their output limit. Use `rk8v4` only when its context gain is worth task-specific quality
validation; do not use it as the default for correctness-sensitive work.

### Qwen3.8 vision

The same Qwen3.8 artifact supports images. Start the server with `--vision`, MTP3, INT8 KV, and a
32K maximum context (or the production `ninfer.service` profile with `rk8v4` KV at 76,800 tokens).

A 1,920×1,080 image expanded to 2,074 prompt tokens and was read correctly. Measured TTFT was
3.29 seconds, decode reached 98.1 tok/s, MTP acceptance was 96.7%, and startup retained 2.16 GiB
free VRAM. The artifact also declares multi-image and video support; this release test directly
validated a single image.

## Qwen3.6-35B-A3B RTX 3090 results

Measured with the compact 20.84 GiB 35B-A3B artifact, a 4K shared INT8 group-64 paged KV pool,
CUDA Graphs, MTP3, greedy decoding, and no competing GPU workload:

| Concurrent requests | 128 output tokens each | Observed VRAM |
|---:|---:|---:|
| 1 | 162.7 aggregate tok/s | 22,427 MiB |
| 2 | 267.9 aggregate tok/s | 22,743 MiB |
| 4 | 366.2 aggregate tok/s | 23,377 MiB |
| 6 | 383.4 aggregate tok/s | 24,038 MiB |
| 8 | rejected at startup | about 503 MiB over the safe reservation limit |

A longer 512-token-per-request check reached **286.8 tok/s at C1** and **399.1 aggregate tok/s at
C2**. These short-prompt measurements include request-level timing and are not directly comparable
to v0.3.1's 1,500-token adaptive prompt-lookup benchmark.

Compatible-prefix reuse was validated end to end: a repeated 26-token prompt reused 24 tokens,
reducing measured prefill from 371 ms to 10 ms.

### Qwen3.6-35B vision

The compact 35B artifact includes its vision encoder and accepts images through the same OpenAI-
compatible API. Start the server with `--vision` and leave speculative decoding disabled
(`--no-spec` or omit `--spec`); the profile is one request, 32K maximum context, INT8 KV.

The safe RTX 3090 profile is **one request, 32K maximum context, INT8 KV, vision enabled, and MTP
disabled**. A current v0.6 test processed three 1,920×1,080 images correctly. Each image expanded
to a 2,081-token prompt; engine TTFT was 3.76–4.07 seconds, decode was about 159 tok/s, and peak
VRAM was 23,944 MiB. This leaves little room for another GPU workload.

MTP is intentionally off for this profile. At 32K, speculative recurrent state would exceed the
3090 memory budget; KV compression alone does not recover enough memory. Text-only 35B profiles can
still use MTP3 as documented above.

## Capabilities

- Native SM86 CLI and server applications for Linux.
- OpenAI Chat Completions, Responses, and Anthropic-compatible APIs.
- ReplaySSM and MTP3 for higher throughput without exceeding 24 GB VRAM.
- `low`, `medium`, `high`, and `xhigh` reasoning modes.
- Qwen3.8 image understanding with ReplaySSM and MTP3.
- Prefix reuse for faster repeated or shared prompts.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B v1 | [pinned compact artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c) | 20.84 GiB | **Recommended for RTX 3090; text C1-C6 at 4K and vision C1 at 32K** |
| Qwen3.6-35B-A3B v2 | [current upstream artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 21.22 GiB | Reader supported by v0.5+; includes DFlash payload and is not the measured 3090 artifact |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Supported with more runtime headroom |
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Validated at C1, C2, C4 and C8/MTP3 with ReplaySSM** |

NInfer-3090 v0.5 and newer recognize both v1 and v2 container magic. The current 21.22 GiB v2
artifact contains additional DFlash weights and is not the artifact used for the published RTX
3090 concurrency results. The pinned compact v1 artifact keeps the measured model payload and
omits DFlash, providing the known 24 GB memory profile.

## Models and platform support

Linux users build the applications from source. This fork targets RTX 3080 (sm_86) and a recent
NVIDIA driver.

Download the [official Qwen3.8 artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) as
`models/qwen3_8_27b.ninfer`, or run `scripts/download-qwen38.sh` (interrupted downloads resume).

For Qwen3.6-35B-A3B, the smaller
[pinned container-v1 artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c)
is recommended on a 24 GB card. Releases v0.5 and newer also read the larger container-v2 file.
An `artifact magic is not NInfer version 1` message means the executable is outdated, not that the
current model download is necessarily corrupt.

Developers can build from source on Windows or Linux. Windows uses Visual Studio 2022 and vcpkg.
Linux uses GCC 13 with system packages or the pinned vcpkg manifest. Both builds require CUDA 12.8
or newer and CMake 3.28 or newer.

See the [Windows build guide](docs/rtx-3090-windows.md) or the
[Linux build guide](docs/rtx-3090-linux.md). Ordinary Windows release users do not need these tools.

## Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `medium` uses the model's normal thinking
prompt. `high` injects a careful, verification-oriented instruction. `xhigh` injects the
checkpoint's extended deliberation instruction, asking it to validate assumptions and consider
alternatives. This is a real prompt-template change, not a sampling alias.

| Value | Qwen3.8 behavior |
|---|---|
| `none` | Disable thinking |
| `low` | Keep reasoning brief and focused |
| `medium` | Use normal Qwen3.8 thinking |
| `high` | Think the task through and verify key steps |
| `xhigh` | Use extended deliberation and verification |

OpenAI Chat Completions accepts a top-level `reasoning_effort` field:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Solve this carefully..."}],
  "reasoning_effort": "xhigh",
  "max_tokens": 4096
}
```

OpenAI Responses uses `"reasoning": {"effort": "xhigh"}`. Anthropic Messages uses
`"output_config": {"effort": "xhigh"}`. For the native CLI, pass
`--reasoning-effort low|medium|high|xhigh`; use `--no-thinking` instead of an effort to disable
reasoning. Chat Completions returns hidden reasoning separately as `message.reasoning_content`.

## Serving APIs

The server supports:

- OpenAI Chat Completions;
- OpenAI Responses Core with streaming and local continuation state;
- Anthropic Messages;
- compatible-prefix reuse;
- prompt-rendered function tools and parsed tool calls;
- bounded pending-request admission and JSONL request logs.

See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## How cohort batching works

The C number is the maximum number of requests NInfer can run together. C1 favors one interactive
user; C8 can combine up to eight active requests into each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running request finishes, the
next waiting request can join at a safe generation boundary. Empty or finished lanes are skipped,
so a C8 server also works normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The maximum number of
users and GPU memory are chosen when the server starts. In return, memory use stays predictable on
a 24 GB card and the server can reuse fast CUDA Graphs instead of rebuilding work continuously.

## Current limits

- One process owns one model; without `--tp` it executes on one GPU, with `--tp` it executes
  tensor-parallel across exactly two GPUs (greedy-only).
- Concurrency is fixed at startup and limited to 1-8 by the API; compact 35B fits C1-C6 and
  Qwen3.8-27B fits C8/8K with MTP3 through ReplaySSM.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- No CPU/GPU weight offload; TP beyond two ranks is not built.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4 and TMA kernels require Blackwell and are unavailable on SM86.
- The paged runtime exposes BF16, INT8, and experimental opt-in `rk8v4` KV. INT8 remains the
  quality-default path.

## Validation

The v0.6.0 Windows gate covered Qwen3.8 generation, materialization, request memory, admission,
paged KV, prefix reuse, speculative rounds, and SM86 W8 Linear paths.

The v0.6.1 Linux source gate completed all 245 Docker compile and link steps with CUDA 13.1 on
Ubuntu 24.04. Both Linux applications returned their `--help` output with GPU access enabled.
A real-artifact Linux generation and Linux performance qualification remain open.

## Upstream

NInfer-3090 is derived from [Neroued/ninfer](https://github.com/Neroued/ninfer). The upstream project
targets RTX 5090/`sm_120a`; this fork carries the Windows and Linux SM86 compatibility layer,
compact 35B artifact support, and RTX 3090-specific schedules and memory planning.

## Contributors

- [airtonix](https://github.com/airtonix) added Linux and Docker build and release support in
  [PR #1](https://github.com/Don-Chad/ninfer-3090/pull/1).
- [ColeWheatley](https://github.com/ColeWheatley) contributed SM86 runtime-count/GDN residency
  fixes, ECC diagnostics, and the GeForce-safe Docker fix in
  [PR #7](https://github.com/Don-Chad/ninfer-3090/pull/7).
- [justinlime](https://github.com/justinlime) added NixOS build support in
  [PR #5](https://github.com/Don-Chad/ninfer-3090/pull/5).
- [sry9681](https://github.com/sry9681) contributed the device-wide GPU-memory startup fix in
  [PR #6](https://github.com/Don-Chad/ninfer-3090/pull/6).

## Contributing

Please read the [Pull Request Policy](PR_POLICY.md) before opening an issue or pull request.
It explains how to keep changes focused and how to document correctness, performance, VRAM, and
compatibility evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE).
