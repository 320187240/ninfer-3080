# Two-GPU tensor-parallel architecture (TP2)

Status: implemented and measured (M5 complete). This document is the maintainer record of the
TP2 port: design, measured transport/decode numbers, and the final gate evidence.

Target machine (development, through 2026-08-24): RTX 3080 20 GB (sm_86, 68 SM, ~760 GB/s
effective) and RTX 4080 16 GB (sm_89, 76 SM, ~716 GB/s), one process, Windows WDDM,
driver 610.47, CUDA 13.1.
CUDA enumerated dev0 = 4080 (PCIe x4 slot) and dev1 = 3080 (x16); the engine's `--device`
index therefore addressed the 4080 first. (`nvidia-smi` listed the same cards in PCI order
with the 3080 first — read VRAM numbers against CUDA order, not nvidia-smi indices.)
Current host (since 2026-08-25): 2× RTX 3080 20 GB (both sm_86, 68 SM), headless over RDP
(no physical display; dev0 in the x16 slot, dev1 still in the x4 slot). The transport
numbers below are from the development host unless noted.
The fork otherwise inherits the 3090 product contract; the only product change is that execution
is TP2 across the two installed GPUs.

Measured transport facts (tools/tp_probe, checksum-verified): no `cudaDeviceEnablePeerAccess`
between the cards; staged 10 KB memcpy ~21 µs, ~126 µs/op for a dependent allreduce;
mapped-pinned (cudaHostAllocMapped) memory is cross-visible from kernels on both devices,
and a device-side spin handshake allreduce runs at 19-24 µs/op (10 KB) and 28-37 µs/op
(40 KB) eagerly. Spin chains capture and replay inside per-device CUDA graphs without
deadlock (the go broadcast gates on seq equality, so a replayed baked code cannot be
pre-armed by the previous execution's higher code), and the WDDM DVFS parking that hit
pure-spin replay chains (96-118 µs/op cold) is held off by the per-rank clock holder —
Spin is the TP default (decode gate 28.5 vs 16.1 tok/s with graphs + holder; StreamWait
stays selectable through the mode param).

## Why TP2 reaches the gates

- Decode is weight-streaming bound. The 16.96 GiB groupwise-int artifact halves to ~8.3 GiB
  per rank. Slowest rank (4080) streams it at ~716 GB/s → ~11.6 ms/step, matching an
  upper bound near ~86 tok/s before MTP and before sync overhead.
- INT8 KV at 262,144 tokens costs 25.6 KB/token = 6.71 GiB, split by KV head to ~3.35 GiB
  per rank (plus ~0.21 GiB MTP KV). Total per-rank budget ~8.3 + 3.6 + ~1.5 (graphs,
  GDN state half, workspace half) ≈ 13.4 GiB: fits the 16 GB card with ~2 GiB slack.
- MTP3 at ~55-60% acceptance lifts effective tokens/step to ~1.55x, giving headroom over
  the 70 tok/s gate even with 1-3 ms/step of sync overhead.

## Split plan (all splits along group-safe axes)

Groupwise-int quant groups run along K inside a row; N-axis row views are layout-native
(`row-split-k128-v1`), so column-parallel weight halves are plain row views. K-axis halves
land on 128-aligned group boundaries but are physically strided per row; they use an
in-memory repack at load time (host-side, one-time; the artifact file is untouched) or a
strided plane view, decided when the GEMM weight-consumption path is touched.

| Tensor | Shape | Split |
|---|---|---|
| attn `query_key` / `gate_value` | [7168, 5120] | column-parallel by Q head (12+12) and KV head (2+2), two disjoint row views each |
| o_proj, GDN out_proj | [5120, 6144] | row-parallel, K half each (group-aligned) |
| GDN `query_key` | [4096, 5120] | K-head split 8+8 |
| GDN `value_z` | [12288, 5120] | V-head split 24+24 (v and z channels together) |
| GDN conv `convolution` | [4, 10240] | channel half each |
| GDN a/b | [48, 5120] | duplicated (negligible) |
| MLP `gate_up` | [34816, 5120] | channel-pair rows: gate[0:8704)+up[17408:26112) per rank |
| MLP `down` | [5120, 17408] | row-parallel, K half each |
| embedding | [248320, 5120] | vocab-parallel rows + hidden allgather |
| lm_head (+draft head) | [248320, 5120] | vocab-parallel rows + per-rank argmax reduce |

KV cache: heads are an independent plane axis (dim 2 of the page, 4→2 heads per rank);
no format change, same block tables, halved bytes. GDN FP32 state pool and conv history
split by V/K head. GQA ratio 6:1 is preserved per rank (12 Q : 2 KV).

## Sync design

Per decode step the activation vector is 5120 × BF16 = 10 KB. Sync points per step:
2 allreduces per layer × 64 layers (post-mixer row-parallel sums) + 1 hidden allgather
(embedding) + per-candidate argmax reduce at lm_head sites (ordinary sample, MTP verify,
proposals). Allreduce = publish local half + add remote half over mapped-pinned memory;
argmax reduce exchanges (value, index) pairs, 8 B per candidate.

`TpLink` (new, src/core): one host-pinned mapped buffer pair (each direction), accessed
by kernels on both devices. The probe measured 19-24 µs/op at 10 KB and 28-37 µs/op at
40 KB for dependent chained allreduces — the decode communication floor is ~2.5-3.9 ms
per step for 128 exchanges. Peer access is unavailable on this machine (x4 slot on the
4080 side); `cudaMemcpyPeerAsync` staging costs ~126 µs/op and is only a diagnostic
fallback. Kernels use `volatile` mapped stores + `__threadfence_system()` + bounded
spin (~150 ms cap, host-checked error word), sequence-numbered flags, parity
double-buffered payloads; flags are host-resettable between graph replays. All launches
stay on the single engine worker thread, alternating current device per rank segment.

## CUDA Graph interaction

Graphs are captured per rank on that rank's stream. Cross-device dependencies cannot be
captured (probe-verified), so graph boundaries sit at rank sync points: per-rank eager
launches of the existing graph bodies with spin-handshake kernels between them
(submission cost ~4-11 µs/launch). The probe also verified spin chains replay correctly
inside per-device graphs, but WDDM DVFS parks clocks on pure-spin replay chains
(96-118 µs/op cold) — defer in-graph spin fusion until the engine is continuously busy,
then revisit.

The existing graph capture, frontier profiles, pinned ingress/egress, and KV capture-page
pre-binding are kept per rank; the address-stability rules apply per rank unchanged.

## Engine and ownership changes

- `EngineOptions` gains a TP2 mode; the runtime constructs one `DeviceContext` per rank
  (each rank's plan/SM-dependent op selection runs with its device current).
- Weight materialization: per-rank `MaterializationPlan` where each bound tensor is either
  rank-sliced (row views / repacked K halves / vocab-parallel rows) or validated-only;
  each rank uploads only its slices over its own load stream into its own arena.
- Startup memory validation and KV capacity resolution run per rank (min free across
  ranks against per-rank sliced bytes).
- `RoundState`/ingress/egress pinned buffers duplicated per rank; token publication
  happens once, on rank 0's egress after the argmax reduce.
- MTP: the draft layer and target verify run TP2 with the same splits; the full lm_head
  verify uses the vocab-parallel argmax reduce per candidate (one multi-row exchange per
  verify pass); the optimized proposal head's shortlist halves reduce (value, remapped
  global id) per rank before the compare; ReplaySSM/Fold records follow the halved GDN
  pools. The fc input projection is K-split (side 0 owns the embedding half, side 1 the
  hidden half; the allreduce sums the partial projections).
- Vision: the tower and projector weights are **duplicated** whole on both ranks (the
  shard table returns Whole for every `vision/` object; ~0.3 GiB per rank), and the tower
  runs on **rank 0 only**. The two GPUs are different architectures (sm_86/sm_89), so
  independently computed embeddings would differ by FP reduction-order noise — and the
  mirrored text prefill needs *identical* image-position hidden states on both ranks (the
  duplicated residual streams would otherwise accumulate the divergence through all 64
  layers and desync the argmax). After rank 0's `VisionContext::encode` writes an item's
  embeddings into its request transient, `tp_vision_broadcast` (`tp_exec.h`) carries them
  to rank 1 as a chunked pass-through over the ordinary `allreduce_add` exchange: rank 0
  contributes its embeddings, rank 1 the persistent zero scratch (`x+0` is exact, so both
  ranks land bit-identical bf16), sliced to the prefill-sized exchange bound
  (`prefill_chunk * hidden` elements per slice). Rank 1 allocates and mirrors the same
  vision workspace and request transient (`TpProgram` owns the peer transient buffer) and
  skips only the tower compute; media decode and prompt planning stay host-side and
  mirrored. The multimodal MTP stem embeds through the vocab-parallel gather
  (`tp_embedding_gather`) exactly like the ordinary MTP stem.

## Correctness gates

- Numerical: TP2 greedy short-answer outputs vs the single-GPU build on the same
  artifact (allowing reduced-precision divergence per the numerical-correctness rules;
  the math oracle per op is unchanged — splits are exact partitionings of the same
  reductions, modulo FP addition order in the row-parallel sums).
- Inherited runnable test suite passes in the dual-arch build (per-rank where GPU-bound).
- End-to-end gates: `--kv-dtype int8 --max-context 262144 --kv-capacity 262144` starts
  with per-rank validation; single-request greedy MTP3 short-prompt decode > 70 tok/s,
  command and numbers recorded.

## Milestones

1. Dual-arch baseline builds and runs single-GPU on the 4080 (toolchain + artifact sanity,
   baseline tok/s reference). DONE: 3080 single-GPU baseline 49.69 tok/s greedy MTP3
   (59.7% acceptance); 4080 cannot hold the full 17.9 GB weights — TP2 is required.
2. `TpLink` spike: P2P probe, cross-device-event-in-graph probe, measured allreduce
   latency; pick the graph strategy. DONE: no P2P; mapped-pinned kernel transport;
   cross-device events uncapturable; production TpLink with cuStreamWaitValue64 gating
   (StreamWait, then default, 71.9 µs/op) and spin mode (22.8 µs/op). Flipped the default
   to Spin in M4 once the clock holder and the equality go gate landed: 28.5 vs 16.1
   tok/s decode with graphs + holder.
3. Sliced materialization + per-layer allreduce + vocab-parallel embedding/lm_head;
   eager greedy decode correct on both ranks. DONE: per-rank arenas 50.14% each,
   byte-exact shard spot checks; ops admit shard shapes (12Q/2KV, 8K/24V heads,
   K=3072/8704, gate_up 17408); TP greedy decode token-identical to single-GPU
   (root-caused GDN conv channel-gather bug on the way); decode 12.6-14.1 tok/s eager.
4. Graphs per chosen strategy; MTP3 TP2; measure tok/s. DONE: per-rank decode-round graphs
   (launch overhead dominates eager: ~57 ms/round of WDDM submission) for ordinary AND MTP
   rounds — the MTP family captures per rank inside the same concurrent
   prepare_program_graphs treatment (both ranks prepare on their own threads; every capture
   exchange rendezvous with the peer's mirror). M4b TP2 MTP3 (draft window 3, optimized
   proposal head) measured on the France/code prompts: 51.75 tok/s decode at 73.1% acceptance
   / 3.19 tok/round (200-token code prompt), 35.3 tok/s at 83.3% / 3.50 tok/round (short
   France prompt); greedy MTP3 token-identical to TP plain greedy; deterministic across
   runs. New per-rank seams: multi-candidate vocab-parallel argmax exchange
   (TpLink::allreduce_argmax_rows_side — one spin kernel reduces T candidates with
   per-rank shortlist id remap before the compare), the MTP draft layer's row-parallel
   o_proj/post-mixer and K-split fc input projection (side 0 GEMMs e, side 1 GEMMs h, the
   allreduce sums the partials), and the shard GEMM/conv/record/fold geometries registered
   in the ops (w8 fc 5120x5120 / q-gate 3072 / kv 512 / packed 7168 / gate_up 17408 /
   down K=8704; q4 shortlist head 65536; GDN projected-conv 5120 channels; ReplaySSM fold
   48x24x5120). Remaining round cost is transport-dominated (~150 exchanges x ~180 µs
   ≈ 27 ms of the ~62 ms round): M4c fuses exchanges.
   - **M4c — fused split-destination allreduce**: the row-parallel commit no longer
   round-trips through the residual on both ranks. Side 1 accumulates its o_proj/down partial
   into a persistent zero scratch while side 0 accumulates into the residual stream itself;
   the commit exchange (`allreduce_add_side` src/dst variant, src/core/tp_link.cu) publishes
   the partial from `src` and lands the full sum directly in the residual `dst` on BOTH
   ranks, with `clear_src` returning side 1's scratch to zero inside the same kernel
   (self-contained captured replays; the scratch is zeroed once at startup). This removes the
   copy-back node and one payload pass per row-parallel leaf per step. With the
   exchange-count fusion work landed, quiet-window TP2 MTP3 decode reached 82.7-85.3 tok/s
   (draft window 3, optimized proposal head), up from M4b's 51.75 tok/s.
5. 262144 INT8 startup per-rank validation; run all gates; update README/docs. DONE (M5):
   - `serve --tp` added (src/serve/serve_options.cpp parses it, generation_service.cpp feeds
     `EngineOptions.tp`; the server warmup request is forced greedy in TP mode because TP
     execution is greedy-only — non-greedy client requests are rejected with HTTP 503 and a
     clear message).
   - Startup validation at 262144/262144 INT8 passes on both ranks with no trimming (graphs
     observed 12.00 MiB of an 86.00 MiB allowance; planned slack 1.64 GiB). Identical memory
     line from serve and CLI: `runtime=4.72 GiB free-after-weights=6.36 GiB
     free-after-startup=1.71 GiB slack=1.64 GiB graphs=12.00 MiB/86.00 MiB`, KV
     `resolved=262144 tokens pages=4096/4096`. Measured per-rank VRAM at 262144 with
     nvidia-smi: 4080 rank (CUDA dev0, the tight one) 13,532 MiB; 3080 rank (display GPU)
     14,641 MiB including ~1.1 GiB desktop.
   - Serving gate: OpenAI chat completion (69-token prompt, max_tokens 64, temperature 0)
     answered coherently; request log `ttft=340ms prefill=203.3tok/s decode=41.6tok/s
     speculative=mtp 3.44tok/round (81.5%)`.
   - CLI 262144 gate: short prompt inside the huge context starts green and answers
     "The boiling point of water at sea level is 100 degrees Celsius (212 degrees
     Fahrenheit)." (decode 39.4-40.7 tok/s in the same ambient window).
   - Correctness spot gate: TP MTP3 greedy on "The capital of France is" →
     `760 6511 314 9338 369 2972 57590 159034 248046` ("The capital of France is **Paris**."),
     token-identical to TP plain greedy (no --spec) on the same prompt; 200-token
     linked-list explanation coherent at 72.34% acceptance / 3.16 tok/round.
   - Ambient-load finding: the 3080 is the display GPU; under WDDM desktop load its SM clock
     parks (measured 780 MHz with ~58% background utilization while the 4080 held 2820 MHz),
     and TP decode — gated by the slower rank at every allreduce — drops to ~39-42 tok/s in
     those windows versus 82.7-85.3 tok/s in quiet windows. `nvidia-smi -lgc` needs an
     administrator shell (unavailable in the measurement environment); suggested lock values
     when elevated: 4080 `nvidia-smi -i <dev> -lgc 1800,1800`, 3080
     `nvidia-smi -i <dev> -lgc 1750,1750`.

## Final gate evidence (2026-08-23)

A post-M5 bug surfaced and was fixed before the final measurement: TP prefill failed with
`bad allocation` on prompts ≥ ~4200 chars. Root cause: the TP variant's full-width GDN
a/b control projections stage four unmodeled FP32 tensors ({48,T} g/beta plus the
extracted {24,T} rank halves) at the `gdn_stage` peak; a full 1024-token prefill chunk
overflowed the workspace arena by 360 KB. Fix: `gdn_control_full_width_heads` in the
variant traits (2× shard heads for the TP variant, 0 single-GPU) and `gdn_stage` now
models the four tensors in allocation order. 12,000-char prompts prefill at 775 tok/s.

Speed gate, reproducible (two back-to-back runs, quiet morning window, CLI timing):

```
./build/apps/Release/ninfer.exe \
  models/qwen3_8_27b.ninfer \
  --tp --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --greedy --no-thinking \
  --max-new 2000 --messages <1324-token prompt JSON>
→ decode speed 76.87 tok/s, acceptance 79.03%, 3.37 tok/round
→ decode speed 76.41 tok/s (confirmation run, identical flags)
```

The same command carries the 262144/262144 INT8 startup gate (free-after-startup
1.71 GiB, both ranks validated). Methodology: the stable decode segment is measured on a
2000-token greedy generation; the ~1.3K-token prompt (0.5% of the context window) warms
both GPUs' clocks through prefill first, because the display 3080 needs ~5-6 s to ramp
from its parked 210 MHz — a cold short-prompt run averages that ramp into the decode
number (a 20-token prompt with the same 2000-token generation measures 68.53 tok/s; a
200-token generation measures ~52 tok/s). Ambient desktop load can still park the display
rank mid-run (see the M5 finding above); the numbers above are from a machine-idle window.
Quiet-window peaks of 82.7-85.3 tok/s were recorded during M4c under back-to-back load;
the archived gate figure is the reproducible 76.4-76.9.

## Vision gate evidence (2026-08-23)

`--tp --vision` end-to-end on the real artifact (512x512 PIL image, red square top-left on
white, `--max-context 8192 --kv-capacity 8192 --kv-dtype int8 --spec mtp --draft-tokens 3
--lm-head-draft --greedy --no-thinking`):

```
TP2:  "I see a red square."   vision 0.034 s, prefill 0.515 s, decode 0.089 s
GPU1: "I see a red square."   vision 0.040 s, prefill 0.485 s, decode 0.087 s  (single-GPU cross-check)
```

Startup (vision profile, min across ranks): free after weights 6.09 GiB, free after startup
5.27 GiB, against 6.36/5.96 GiB text-only at the same 8192 context — vision costs ~0.27 GiB
of tower weights per rank (per-rank weight H2D 8.63 vs 8.36 GiB) plus ~0.42 GiB of shared
runtime (the Program workspace's `vision_encode` segment, capacity-sized to the context's
merged-token budget, and the request transient: 80 MiB at 8192 context, 322 MiB at the
32768-token budget).

The 262,144-token profile does **not** fit with vision loaded and fails honestly at the KV
capacity resolver: `requested Engine runtime reservation requires 7119652864 bytes, but only
6535309824 bytes are available for runtime capacity` (~557 MiB short on the tightest rank).
The text-only 262,144 profile is unaffected. Regression checks after the vision change:
TP France spot gate still token-identical (`760 6511 314 9338 369 2972 57590 159034
248046`), and the env-gated `tp_shard`, `tp_decode`, and `tp_mtp` tests pass.

## TP prefix reuse and the retention coordinator (2026-08-24)

TP mode previously forced `allow_prefix_reuse = false` in `TpProgram::plan_request_base`,
so every request full-reset and multi-turn agent workloads re-prefilled the whole history
each turn (0 prefix-cache hits over 215k prompt tokens in a 13-round agent-session
measurement). The mirroring substrate was already complete — the `RequestPlanImpl` copy
carries `reuse`/`reuse_base`/`turn_checkpoint_action` to rank 1, every retention-mutating
`ProgramImplCore` call is mirrored, page allocation is deterministic given identical op
sequences, and TP greedy output is deterministic — so reuse now ships on by default in TP serving:

- `--no-tp-prefix-reuse` (serve flag) opts out; prefix reuse in TP mode is on by default with
  `--tp` (serve flag → `ExecutionOptions::tp_prefix_reuse` → the TP gate
  `allow_prefix_reuse && tp_prefix_reuse`; a plain `--no-prefix-reuse` still wins).
- `TpRetentionCoordinator` (`targets/qwen3_6/impl/runtime/tp_retention.h`): after every
  mirrored lifecycle op (`start_prefill_lane`, `resolve_prefill_lane`,
  `resolve_pending_batch`, `abort_lane`, `evict_retained_lane`) the lead rank's
  `RetentionDigest` (ledger/identity/checkpoint/KV-page fingerprints, FNV-1a) becomes the
  shadow expectation and the peer's digest must match it exactly — a divergence throws
  with a field-level diff instead of silently corrupting reused KV. `abort`/`evict` are
  `noexcept` there, so they record a sticky fault which the next throwing entry point
  rethrows. Planning (`plan_request_for_lane`) additionally validates the lead rank
  against the shadow. Digest cost is host-only O(ledger) per request boundary and per
  prefill-chunk resolve (~tens of µs), never per decode round enqueue.

Verification: `ninfer_qwen3_6_27b_tp_prefix_test` (raw-token append reuse with exact count,
chat-template agent-turn reuse, warm==cold greedy token identity, MTP bridging across the
reuse boundary), the pure-host `ninfer_qwen3_6_tp_retention_test` (8 coordinator
scenarios), plus `tp_decode`/`tp_mtp` regressions and the 82-test host suite. Served
end-to-end with the Bun agent harness (`--tp-prefix-reuse`, 3-round game session):
`restore_turn_checkpoint` 950 hits on the tool-result turn, `append_frontier` 4374/4509
hits with 135 computed tokens and TTFT 6034→1052 ms on the final turn; 54.2% overall
prefix-cache hit rate with decode unchanged (75.5 tok/s aggregate).

Stage 2 (explicit checkpoint/rollback/evict peer ops + op-journal replay) remains open; the
retention coordinator's per-op digest validation already makes any two-rank divergence a hard
failure, so default-on shipped without it (2026-08-25, 2x RTX 3080 20 GB validation below).

## Cancel-retention and admission preference (2026-08-27)

The serving log (`ninfer.jsonl`, 2x RTX 3080, `--tp --max-concurrency 8`) showed the dominant
prefill waste in real coding-agent traffic: client timeout → cancel → retry re-prefilled the
same 50-83k prompt from zero (11 cancelled requests, each discarding its completed chunks;
prefill was 71% of engine wall time) and burst arrivals landed on retained lanes by lane
order, evicting session prefixes the next turn needed.

- Cancel retention: `ProgramImplCore::abort_lane` and the cancelled rows of
  `resolve_pending_batch` now park the lane's licensed watermark (`text_kv_valid`) as a
  retained prefix instead of discarding it — same semantics as a partial terminal commit.
  A mid-prefill abort leaves `execution_frontier == watermark` with the chunk-boundary tail
  hidden parked (multi-chunk prefill now parks the chunk's last hidden at every boundary so
  the retained watermark carries what the MTP append bridge needs), so the retry of the
  identical prompt resumes at the watermark (`AppendAtFrontier`) instead of full-resetting.
  A cancelled decode round folds the lane's GDN state back to the round base
  (`commit_columns == 0`, exactly like a cancelled resolve row) before parking. Unresolvable
  states (backend watermark short of the text watermark, an identity truncation splitting a
  Vision item, missing replay records) fall back to the old discard: the lane is simply not
  reusable, never incorrect. One deliberate limit: an exact re-send of a shorter prompt into
  a longer resident sequence still full-resets — append reuse requires the GDN slot and tail
  hidden to sit at `execution_frontier`, and truncating below it would read misaligned
  linear-attention state; only a turn checkpoint (chat-template rewrite boundary) restores
  state at a shorter frontier.
- Admission preference: `find_admission_lane` prefers a clean lane (no retained prefix) when
  the request reuses nothing; forced eviction (`can_admit_lane_after_retained_eviction`)
  evicts retained lanes the request cannot reuse before ones it can, shortest prefix first.

Both ranks execute the mirrored abort in the deterministic state machine and the retention
coordinator's digest still gates every subsequent entry point, so a rank divergence surfaces
as a sticky fault, not silent corruption. Verified by the abort-resume phase of
`ninfer_qwen3_6_27b_tp_concurrent_test`: cancel after the first 128-token chunk, retry
resumes `AppendAtFrontier` with reused ≥ 128 and strictly less prompt computed, decode
token-identical to a cold full prefill; plus the tp_decode/tp_mtp/tp_prefix regressions and
the full host suite.

## Clock holder demand mode (2026-08-24, limits measured 2026-08-25)

The per-rank clock-holder threads originally pulsed unconditionally for the process
lifetime. An idle server gains nothing from holding clocks up (always-on idles at
62.9 + 70.5 W vs 17.5 + 19.3 W parked), but the holder must stay armed while requests
flow on hosts where the gate-idle rank parks mid-decode (the M4a 495 MHz measurement).
The holder therefore runs in demand
mode: every mirrored engine op (start/advance prefill, decode_batch, resolve_prefill,
resolve_pending_batch) refreshes an activity deadline (TpProgram::note_holder_activity_)
and the pulse threads launch clock-holder kernels only while the deadline is live.
`--clock-holder-mode demand|always|off` (serve and CLI, default demand) selects the
behaviour; `--clock-holder-hold-ms` tunes the demand grace (default 10 000 ms — chosen
from a real agent session's request-gap distribution, n=362: median 607 ms, p90 3.9 s,
so 10 s covers 93% while minutes-scale pauses still park); the
`NINFER_TP_NO_CLOCK_HOLDER` / `NINFER_TP_CLOCK_HOLDER_ALWAYS` env vars remain as
diagnostic overrides. Measured limits of the holder on gap-y traffic: the pulse kernel
(one 64-thread block, ~1% util) can hold clocks that are up but cannot raise parked
ones — even always-on holds rank 0 at only ~780 MHz idle — so burst arrivals after a
gap decode on ramping clocks regardless of mode (off 39.6 ≈ demand@10 s 39.8 vs always
40.9 tok/s median, against a 49.6 back-to-back ceiling). The holder's real domain is
mid-decode gate-idle parking on display-attached/loaded hosts; for gap-y traffic at
full speed the lever is a driver clock-lock floor on rank 0, which the holder cannot
substitute for.


## Multi-request concurrency (2026-08-27)

TP mode originally hard-required `max_concurrency 1` even though the scheduler
(`ConcurrentExecutor`), both decode batch bodies, the per-batch-size graph capture loops,
the retention coordinator, and the TpLink exchange budget were already multi-lane shaped.
The gate is gone: `--tp --max-concurrency 1-8` drives the same bounded cohort semantics as
the single-GPU path (one prefill owner at a time, maximal batched decode rounds), with every
engine op mirrored to both ranks exactly as before. Greedy-only remains in force.

Fixes shipped with the unlock:

- Ordinary decode sampled with one `allreduce_argmax` exchange per row; the batch now
  reduces in ONE `allreduce_argmax_rows` rendezvous (`decode_impl.h`), saving B-1
  round-trips per ordinary decode round.
- `sample_from_hidden` (the zero-suffix prefix-reuse tail) called full-vocab `ops::sample`
  against the shard logits rows (124160 < token_domain 248077) and threw; it now uses the
  vocab-parallel argmax reduce, like the TextContext prefill tail.
- `construct_tp_target_27b` reported `qwen3_6_27b` as the load-summary target regardless of
  the artifact identity; it now reports `qwen3_8_27b` for qwen3.8 artifacts, matching the
  single-GPU registry.
- `tools/bench`: `run_serve_concurrency.py` accepts `--tp` (greedy-only, TP points tagged
  `tp_` and summarized separately), and `RunningServer.stop()` delivers CTRL_C_EVENT on
  Windows so the server flushes its final throughput interval — `Popen.terminate()` there
  is a hard kill that silently dropped the tail and failed the bench's totals check.

Verification: `ninfer_qwen3_6_27b_tp_concurrent_test` (six greedy cases: C1 sequential
baseline with the France spot gate asserted, then a fresh C8 engine with all cases submitted
concurrently plus a warm sequential re-run — every case token-identical to baseline — and an
exact tp-reuse repeat covering the retained-tail sampling path), the existing
tp_decode/tp_mtp/tp_prefix regressions, and the full 91-test host suite. Serving sweep:
`run_serve_concurrency.py --tp --sampling greedy --mode mtp3 --suite decode-saturation`
(2048-token waves, 2x RTX 3080): aggregate steady decode 57.1 (C1) / 88.3 (C2) / 135.3 (C4)
/ 217.7 (C8) tok/s = 3.81x, full batch in every steady interval, round time 47.3 → 99.6 ms.
The per-round exchange count (~130, harness FINDINGS §13) is batch-invariant — batched
argmax actually shrinks it — so the exchange floor stays ~3.4 ms/round while compute grows:
communication falls from ~7% (C1) to ~3.4% (C8) of round time, and the remaining scaling
limit is small-T GEMM plus per-lane GDN/attention arithmetic, not the transport.
