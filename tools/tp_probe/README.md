# tp_probe — 2-GPU cross-communication facts (Windows WDDM, single process)

Standalone CUDA probe measuring the cross-GPU communication facts needed to design a
2-GPU tensor-parallel inference engine on this machine. One self-contained source file,
no dependencies beyond CUDA 13.1 and the MSVC host compiler used by `nvcc`. Q1–Q5
established the baseline (no P2P, staged memcpy ~126 µs per 10 KB allreduce, no
cross-device events in graphs). Q6–Q10 test the proposed fix: host-pinned mapped
memory + device-side spin flags as the allreduce transport, including inside CUDA
graphs.

## Environment

| Item | Value |
|---|---|
| OS / display mode | Windows, WDDM (no TCC) |
| Driver | 610.47 (driver API / runtime 13030) |
| CUDA toolkit | nvcc 13.1 (V13.1.115) |
| device 0 (CUDA order) | NVIDIA GeForce RTX 4080, sm_89, PCIe Gen4 **x4** (bus 07:00.0) |
| device 1 (CUDA order) | NVIDIA GeForce RTX 3080 20GB, sm_86, PCIe Gen4 x16 (bus 01:00.0) |
| process model | single process, two contexts, `cudaDeviceScheduleSpin` |

Note: CUDA enumerates the 4080 as device 0 and the 3080 as device 1 (opposite of the
informal naming). The binary embeds `compute_86` PTX and JITs on the sm_89 4080 without
issue. The 4080's x4 slot is the bandwidth bottleneck for every transfer touching it.

## Build

```bash
nvcc -O2 -std=c++17 -gencode arch=compute_86,code=compute_86 \
     --compiler-bindir "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/<version>/bin/Hostx64/x64" \
     -o tp_probe.exe tp_probe.cu
./tp_probe.exe
```

`<version>` is the installed MSVC toolset version (see
`dir "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"`).

## Methodology

Warmup then timed window on device-side `cudaEventElapsedTime`, min of 3 runs;
1000 iterations per transfer measurement; 128-iteration windows for the allreduce.
Every CUDA call is checked; the final run reported 0 CUDA errors. Correctness of every
synchronization claim is verified with a delayed-producer + checksum pattern (a
~130 µs `clock64()` spin kernel writes alternating values, the consumer must observe
exactly the value of the current round). CUDA events are bound to the device current at
creation; the probe keeps separate event pools per device (recording a dev0-created
event on a dev1 stream fails with `invalid resource handle` — relevant API fact).

Q6–Q10 methodology: one 10 MB `cudaHostAlloc(Mapped|Portable)` region is shared by both
devices (same virtual address on each). The spin-allreduce kernel is a single kernel per
device per iteration: publish local data into the mapped buffer with volatile stores,
`__threadfence_system()`, grid-wide arrival counter (last block releases a sequence-numbered
flag), bounded spin (`clock64`, ~150 ms cap, far under the 2 s WDDM TDR) on the peer flag,
device-side go-flag broadcast, then consume + add the peer payload. Payloads are parity
double-buffered; flags are sequence numbers, reset from the host between rounds. Two host
threads (one per device, own stream) drive the eager path; graphs contain kernels only.
Every timing number is checksum-verified (exact bf16-feedback allreduce chains, exact fp32
accumulators) and min-of-3 after warmup. Q8 additionally runs both launch orders with a
10 s host watchdog, 12 consecutive replays (DVFS warm-up test), a compute-burn
hot-clock control, and on-device `clock64` instrumentation (peer-flag wait duration,
kernel start-to-start cadence → implied SM clock). Q8 executes last so a hypothetical
deadlock cannot poison earlier questions. `eff GB/s` for Q10 counts 2×payload per
allreduce op (data crosses each GPU's link once per direction).

## Results (QUESTION=ANSWER)

```
Q1 PEER_ACCESS=UNSUPPORTED
Q1 DETAIL: cudaDeviceCanAccessPeer(0,1)=0 ; (1,0)=0
Q1 DETAIL: cudaDeviceEnablePeerAccess -> "peer access is not supported between these two devices" (both ways)
Q2 PEERAPI_0TO1_10KB=19.37us_0.53GBps   (cudaMemcpyPeerAsync, driver stages internally via host)
Q2 PEERAPI_1TO0_10KB=22.12us_0.46GBps
Q2 STAGED_0TO1_10KB=21.25us_0.48GBps    (manual D2H+pinned+H2D, lockstep, 2 cross-device event handshakes)
Q2 STAGED_1TO0_10KB=21.92us_0.47GBps
Q2 PEERAPI_0TO1_512KB=122.48us_4.28GBps
Q2 PEERAPI_1TO0_512KB=119.47us_4.39GBps
Q2 STAGED_0TO1_512KB=121.87us_4.30GBps
Q2 STAGED_1TO0_512KB=119.21us_4.40GBps
Q3 PEER_MAPPED_WRITES=SKIPPED            (peer access unavailable; a dev1 kernel cannot receive a dev0 pointer)
Q4a CROSS_DEVICE_EVENT_STREAM_SYNC=WORKS (both directions, 100/100 rounds correct)
Q4b GRAPHS_CROSS_DEVICE_EVENTS=UNSUPPORTED
Q4b DETAIL: capture error (Global AND Relaxed modes, both directions):
       "dependency created on uncaptured work in another stream"
       (cudaStreamWaitEvent on an event recorded outside the capture, on the other device's stream)
Q4b DETAIL: capture-time cross-device cudaEventRecord (record on dev0 stream while capturing
       dev1): succeeds ("no error") and does not break the capture — but the wait side cannot
       be captured, so this yields no usable cross-device graph synchronization
Q4b DETAIL: graph-internal cross-graph event chain (record node in graph A, wait node in
       graph B, event never recorded before capture): graph A captures, graph B fails with
       "invalid argument" (Relaxed mode included) — cross-graph event chains are not capturable
Q5 ALLREDUCE10KB_EAGER_STAGED=126.24us_op 16.159ms_per_128
Q5 ALLREDUCE10KB_EAGER_PEERAPI=131.31us_op 16.808ms_per_128
Q5 ALLREDUCE10KB_GRAPH=NOT_AVAILABLE_OR_UNSAFE
Q6 MAPPED_PINNED_CROSS_VISIBLE=WORKS
Q6 DETAIL: cudaHostAlloc(Mapped|Portable) 10MB OK; cudaHostGetDevicePointer identical VA on both
       devices (UVA); 50 alternating kernel-write -> other-GPU-kernel-read rounds checksum-exact
Q7 SPIN_ALLREDUCE_EAGER_10KB=19-24us_op ~2.5ms_per_128 (6.3x faster than 126us staged baseline)
Q7 SPIN_ALLREDUCE_EAGER_10KB_NSLEEP=18-23us_op (nanosleep(400) backoff ~= tight poll)
Q7 SPIN_ALLREDUCE_EAGER_40KB=28-30us_op ~3.8ms_per_128 (4.3x faster than staged)
Q8 SPIN_ALLREDUCE_IN_GRAPHS_10KB=WORKS 112.63us_op 14.417ms_per_128 hot_clock=70.55us_op
Q8 SPIN_ALLREDUCE_IN_GRAPHS_40KB=WORKS 96.26us_op 12.321ms_per_128 hot_clock=59.46us_op
Q8 DETAIL: no deadlock in either launch order (B-then-A, A-then-B), no spin timeouts, checksums
       exact; 12 consecutive replays stay 104-128us/op (no warm-up effect)
Q8 DETAIL (root cause of slow graphs): kernel-internal peer-flag waits are ~5us and cycle counts
       per iteration equal the eager path, but implied SM clock during graph replay is only
       0.20-0.65 GHz vs 1.3-2.7 GHz when the same kernels are eagerly launched -> WDDM DVFS
       keeps both GPUs in a low P-state for graph-replayed spin chains; a dense compute burn
       right before each replay recovers only part of it (70/59us), and in one confirmation
       run the hot-clock 10KB replay reached 18.7us/op -- full eager-level speed, proving the
       graph path itself is not slow, only clock-state-dependent (unreliable on WDDM)
Q8 GRAPH_TINY_NODE_SCHED=2.6-3.9us_per_node (128 chained nodes/graph, no comms)
Q8 GRAPH_MAPPEDWRITE_NODE_SCHED=6.2-6.6us_per_node (same, but nodes write mapped host memory)
Q9 GRAPH_LAUNCH_10NODE_SUBMISSION_dev0=3.96-9.46us dev1=6.74-11.24us (1000 back-to-back
       submissions per thread, two threads; run-to-run varies with WDDM batching)
Q10 MAPPED_SPIN_ALLREDUCE_10KB=25.09-25.67us_op 0.80GBps_eff
Q10 STAGED_ALLREDUCE_10KB=122-128us_op 0.16-0.17GBps_eff
Q10 MAPPED_SPIN_ALLREDUCE_40KB=35.10-37.10us_op 2.21-2.33GBps_eff
Q10 STAGED_ALLREDUCE_40KB=130.13-134.06us_op 0.61-0.63GBps_eff
Q10 MAPPED_SPIN_ALLREDUCE_160KB=75.62-77.13us_op 4.25-4.33GBps_eff
Q10 STAGED_ALLREDUCE_160KB=193.93-196.60us_op 1.67-1.69GBps_eff
Q10 MAPPED_SPIN_ALLREDUCE_640KB=224.23-230.32us_op 5.69-5.85GBps_eff
Q10 STAGED_ALLREDUCE_640KB=378.74-393.47us_op 3.33-3.46GBps_eff
Q10 MAPPED_SPIN_ALLREDUCE_2.5MB=879-883us_op 5.94-5.96GBps_eff
Q10 STAGED_ALLREDUCE_2.5MB=1147-1152us_op 4.55-4.57GBps_eff
```

## Findings

1. **No P2P.** `cudaDeviceCanAccessPeer` is 0 both ways on this consumer pair under WDDM;
   `cudaDeviceEnablePeerAccess` fails. Everything crosses the CPU via staged copies. Peer
   kernels writing the other GPU's pointers are therefore unavailable (Q3 skipped).
2. **`cudaMemcpyPeerAsync` works without peer enable** (the driver stages through host
   memory internally) and performs identically to a hand-rolled D2H→pinned→H2D chain:
   ~20 µs for 10 KB, ~120 µs for 512 KB. Effective throughput ~4.3 GB/s per transfer
   (each "transfer" moves the payload twice, D2H then H2D, so ~8.6 GB/s of bus traffic),
   ceiling set by the 4080's PCIe x4 link.
3. **Cross-device events work on streams** (Q4a): an event recorded on one device and
   waited on the other orders work correctly in both directions — this is the only
   working inter-GPU synchronization primitive on this machine.
4. **Cross-device events cannot be captured into CUDA graphs** (Q4b): waiting on an
   externally recorded event fails capture with "dependency created on uncaptured work
   in another stream" in both Global and Relaxed modes; a record-node-in-A /
   wait-node-in-B chain across two graphs fails with "invalid argument". The
   capture-time cross-device record itself succeeds but is useless without the wait side.
   Two-device CUDA-graph lockstep is therefore not expressible on this stack.
5. **128 × 10 KB lockstep allreduce costs ~16 ms eager** (~126 µs/op). Per-op latency
   (two staged hops + two cross-device event round trips, serialized by lockstep)
   dominates: 1.25 MB total payload moves at an effective 0.08 GB/s, ~50× below the
   staged path's bulk rate.
6. **Mapped pinned memory is cross-device visible from kernels (Q6)** — the design lives.
   Both devices map the same host pages at the same virtual address; volatile stores +
   `__threadfence_system()` from a dev0 kernel are observable by dev1 kernel volatile
   loads, 50/50 alternating rounds checksum-exact. No memcpy, no events, no peer access
   needed: the PCIe fabric is the interconnect and the CPU is not on the critical path.
7. **Eager spin allreduce beats staged memcpy 4–6× (Q7/Q10)**: ~19–24 µs/op at 10 KB and
   ~28–37 µs/op at 40 KB (vs ~126–134 µs staged), because one kernel per side per op
   replaces D2H + cross-event + H2D + add with a single full-duplex mapped exchange.
   `__nanosleep(400)` backoff is indistinguishable from tight polling. Throughput
   saturates at ~5.9 GB/s effective (2×payload accounting) from 640 KB up — i.e. the
   4080's PCIe x4 is reachable by kernels alone; staged memcpy peaks at ~4.5 GB/s.
8. **Spin allreduce works inside CUDA graphs but is clock-state-fragile on WDDM (Q8)**:
   correct in both launch orders with zero timeouts, but ~96–118 µs/op cold vs ~19–30 µs
   eager. Root cause is not graph node scheduling (2.6–3.9 µs/node), not mapped writes
   (6.2–6.6 µs/node), and not the handshake (in-kernel waits ~5 µs): on-device cycle
   counts per iteration equal the eager path while wall time is ~5× longer — the
   implied SM clock during graph replay is 0.20–0.65 GHz vs 1.3–2.7 GHz eagerly launched.
   WDDM DVFS holds both GPUs in a low P-state for graph-replayed spin chains; a dense
   compute burn before replay recovers part of it (59–85 µs/op) and, in one run, the full
   eager level (18.7 µs/op). So the graph path itself is sound — its speed just cannot be
   relied upon when submission is bursty; a continuously busy decode engine may hold the
   clocks up, but the eager path does not depend on that assumption.
9. **Graph replay submission is cheap (Q9)** — ~4–11 µs per `cudaGraphLaunch` of a
   10-node graph from two threads — so per-decode-step host submission cost is bounded
   by ~10 µs per graph, not by node count.

## Recommendation

Use eager streams with cross-device events, never graphs, for inter-GPU work on this
machine — but above all, do not run 128 independent 10 KB lockstep allreduces. Since
peer access is unavailable, transport should be `cudaMemcpyPeerAsync` (single call,
driver-staged, same speed as manual pinned staging; keep the manual staged path only if
you need to overlap the D2H/H2D hops of both directions on separate streams). CUDA
graphs must stay strictly per-device: capture each GPU's compute sequence without
cross-device event nodes and synchronize the two devices between replay launches with
eager `cudaEventRecord`/`cudaStreamWaitEvent` (the only verified-correct mechanism).
The decisive optimization is aggregation, not transport: fuse the per-op 10 KB
exchanges into as few bulk transfers as possible — one 1.25 MB exchange costs ~300 µs
(two hops at the measured ~8.6 GB/s bus rate) versus 16 ms for 128 lockstep round
trips, a ~50× reduction; a decode step's communication should be structured as
buffered, batched exchanges around each device-local (graph-replayable) compute
segment, not fine-grained per-operator allreduce.

## Update (Q6–Q10): the transport decision is superseded — use eager mapped-pinned spin

The Q1–Q5 recommendation above assumed staged memcpy was the only transport. Q6–Q10
prove a better one exists and quantify all three candidates for the decode critical
path (128 dependent 40 KB bf16 allreduces per step):

- **(a) eager staged memcpy**: ~130–134 µs/op → **~17 ms per decode step** (Q10).
- **(b) eager mapped-pinned spin** (one kernel per device per op, two host threads):
  ~28–37 µs/op → **~3.8–4.7 ms per decode step** (Q7/Q10). 4–6× faster than (a) at
  10–160 KB and verified exact at every size.
- **(c) in-graph spin with two-thread replay**: works (no deadlock either launch
  order, checksums exact) but ~96–118 µs/op cold; hot clocks give 59–85 µs/op and in
  one run the full eager level (18.7 µs/op) → **~2.4–15 ms per decode step depending
  on DVFS state** — unreliable on this WDDM stack because DVFS holds the GPUs at
  0.2–0.65 GHz during graph-replayed spin chains (Q8 root-cause data). Graphs also
  cannot contain cross-device event nodes anyway (Q4b).

**Design: use eager mapped-pinned spin as the allreduce transport.** Allocate the
exchange buffers with `cudaHostAlloc(Mapped|Portable)`, run one persistent host thread
per device, and exchange via publish → `__threadfence_system()` → sequence flag →
bounded spin → consume inside a single kernel per op. Keep CUDA graphs per-device for
compute only, and launch the communication kernels eagerly between replays (Q9 bounds
the per-launch host cost at ~4–11 µs; graph replay submission itself is not the
bottleneck). Keep every spin bounded (~150 ms ≪ 2 s TDR) with an error word the host
can check. In-graph spin is a legitimate follow-up experiment in a continuously busy
engine (it reached eager-level 18.7 µs/op once with hot clocks), but do not make it
the initial design's critical path on WDDM. For MTP verify batches: batch 4 →
40 KB/allreduce costs ~35 µs (2.2 GB/s effective); the curve is latency-dominated
below ~160 KB, so batching further still helps (640 KB → 5.9 GB/s), but even
unbatched 40 KB ops at ~37 µs put 128 of them at ~4.7 ms — the decode-step
communication budget this transport buys on this machine.

## Files

- `tp_probe.cu` — the probe (single file, ~1245 lines; Q6–Q10 additions ~630 lines incl.
  diagnostics: wait/cadence instrumentation, DVFS hot-clock control, graph node-cost diags)
- `tp_probe.exe` — built with the command above
- `run_q6_q10.log` — full output of the final Q6–Q10 run (0 CUDA errors, exit 0)

## Update (Q13, M4a): StreamWait gates capture into CUDA graphs — the M4a transport

`waitvalue_graph_probe.cu` (built with the same nvcc command) tests the M4a design
end-to-end: per-rank CUDA graphs containing a full 130-exchange round of
publish kernel → `cuStreamWaitValue64(peer release, code, GEQ)` → consume kernel, replayed
from two host threads with pinned round tags.

```
Q13 WAITVALUE_IN_GRAPHS=WORKS
Q13 DETAIL: cudaStreamBeginCapture(ThreadLocal) accepts cuStreamWaitValue64 on the
       capturing stream; EndCapture/instantiate/upload all clean
Q13 REPLAY_REARM=WORKS: the consume kernel resets the gated peer word to 0 after its
       payload loads; 100 replays x 130 exchanges, exact 130*tag sums on both ranks every
       replay with NO host transport reset between replays, words back at 0 after each round
Q13 DETAIL: codes are baked per node at capture (unique per exchange via the per-side seq
       counter); the reset discipline makes every exchange self-contained, so a gate can
       only pass on the current exchange's publish even though codes repeat across replays
Q13 ROUND_CADENCE: eager round 10.75 ms vs graph round 8.35 ms (130 exchanges + 6 light
       compute kernels; the graph removes the per-op submission cost only — the ~64 us
       exchange RTT remains, as in Q8/Q12)
```

Consequence for the engine: the whole TP decode round — TpLink publishes, channel gates,
consumes, and all compute — captures into one graph per rank per frontier profile, and
replay is one launch per rank per round (measured enqueue 57 ms -> 0.9 ms). The wall
clock barely moved, because the device round itself is ~65-70 ms: per-exchange gate
transitions after real compute cost ~0.5 ms each on WDDM (probe: 64 us/exchange with no
interleaved compute vs ~630 us with one kernel between every exchange), amplified by DVFS
parking whichever rank is gate-idle (RTX 4080 at 495 MHz / 21% util while its peer held
1980 MHz / 99%). The engine ships a per-rank clock holder (core/tp_link.cu,
`tp_clock_holder_pulse`, pulsed by a TpProgram host thread): 200-token steady state
14.5 -> 16.4 tok/s. Measured constraints: a never-ending holder kernel deadlocks the WDDM
queue, and a separate-process warmer time-slices the GPU instead (71 -> 106 ms).

- `waitvalue_graph_probe.cu` / `.exe` — the Q13 probe
- `clock_warmer.cu` / `.exe` — the separate-process warmer diagnostic (kept as the
  negative result: cross-process WDDM time-slicing)

## Update (M4c): engine-shape graph A/B, all-blocks spin, clock-holder footprint

`tp_engine_repro.cu` gained `--graph` (capture each side's whole mirrored round into one
CUDA graph and replay it from two host threads — the engine's decode execution shape),
`--spin-mode 0|1` (0 = block-0 spin + device-local go broadcast, 1 = every block spins on
the peer release directly), and `--gemm-k/--gemm-blocks` GEMM-stand-in calibration knobs.
Two probe defects were fixed: the GEMM stand-in's 2048-thread block was an invalid launch
config that failed silently on every earlier run (now 1024 threads × 2 z-halves), and
`cudaSetDevice` inside a ThreadLocal capture invalidated the capture (now guarded; the two
sides capture sequentially — capture does not execute, so no rendezvous is needed).

```
engine shape, 130 exchanges + GDN/attn compute per round, RTX 4080 + RTX 3080:
eager relay (spin-mode 0)      40.0 ms/round
eager all-blocks (spin-mode 1) 42.1 ms/round   (+17 us/exchange: 20 ld.global.cv pollers
                                                congest the 4080's PCIe x4 payload path)
graph relay (spin-mode 0)      32.5 ms/round   (250 us/exchange-slot incl. compute stand-ins)
graph all-blocks (spin-mode 1) 34.7 ms/round   (same regression under replay)
graph, exchanges only          4.9 ms/round    (38 us/exchange: the graph-node floor)
```

Conclusions: (1) the go-broadcast relay is the right shape — replacing it with direct
per-block polling of the peer flag costs more in PCIe read traffic than the relay's second
dependent wait chain saves, so `core/tp_link.cuh` keeps the relay; (2) under graph replay
at engine shape the per-exchange floor is ~38 us with no compute, and the engine's own
small compute kernels account for the rest of the per-exchange slot; (3) during real MTP3
decode both ranks park at 210 MHz (nvidia-smi) even with the clock holder running — a
1-block pulse is too small an SM-active signal to hold the governor, but widening the
pulse footprint (4-34 blocks) was measured end to end and REJECTED: 4-16 blocks cost more
than they recover and 34 blocks (half the smaller rank's SMs) stalls rounds outright
through contention with the residency-sensitive exchange/GDN kernels. External clock
locking (`nvidia-smi -lgc`) needs admin rights this account does not have, so the parked-
clock penalty remains open; measured end-to-end decode on the 200-token MTP3 gate swings
between ~30 and ~80 tok/s purely with ambient WDDM load (compositor, browser, chat
clients), which dominates all finer-grained tuning differences.
