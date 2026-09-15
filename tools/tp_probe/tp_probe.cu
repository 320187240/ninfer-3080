// tp_probe.cu — single-process 2-GPU communication probe (RTX 4080 sm_89 + RTX 3080 sm_86,
// Windows WDDM) measuring facts needed to design a tensor-parallel inference engine.
//
// Build (compute_86 PTX is embedded and JITs on sm_89):
//   nvcc -O2 -std=c++17 -gencode arch=compute_86,code=compute_86 \
//        --compiler-bindir <MSVC>/bin/Hostx64/x64 -o tp_probe.exe tp_probe.cu
//
// Q1  cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess, both directions
// Q2  transfer latency+bandwidth, n=5120 bf16 (10KB) and n=262144 bf16 (512KB):
//     peer path (cudaMemcpyPeerAsync) vs staged path (device -> pinned host -> device)
// Q3  kernel on one GPU writing directly into the other GPU's pointer (peer-mapped writes)
// Q4a cross-device events without capture (write-then-read, checksum verified)
// Q4b cross-device event wait nodes inside CUDA graphs + capture-time cross-device record
// Q5  10KB bf16 allreduce x128 (decode step): eager streams vs two device graphs
// Q6  cudaHostAlloc(Mapped|Portable) visibility: dev0 kernel writes via its mapped pointer,
//     dev1 kernel reads via ITS mapped pointer (both map the same pinned host pages)
// Q7  spin-handshake allreduce over the mapped buffer, eager, two host threads, 128 dependent
//     ops; pure spin poll and __nanosleep backoff variants
// Q8  the same spin allreduce captured into per-device CUDA graphs (kernels only, device-side
//     flags), two-thread concurrent replay, both launch orders, watchdog + bounded spins
// Q9  cudaGraphLaunch submission cost of a ~10-node graph from two threads
// Q10 allreduce bandwidth curve 10KB..2.5MB: mapped-spin vs staged memcpy
//
// NOTE: CUDA events are bound to the device current at creation; recording an event on the
// other device's stream fails with "invalid resource handle". Events recorded on s0 are
// created with dev0 current (e0*/ge0[]), events recorded on s1 with dev1 current (e1*/ge1[]).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>

using bf16 = __nv_bfloat16;
using ull  = unsigned long long;

static std::atomic<int> g_nerr{0};
static void report_err(const char* what, cudaError_t e, int line){
  g_nerr++;
  if(g_nerr <= 25) printf("    [CUDA ERR] %s -> %s (tp_probe.cu:%d)\n", what, cudaGetErrorString(e), line);
}
#define CK(x)  do{ cudaError_t _e=(x); if(_e!=cudaSuccess) report_err(#x,_e,__LINE__); }while(0)
#define CKF(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  printf("    [CUDA FATAL] %s -> %s (tp_probe.cu:%d)\n", #x, cudaGetErrorString(_e), __LINE__); exit(1);} }while(0)

static const int  N_SML = 5120;        // 10 KB of bf16
static const int  N_BIG = 262144;      // 512 KB of bf16
static const int  AOPS  = 128;         // allreduces per decode step
static const long long SPIN = 300000;  // ~130us clock64() spin, amplifies a missing sync
static const int  GRS   = (N_SML+255)/256;

// ---------------- kernels ----------------
__global__ void k_fill(bf16* p, int n, float v){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) p[i] = __float2bfloat16(v);
}
__global__ void k_fill_delay(bf16* p, int n, float v, long long spin){
  long long t0 = clock64();
  while(clock64() - t0 < spin) {}
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) p[i] = __float2bfloat16(v);
}
__global__ void k_add(bf16* dst, const bf16* src, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) dst[i] = __hadd(dst[i], src[i]);
}
__global__ void k_sum(const bf16* p, int n, float* out){
  __shared__ float sm[256];
  float acc = 0.f;
  for(int i = threadIdx.x; i < n; i += blockDim.x) acc += __bfloat162float(p[i]);
  sm[threadIdx.x] = acc; __syncthreads();
  for(int s = 128; s > 0; s >>= 1){ if(threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x == 0) *out = sm[0];
}

// ---------------- Q6-Q10: mapped-pinned zero-copy spin transport ----------------
static const int    NMAX     = 1310720;             // 2.5 MB of bf16: largest Q10 message
static const ull    SPINWAIT = 300000000ULL;        // ~150 ms clock64 budget per spin (<< WDDM TDR)
static const size_t OFF_FLAGS = (size_t)NMAX*2*4;   // flag[side][par] (4 x ull), then err[side] (2 x ull)
static const size_t OFF_ERR   = OFF_FLAGS + 32;
static const size_t MAP_BYTES = OFF_ERR + 64;
static size_t pay_off(int side,int par){ return (size_t)(side*2+par)*(size_t)NMAX*2; }
static int grdsz(int n){ return (n+255)/256; }
static int grid_spin(int n){ int g=(n/2+255)/256; if(g>64)g=64; if(g<1)g=1; return g; }  // <=68 SMs, fully resident

__global__ void k_write_mapped(volatile bf16* p, int n, float v){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n){ p[i] = __float2bfloat16(v); __threadfence_system(); }
}
__global__ void k_sum_mapped(volatile const bf16* p, int n, float* out){
  __shared__ float sm[256];
  float acc = 0.f;
  for(int i = threadIdx.x; i < n; i += blockDim.x) acc += __bfloat162float(p[i]);
  sm[threadIdx.x] = acc; __syncthreads();
  for(int s = 128; s > 0; s >>= 1){ if(threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x == 0) *out = sm[0];
}
__global__ void k_fillf(float* p, int n, float v){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) p[i] = v;
}
__global__ void k_burn(float* out, long long iters){   // dense FMA load to pull DVFS clocks up
  float a = threadIdx.x*1e-6f + 1.f, b = 1.0000001f;
  for(long long i=0;i<iters;i++) a = fmaf(a,b,0.1f);
  if(a < 0.f) *out = a;                                // never taken; defeats dead-code elimination
}
__global__ void k_sumf(const float* p, int n, float* out){
  __shared__ float sm[256];
  float acc = 0.f;
  for(int i = threadIdx.x; i < n; i += blockDim.x) acc += p[i];
  sm[threadIdx.x] = acc; __syncthreads();
  for(int s = 128; s > 0; s >>= 1){ if(threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x == 0) *out = sm[0];
}

// One spin-allreduce step entirely over mapped pinned host memory: no cudaMemcpy, no events.
// Publish local -> my mapped payload (volatile stores, parity double-buffered); grid-wide
// arrival counter releases my flag (seq per iteration); block 0 polls the peer flag (bounded
// spin) and broadcasts a device-side go flag; all blocks then consume the peer payload.
// MODE 0: acc[i] += peer[i] (constant publish, fp32 accumulate) -- timing shape, execution
//         dependency chain identical to a dependent allreduce sequence.
// MODE 1: local[i] = local[i] + peer[i] (true bf16 feedback allreduce) -- exact verify shape.
template<int MODE, bool BACKOFF>
__global__ void k_spin_step(bf16* local, float* acc, int n,
                            volatile bf16* myPayBase, volatile bf16* otPayBase,
                            volatile ull* myFlagBase, volatile ull* otFlagBase,
                            ull seq, int par, volatile ull* goFlag, unsigned* cnt,
                            ull waitLimit, volatile ull* err, ull* dbg)
{
  if(dbg && blockIdx.x==0 && threadIdx.x==0) dbg[AOPS+1+(int)seq] = clock64();   // start stamp
  const int nw = n >> 1;                                   // n is even: 32-bit words = 2 bf16 each
  const unsigned int* srcL = (const unsigned int*)local;
  volatile unsigned int* myV = (volatile unsigned int*)(myPayBase + (size_t)par*NMAX);
  for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < nw; i += gridDim.x*blockDim.x) myV[i] = srcL[i];
  __threadfence_system();
  __syncthreads();
  if(threadIdx.x == 0){
    unsigned a = atomicAdd(&cnt[par], 1u);
    if((a + 1u) % gridDim.x == 0u) myFlagBase[par] = seq;  // last arriving block releases
  }
  if(blockIdx.x == 0){
    if(threadIdx.x == 0){
      ull t0 = clock64();
      while(otFlagBase[par] < seq){
        if((ull)(clock64() - t0) > waitLimit){ *err = seq; break; }
        if(BACKOFF) __nanosleep(400);
      }
      if(dbg) dbg[seq] = clock64() - t0;                // peer-flag wait duration (diagnostic)
      __threadfence_system();
      goFlag[par] = seq;                                   // also unblocks after a timeout
    }
    __syncthreads();
  }
  if(threadIdx.x == 0){
    ull t0 = clock64();
    while(goFlag[par] < seq){
      if((ull)(clock64() - t0) > waitLimit){ *err = seq + 1000000ULL; break; }
    }
  }
  __syncthreads();
  volatile unsigned int* otV = (volatile unsigned int*)(otPayBase + (size_t)par*NMAX);
  for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < nw; i += gridDim.x*blockDim.x){
    unsigned int v = otV[i];
    const __nv_bfloat162 h = *(const __nv_bfloat162*)&v;
    if(MODE == 1){
      bf16* d = local + i*2;
      d[0]=__hadd(d[0],h.x); d[1]=__hadd(d[1],h.y);
    } else {
      float* a = acc + i*2;
      a[0]+=__bfloat162float(h.x); a[1]+=__bfloat162float(h.y);
    }
  }
}

// ---------------- helpers ----------------
static float dev_sum(bf16* d, int n, float* scratch, cudaStream_t s){
  float zero = 0.f, r = -1.f;
  CK(cudaMemcpyAsync(scratch, &zero, 4, cudaMemcpyHostToDevice, s));
  k_sum<<<1,256,0,s>>>(d, n, scratch); CK(cudaGetLastError());
  CK(cudaMemcpyAsync(&r, scratch, 4, cudaMemcpyDeviceToHost, s));
  CK(cudaStreamSynchronize(s));
  return r;
}
static void sync2(int d0, cudaStream_t a, int d1, cudaStream_t b){
  cudaSetDevice(d0); CK(cudaStreamSynchronize(a));
  cudaSetDevice(d1); CK(cudaStreamSynchronize(b));
}
template<class F> static float bench_us(F fn, int dev, cudaStream_t ts, int iters){
  cudaSetDevice(dev);
  cudaEvent_t a,b; CKF(cudaEventCreate(&a)); CKF(cudaEventCreate(&b));
  CKF(cudaEventRecord(a, ts));
  for(int i=0;i<iters;i++) fn(i);
  CKF(cudaEventRecord(b, ts));
  CKF(cudaEventSynchronize(b));
  float ms=-1; CKF(cudaEventElapsedTime(&ms,a,b));
  CKF(cudaEventDestroy(a)); CKF(cudaEventDestroy(b));
  return ms*1000.f/iters;   // us per iteration
}
template<class F> static float bench_min3(F fn, int dev, cudaStream_t ts, int warm, int iters){
  for(int i=0;i<warm;i++) fn(i);
  CK(cudaStreamSynchronize(ts));
  float best=1e30f;
  for(int r=0;r<3;r++){ float us=bench_us(fn,dev,ts,iters); if(us<best)best=us; CK(cudaStreamSynchronize(ts)); }
  return best;
}
static float gbps(size_t bytes, float us){ return bytes/(us*1000.0); }

// capture a graph on sCap whose first node waits ev (recorded on sRec's device), then H2D dst<-hObs.
// verify: producer fills aProd (delayed), copies to hObs, records ev (not synced), replay, checksum dst.
// returns 0=works, 1=capture/instantiate failure (err set), 2=capture ok but replay ordering wrong.
static int graph_wait_test(cudaStream_t sCap, cudaStream_t sRec, int capDev, int recDev,
                           bf16* aProd, bf16* dst, float* scratch, char* hObs, cudaEvent_t ev,
                           cudaStreamCaptureMode mode, char* err, size_t errlen){
  const size_t bytes = (size_t)N_SML*2;
  sync2(capDev, sCap, recDev, sRec);
  cudaSetDevice(capDev);
  cudaGraph_t g=NULL; cudaGraphExec_t gx=NULL;
  cudaError_t cb = cudaStreamBeginCapture(sCap, mode);
  if(cb!=cudaSuccess){ snprintf(err,errlen,"BeginCapture: %s", cudaGetErrorString(cb)); return 1; }
  cudaError_t e1 = cudaStreamWaitEvent(sCap, ev, 0);
  cudaError_t e2 = (e1==cudaSuccess) ? cudaMemcpyAsync(dst, hObs, bytes, cudaMemcpyHostToDevice, sCap) : cudaSuccess;
  cudaError_t e3 = cudaStreamEndCapture(sCap, &g);
  if(e1||e2||e3){
    snprintf(err,errlen,"capture: wait=%s memcpy=%s end=%s",
             e1?cudaGetErrorString(e1):"ok", e2?cudaGetErrorString(e2):"ok", e3?cudaGetErrorString(e3):"ok");
    cudaGetLastError();
    if(g) cudaGraphDestroy(g);
    return 1;
  }
  cudaError_t ei = cudaGraphInstantiate(&gx, g, 0);
  if(ei){ snprintf(err,errlen,"instantiate: %s", cudaGetErrorString(ei)); cudaGraphDestroy(g); return 1; }
  bool ok=true; int shown=0;
  for(int rep=0; rep<50; rep++){
    float v=(rep&1)?3.f:5.f;
    cudaSetDevice(recDev);
    k_fill_delay<<<GRS,256,0,sRec>>>(aProd,N_SML,v,SPIN); CK(cudaGetLastError());
    CK(cudaMemcpyAsync(hObs, aProd, bytes, cudaMemcpyDeviceToHost, sRec));
    CK(cudaEventRecord(ev, sRec));            // deliberately NOT synced before replay
    cudaSetDevice(capDev);
    CK(cudaGraphLaunch(gx, sCap));
    CK(cudaStreamSynchronize(sCap));
    float s = dev_sum(dst, N_SML, scratch, sCap);
    if(fabsf(s - v*N_SML) > 1.f){ ok=false; if(shown<3){ printf("    rep %d: graph saw %.1f expected %.1f\n", rep, s, v*N_SML); shown++; } }
  }
  cudaGraphExecDestroy(gx); cudaGraphDestroy(g);
  return ok?0:2;
}

// summary accumulators
static float g_peer_us[2][2]  = {{-1,-1},{-1,-1}};  // [size][dir: 0->1, 1->0]
static float g_stg_us[2][2]   = {{-1,-1},{-1,-1}};
static size_t g_bytes[2] = {(size_t)N_SML*2, (size_t)N_BIG*2};
static float g_pp_us = -1, g_eager_stg_us = -1, g_eager_peer_us = -1, g_graph_us = -1;
static const char* g_sztag[2] = {"10KB","512KB"};

// Q6-Q10 shared state (file scope: worker threads and graph builders need them).
// e0x/p0e are dev0-recorded, e1x/p1e dev1-recorded (events bind to the current device).
static cudaStream_t s0=NULL, s1=NULL;               // sX lives on device X
static bf16 *a0=NULL,*b0=NULL,*a1=NULL,*b1=NULL; static float *sc0=NULL,*sc1=NULL;
static cudaEvent_t e0x=NULL,e1x=NULL,p0e=NULL,p1e=NULL;
static char *hMap=NULL, *m0=NULL, *m1=NULL;         // mapped pinned host buffer + per-device device pointers
static float *acc0=NULL,*acc1=NULL;                 // fp32 allreduce accumulators (NMAX)
static ull *go0=NULL,*go1=NULL;                     // device side: goFlag[2] + arrival cnt[2] per side
static ull *dbg0=NULL,*dbg1=NULL;                   // device side: per-iteration peer-flag wait (clocks)
static bf16 *qs0=NULL,*qs1=NULL,*rs0=NULL,*rs1=NULL; static char *hqs0=NULL,*hqs1=NULL;  // Q10 staged bufs
static bool  g_map_ok=false;
static float g_q7_us[2][2]    = {{-1,-1},{-1,-1}};  // [10KB,40KB][spin, nanosleep]
static float g_q8_us[2]       = {-1,-1};            // [10KB,40KB] in-graph us/op
static float g_q8_ms[2]       = {-1,-1};
static bool  g_q8_ok[2]       = {false,false};
static float g_q8_node_us     = -1;                 // diag: per-node cost, 128 tiny kernels/graph
static float g_q8_mnode_us    = -1;                 // diag: per-node cost when kernels write mapped host mem
static float g_q8_hot_us[2]   = {-1,-1};            // in-graph us/op with a compute burn right before replay
static float g_q9_us[2]       = {-1,-1};
static float g_q10_spin_us[5] = {-1,-1,-1,-1,-1}, g_q10_stg_us[5] = {-1,-1,-1,-1,-1};

static double now_us(){ using namespace std::chrono;
  return duration_cast<duration<double,std::micro>>(steady_clock::now().time_since_epoch()).count(); }

static float dev_sumf(float* d, int n, float* scratch, cudaStream_t s){
  float zero=0.f, r=-1.f;
  CK(cudaMemcpyAsync(scratch,&zero,4,cudaMemcpyHostToDevice,s));
  k_sumf<<<1,256,0,s>>>(d,n,scratch); CK(cudaGetLastError());
  CK(cudaMemcpyAsync(&r,scratch,4,cudaMemcpyDeviceToHost,s));
  CK(cudaStreamSynchronize(s));
  return r;
}
static float dev_sum_mapped(volatile bf16* d, int n, float* scratch, cudaStream_t s){
  float zero=0.f, r=-1.f;
  CK(cudaMemcpyAsync(scratch,&zero,4,cudaMemcpyHostToDevice,s));
  k_sum_mapped<<<1,256,0,s>>>(d,n,scratch); CK(cudaGetLastError());
  CK(cudaMemcpyAsync(&r,scratch,4,cudaMemcpyDeviceToHost,s));
  CK(cudaStreamSynchronize(s));
  return r;
}

// report avg/max peer-flag spin-wait durations (clock64 cycles) recorded by the last round.
// us_per_op (wall) lets us derive the implied SM clock from the kernel start-to-start cadence.
static void dump_dbg(const char* tag, int iters, float us_per_op){
  static ull h0[2*AOPS+4], h1[2*AOPS+4];
  cudaSetDevice(0); CK(cudaMemcpy(h0, dbg0, sizeof(ull)*(AOPS+2+iters+1), cudaMemcpyDeviceToHost));
  cudaSetDevice(1); CK(cudaMemcpy(h1, dbg1, sizeof(ull)*(AOPS+2+iters+1), cudaMemcpyDeviceToHost));
  ull mx0=0,mx1=0; double s0=0,s1=0,g0=0,g1=0;
  for(int i=1;i<=iters;i++){ s0+=(double)h0[i]; s1+=(double)h1[i]; if(h0[i]>mx0)mx0=h0[i]; if(h1[i]>mx1)mx1=h1[i]; }
  for(int i=2;i<=iters;i++){ g0+=(double)(h0[AOPS+1+i]-h0[AOPS+1+i-1]); g1+=(double)(h1[AOPS+1+i]-h1[AOPS+1+i-1]); }
  double c0=g0/(iters-1)/us_per_op/1000.0, c1=g1/(iters-1)/us_per_op/1000.0;   // cycles/us -> GHz
  printf("    %s: peer-flag wait avg %.3f/%.3f Mcycles (max %.3f/%.3f) ; start-to-start %.3f/%.3f Mcycles -> implied SM clock %.2f/%.2f GHz\n",
         tag, s0/iters/1e6, s1/iters/1e6, mx0/1e6, mx1/1e6, g0/(iters-1)/1e6, g1/(iters-1)/1e6, c0, c1);
}

// reset mapped flags / err words, device go+counters, local fills (1,2) and accumulators
static void spin_reset(int n){
  sync2(0,s0,1,s1);
  memset((char*)hMap+OFF_FLAGS, 0, 64);
  cudaSetDevice(0); CK(cudaMemsetAsync(go0, 0, 24, s0));
  k_fill<<<grdsz(n),256,0,s0>>>(qs0,n,1.f); CK(cudaGetLastError());
  k_fillf<<<grdsz(n),256,0,s0>>>(acc0,n,0.f); CK(cudaGetLastError());
  cudaSetDevice(1); CK(cudaMemsetAsync(go1, 0, 24, s1));
  k_fill<<<grdsz(n),256,0,s1>>>(qs1,n,2.f); CK(cudaGetLastError());
  k_fillf<<<grdsz(n),256,0,s1>>>(acc1,n,0.f); CK(cudaGetLastError());
  sync2(0,s0,1,s1);
}

// check mapped err words (bounded-spin timeouts) + checksums; print details on failure or verbose
static bool spin_check(int n, int iters, int mode, bool verbose){
  ull w0=*(const ull*)(hMap+OFF_ERR), w1=*(const ull*)(hMap+OFF_ERR+8);
  bool ok = (!w0 && !w1);
  if(w0||w1) printf("    [SPIN TIMEOUT] err word dev0=%llu dev1=%llu (nonzero = bounded spin gave up)\n",w0,w1);
  if(mode==1){
    cudaSetDevice(0); float v0=dev_sum(qs0,n,sc0,s0);
    cudaSetDevice(1); float v1=dev_sum(qs1,n,sc1,s1);
    float exp=384.f*(float)n;                       // seeds 1,2 -> 3*2^(iters-1), iters==8
    bool okv = fabsf(v0-exp)<=0.01f*exp && fabsf(v1-exp)<=0.01f*exp;
    if(verbose||!okv) printf("    verify %d bf16-feedback allreduces: sum0=%.1f sum1=%.1f expected=%.1f -> %s\n",
                             iters,v0,v1,exp,okv?"OK":"FAIL");
    return ok&&okv;
  }
  cudaSetDevice(0); float v0=dev_sumf(acc0,n,sc0,s0);
  cudaSetDevice(1); float v1=dev_sumf(acc1,n,sc1,s1);
  float e0v=2.f*iters*(float)n, e1v=1.f*iters*(float)n;
  bool okv = fabsf(v0-e0v)<=2e-3f*e0v && fabsf(v1-e1v)<=2e-3f*e1v;
  if(verbose||!okv) printf("    verify %d constant allreduces: acc0=%.1f (exp %.0f) acc1=%.1f (exp %.0f) -> %s\n",
                           iters,v0,e0v,v1,e1v,okv?"OK":"FAIL");
  return ok&&okv;
}

// one device side of the eager spin allreduce: its own stream, its own host thread
static void spin_worker(int dev, cudaStream_t s, bf16* local, float* acc, int n, int iters,
                        const char* map, int side, int mode, bool backoff)
{
  cudaSetDevice(dev);
  volatile bf16* myPay=(volatile bf16*)(map+pay_off(side,0));
  volatile bf16* otPay=(volatile bf16*)(map+pay_off(1-side,0));
  volatile ull* myFlag=(volatile ull*)(map+OFF_FLAGS)+side*2;
  volatile ull* otFlag=(volatile ull*)(map+OFF_FLAGS)+(1-side)*2;
  volatile ull* err=(volatile ull*)(map+OFF_ERR)+side;
  ull* go = side? go1 : go0; unsigned* cnt = side? (unsigned*)(go1+2) : (unsigned*)(go0+2);
  ull* dbg = side? dbg1 : dbg0;
  int g = grid_spin(n);
  for(int i=0;i<iters;i++){
    ull seq=(ull)(i+1); int par=i&1;
    if(mode==0 && !backoff)     k_spin_step<0,false><<<g,256,0,s>>>(local,acc,n,myPay,otPay,myFlag,otFlag,seq,par,go,cnt,SPINWAIT,err,dbg);
    else if(mode==0)            k_spin_step<0,true ><<<g,256,0,s>>>(local,acc,n,myPay,otPay,myFlag,otFlag,seq,par,go,cnt,SPINWAIT,err,dbg);
    else                        k_spin_step<1,false><<<g,256,0,s>>>(local,acc,n,myPay,otPay,myFlag,otFlag,seq,par,go,cnt,SPINWAIT,err,dbg);
    CK(cudaGetLastError());
  }
  CK(cudaStreamSynchronize(s));
}

// reset, run both sides on two host threads, wall-clock us/op, verify
static float spin_ar_run(int n, int iters, int mode, bool backoff, bool verbose){
  spin_reset(n);
  double t0=now_us();
  std::thread ta(spin_worker,0,s0,qs0,acc0,n,iters,(const char*)m0,0,mode,backoff);
  std::thread tb(spin_worker,1,s1,qs1,acc1,n,iters,(const char*)m1,1,mode,backoff);
  ta.join(); tb.join();
  float us=(float)((now_us()-t0)/(double)iters);
  spin_check(n,iters,mode,verbose);
  return us;
}
static float spin_ar_bench(int n, int iters, bool backoff){    // warmup + min of 3 + one checksum round
  spin_ar_run(n,iters,0,backoff,false);
  float best=1e30f;
  for(int r=0;r<3;r++){ float us=spin_ar_run(n,iters,0,backoff,false); if(us<best)best=us; }
  spin_ar_run(n,iters,0,backoff,true);
  dump_dbg("eager spin-wait (last round)", iters, best);
  return best;
}

// capture iters chained spin-allreduce steps as one per-device graph (kernel nodes only)
static cudaGraphExec_t build_spin_graph(int dev, cudaStream_t s, int side, const char* map,
                                        bf16* local, float* acc, int n, int iters, int mode,
                                        bool backoff, char* errb, size_t el)
{
  cudaSetDevice(dev);
  cudaGraph_t g=NULL; cudaGraphExec_t x=NULL;
  cudaError_t cb = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  if(cb!=cudaSuccess){ snprintf(errb,el,"BeginCapture: %s",cudaGetErrorString(cb)); return NULL; }
  volatile bf16* myPay=(volatile bf16*)(map+pay_off(side,0));
  volatile bf16* otPay=(volatile bf16*)(map+pay_off(1-side,0));
  volatile ull* myFlag=(volatile ull*)(map+OFF_FLAGS)+side*2;
  volatile ull* otFlag=(volatile ull*)(map+OFF_FLAGS)+(1-side)*2;
  volatile ull* errp=(volatile ull*)(map+OFF_ERR)+side;
  ull* go = side? go1 : go0; unsigned* cnt = side? (unsigned*)(go1+2) : (unsigned*)(go0+2);
  ull* dbgp = side? dbg1 : dbg0;
  int gd = grid_spin(n);
  cudaError_t t = cudaSuccess;
  for(int i=0;i<iters && t==cudaSuccess;i++){
    ull seq=(ull)(i+1); int par=i&1;
    if(mode==0 && !backoff) k_spin_step<0,false><<<gd,256,0,s>>>(local,acc,n,myPay,otPay,myFlag,otFlag,seq,par,go,cnt,SPINWAIT,errp,dbgp);
    else if(mode==0)        k_spin_step<0,true ><<<gd,256,0,s>>>(local,acc,n,myPay,otPay,myFlag,otFlag,seq,par,go,cnt,SPINWAIT,errp,dbgp);
    else                    k_spin_step<1,false><<<gd,256,0,s>>>(local,acc,n,myPay,otPay,myFlag,otFlag,seq,par,go,cnt,SPINWAIT,errp,dbgp);
    t = cudaGetLastError();
  }
  cudaError_t ee = cudaStreamEndCapture(s,&g);
  if(t!=cudaSuccess||ee!=cudaSuccess||!g){
    snprintf(errb,el,"capture body=%s end=%s",cudaGetErrorString(t),cudaGetErrorString(ee));
    cudaGetLastError(); if(g) cudaGraphDestroy(g); return NULL;
  }
  cudaError_t ei = cudaGraphInstantiate(&x,g,0);
  if(ei){ snprintf(errb,el,"instantiate: %s",cudaGetErrorString(ei)); cudaGraphDestroy(g); return NULL; }
  cudaGraphDestroy(g);
  return x;
}

struct Gate { std::atomic<bool> launched{false}; std::atomic<bool> done{false}; };
static void replay_worker(int dev, cudaGraphExec_t x, cudaStream_t s, const Gate* waitOn, Gate* self){
  cudaSetDevice(dev);
  if(waitOn) while(!waitOn->launched.load(std::memory_order_acquire)) std::this_thread::yield();
  cudaError_t e = cudaGraphLaunch(x, s);
  if(e!=cudaSuccess) report_err("Q8 cudaGraphLaunch",e,__LINE__);
  if(self) self->launched.store(true,std::memory_order_release);
  e = cudaStreamSynchronize(s);
  if(e!=cudaSuccess) report_err("Q8 cudaStreamSynchronize",e,__LINE__);
  if(self) self->done.store(true,std::memory_order_release);
}
static bool wait_gates(Gate& a, Gate& b, double sec){       // wall-clock watchdog
  auto t0 = std::chrono::steady_clock::now();
  while(!a.done.load(std::memory_order_acquire) || !b.done.load(std::memory_order_acquire)){
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() > sec) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return true;
}
// replay the graph pair concurrently from two threads; returns us/op, negative on watchdog timeout
static double graph_pair_run(cudaGraphExec_t xA, cudaGraphExec_t xB, bool bFirst,
                             int n, int iters, int mode, bool verbose){
  spin_reset(n);
  Gate gA, gB;
  double t0 = now_us();
  std::thread ta, tb;
  if(bFirst){ tb = std::thread(replay_worker,1,xB,s1,nullptr,&gB);
              ta = std::thread(replay_worker,0,xA,s0,(const Gate*)&gB,&gA); }
  else      { ta = std::thread(replay_worker,0,xA,s0,nullptr,&gA);
              tb = std::thread(replay_worker,1,xB,s1,(const Gate*)&gA,&gB); }
  bool to = !wait_gates(gA,gB,10.0);
  ta.join(); tb.join();
  double us = (now_us()-t0)/(double)iters;
  if(to) printf("    !! watchdog: replays still not idle after 10 s\n");
  spin_check(n,iters,mode,verbose);
  return to ? -us : us;
}

// Q5-style eager staged lockstep allreduce at arbitrary n (32 iters for the Q10 curve)
static float staged_ar_us(int n, int iters){
  size_t bytes=(size_t)n*2; int g=grdsz(n);
  auto reset=[&]{
    sync2(0,s0,1,s1);
    cudaSetDevice(0); k_fill<<<g,256,0,s0>>>(qs0,n,1.f); CK(cudaGetLastError());
    CK(cudaMemcpyAsync(hqs0,qs0,bytes,cudaMemcpyDeviceToHost,s0)); CK(cudaEventRecord(p0e,s0));
    cudaSetDevice(1); k_fill<<<g,256,0,s1>>>(qs1,n,2.f); CK(cudaGetLastError());
    CK(cudaMemcpyAsync(hqs1,qs1,bytes,cudaMemcpyDeviceToHost,s1)); CK(cudaEventRecord(p1e,s1));
    sync2(0,s0,1,s1);
  };
  auto op=[&](int){
    cudaSetDevice(0);
    CK(cudaStreamWaitEvent(s0,p1e,0));
    CK(cudaMemcpyAsync(rs0,hqs1,bytes,cudaMemcpyHostToDevice,s0));
    k_add<<<g,256,0,s0>>>(qs0,rs0,n); CK(cudaGetLastError());
    CK(cudaMemcpyAsync(hqs0,qs0,bytes,cudaMemcpyDeviceToHost,s0));
    CK(cudaEventRecord(p0e,s0));
    cudaSetDevice(1);
    CK(cudaStreamWaitEvent(s1,p0e,0));
    CK(cudaMemcpyAsync(rs1,hqs0,bytes,cudaMemcpyHostToDevice,s1));
    k_add<<<g,256,0,s1>>>(qs1,rs1,n); CK(cudaGetLastError());
    CK(cudaMemcpyAsync(hqs1,qs1,bytes,cudaMemcpyDeviceToHost,s1));
    CK(cudaEventRecord(p1e,s1));
  };
  reset();   // verify: 8 lockstep ops; the Q5 op shape is a skew-add (a<-a+b then b<-a'+b),
  for(int i=0;i<8;i++) op(i);   // so seeds (1,2) grow Fibonacci-style: a=F(2k+4), b=F(2k+5)
  sync2(0,s0,1,s1);
  cudaSetDevice(0); float v0=dev_sum(qs0,n,sc0,s0);
  cudaSetDevice(1); float v1=dev_sum(qs1,n,sc1,s1);
  float ea=1.f, eb=2.f;
  for(int i=0;i<8;i++){ ea=ea+eb; eb=ea+eb; }
  float e0=ea*(float)n, e1=eb*(float)n;
  printf("    staged verify (8 lockstep ops): sum0=%.1f (exp %.0f) sum1=%.1f (exp %.0f) -> %s\n",
         v0,e0,v1,e1,(fabsf(v0-e0)<=0.02f*e0 && fabsf(v1-e1)<=0.02f*e1)?"OK":"FAIL");
  float best=1e30f;
  for(int r=0;r<4;r++){                                     // 1 warm + 3 timed
    reset();
    cudaSetDevice(0);
    cudaEvent_t ta,tb; CKF(cudaEventCreate(&ta)); CKF(cudaEventCreate(&tb));
    CKF(cudaEventRecord(ta,s0));
    for(int i=0;i<iters;i++) op(i);
    CK(cudaStreamWaitEvent(s0,p1e,0));
    CKF(cudaEventRecord(tb,s0)); CKF(cudaEventSynchronize(tb));
    float ms; CKF(cudaEventElapsedTime(&ms,ta,tb));
    CKF(cudaEventDestroy(ta)); CKF(cudaEventDestroy(tb));
    sync2(0,s0,1,s1);
    if(r>0 && ms*1000.f/iters<best) best=ms*1000.f/iters;
  }
  return best;
}


int main(){
  setbuf(stdout, NULL);
  printf("=== tp_probe: 2-GPU tensor-parallel communication probe ===\n");
  int nd=0; CKF(cudaGetDeviceCount(&nd));
  printf("device count = %d\n", nd);
  if(nd < 2){ printf("FATAL: need 2 GPUs\n"); return 1; }
  int dv=0, rv=0; cudaDriverGetVersion(&dv); cudaRuntimeGetVersion(&rv);
  printf("driver API version %d, runtime version %d\n", dv, rv);

  cudaSetDevice(0); cudaError_t f0 = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  cudaSetDevice(1); cudaError_t f1 = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  if(f0||f1) printf("note: cudaSetDeviceFlags(Spin): %s / %s (continuing with default)\n",
                    cudaGetErrorString(f0), cudaGetErrorString(f1));

  cudaDeviceProp p0p, p1p;
  cudaSetDevice(0); cudaGetDeviceProperties(&p0p, 0);
  cudaSetDevice(1); cudaGetDeviceProperties(&p1p, 1);
  printf("dev0: %s sm_%d%d  |  dev1: %s sm_%d%d\n", p0p.name, p0p.major, p0p.minor, p1p.name, p1p.major, p1p.minor);

  // s0/s1, a0..b1, sc0/sc1, e0x/e1x/p0e/p1e are file-scope (shared with Q6-Q10 harnesses)
  cudaSetDevice(0); CKF(cudaMalloc(&a0,(size_t)N_BIG*2)); CKF(cudaMalloc(&b0,(size_t)N_BIG*2)); CKF(cudaMalloc(&sc0,4));
  cudaSetDevice(1); CKF(cudaMalloc(&a1,(size_t)N_BIG*2)); CKF(cudaMalloc(&b1,(size_t)N_BIG*2)); CKF(cudaMalloc(&sc1,4));

  cudaSetDevice(0); CKF(cudaStreamCreate(&s0));
  cudaSetDevice(1); CKF(cudaStreamCreate(&s1));
  char *hA,*hB,*hV,*hW,*h0s,*h1s,*h0g[2],*h1g[2];
  CKF(cudaMallocHost(&hA,(size_t)N_BIG*2));  // staging 0->1
  CKF(cudaMallocHost(&hB,(size_t)N_BIG*2));  // staging 1->0
  CKF(cudaMallocHost(&hV,(size_t)N_BIG*2));  // Q4b observe buffer (dev0 publishes)
  CKF(cudaMallocHost(&hW,(size_t)N_BIG*2));  // Q4b observe buffer (dev1 publishes)
  CKF(cudaMallocHost(&h0s,(size_t)N_BIG*2)); // Q5 eager allreduce host bufs
  CKF(cudaMallocHost(&h1s,(size_t)N_BIG*2));
  for(int i=0;i<2;i++){ CKF(cudaMallocHost(&h0g[i],(size_t)N_BIG*2)); CKF(cudaMallocHost(&h1g[i],(size_t)N_BIG*2)); }
  // events recorded on s0 created with dev0 current; events recorded on s1 with dev1 current
  cudaEvent_t eA,ebpB,pe0;                 // dev0-recorded (e0x,p0e file-scope for Q6-Q10)
  cudaEvent_t eB,ebpA,pe1;                 // dev1-recorded (e1x,p1e file-scope for Q6-Q10)
  cudaEvent_t* ge0 = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*AOPS);  // dev0-recorded
  cudaEvent_t* ge1 = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*AOPS);  // dev1-recorded
  cudaSetDevice(0);
  { cudaEvent_t* evs[] = {&eA,&ebpB,&e0x,&pe0,&p0e};
    for(auto ev : evs) CKF(cudaEventCreateWithFlags(ev, cudaEventDisableTiming)); }
  for(int i=0;i<AOPS;i++) CKF(cudaEventCreateWithFlags(&ge0[i], cudaEventDisableTiming));
  cudaSetDevice(1);
  { cudaEvent_t* evs[] = {&eB,&ebpA,&e1x,&pe1,&p1e};
    for(auto ev : evs) CKF(cudaEventCreateWithFlags(ev, cudaEventDisableTiming)); }
  for(int i=0;i<AOPS;i++) CKF(cudaEventCreateWithFlags(&ge1[i], cudaEventDisableTiming));

  // ================= Q1: peer access =================
  printf("\n===== Q1: PEER ACCESS =====\n");
  int c01=-1, c10=-1;
  CKF(cudaDeviceCanAccessPeer(&c01, 0, 1));
  CKF(cudaDeviceCanAccessPeer(&c10, 1, 0));
  printf("cudaDeviceCanAccessPeer(0 can access 1) = %d ; (1 can access 0) = %d\n", c01, c10);
  bool peer01=false, peer10=false;
  cudaSetDevice(0);
  { cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
    if(e==cudaSuccess) peer01=true;
    else { printf("cudaDeviceEnablePeerAccess(0 -> 1): %s\n", cudaGetErrorString(e)); cudaGetLastError(); } }
  cudaSetDevice(1);
  { cudaError_t e = cudaDeviceEnablePeerAccess(0, 0);
    if(e==cudaSuccess) peer10=true;
    else { printf("cudaDeviceEnablePeerAccess(1 -> 0): %s\n", cudaGetErrorString(e)); cudaGetLastError(); } }
  bool peerOK = peer01 && peer10 && c01 && c10;
  printf("PEER_ACCESS=%s\n", peerOK ? "ENABLED" : "UNAVAILABLE (fallback: staged through pinned host / cudaMemcpyPeerAsync)");

  // ================= Q2: bandwidth / latency =================
  printf("\n===== Q2: BANDWIDTH / LATENCY (min of 3 runs of 1000 transfers) =====\n");
  { int ns[2] = {N_SML, N_BIG};
    for(int k=0;k<2;k++){
      int n = ns[k]; size_t bytes = g_bytes[k];
      printf("-- n=%d bf16 = %d bytes --\n", n, (int)bytes);
      if(peerOK){
        float us = bench_min3([&](int){ CK(cudaMemcpyPeerAsync(b1,1,a0,0,bytes,s1)); }, 1, s1, 50, 1000);
        sync2(0,s0,1,s1); g_peer_us[k][0]=us;
        printf("PEER    0->1 : %8.2f us/op  %8.2f GB/s\n", us, gbps(bytes,us));
        us = bench_min3([&](int){ CK(cudaMemcpyPeerAsync(b0,0,a1,1,bytes,s0)); }, 0, s0, 50, 1000);
        sync2(0,s0,1,s1); g_peer_us[k][1]=us;
        printf("PEER    1->0 : %8.2f us/op  %8.2f GB/s\n", us, gbps(bytes,us));
      } else {
        // peer disabled: cudaMemcpyPeerAsync still succeeds; driver stages internally
        float us = bench_min3([&](int){ CK(cudaMemcpyPeerAsync(b1,1,a0,0,bytes,s1)); }, 1, s1, 50, 1000);
        sync2(0,s0,1,s1); g_peer_us[k][0]=us;
        printf("PEERAPI 0->1 : %8.2f us/op  %8.2f GB/s  (cudaMemcpyPeerAsync, peer disabled -> driver-internal staging)\n", us, gbps(bytes,us));
        us = bench_min3([&](int){ CK(cudaMemcpyPeerAsync(b0,0,a1,1,bytes,s0)); }, 0, s0, 50, 1000);
        sync2(0,s0,1,s1); g_peer_us[k][1]=us;
        printf("PEERAPI 1->0 : %8.2f us/op  %8.2f GB/s  (same)\n", us, gbps(bytes,us));
      }
      // staged 0->1: D2H on s0, cross-device event handshake, H2D on s1, back-pressure event
      auto f01 = [&](int){
        cudaSetDevice(0);
        CK(cudaMemcpyAsync(hA, a0, bytes, cudaMemcpyDeviceToHost, s0));
        CK(cudaEventRecord(eA, s0));
        cudaSetDevice(1);
        CK(cudaStreamWaitEvent(s1, eA, 0));
        CK(cudaMemcpyAsync(b1, hA, bytes, cudaMemcpyHostToDevice, s1));
        CK(cudaEventRecord(ebpA, s1));
        cudaSetDevice(0);
        CK(cudaStreamWaitEvent(s0, ebpA, 0));
      };
      sync2(0,s0,1,s1);
      float us = bench_min3(f01, 0, s0, 50, 1000); sync2(0,s0,1,s1); g_stg_us[k][0]=us;
      printf("STAGED  0->1 : %8.2f us/op  %8.2f GB/s  (D2H+H2D serialized incl. 2 cross-device event handshakes)\n",
             us, gbps(bytes,us));
      auto f10 = [&](int){
        cudaSetDevice(1);
        CK(cudaMemcpyAsync(hB, a1, bytes, cudaMemcpyDeviceToHost, s1));
        CK(cudaEventRecord(eB, s1));
        cudaSetDevice(0);
        CK(cudaStreamWaitEvent(s0, eB, 0));
        CK(cudaMemcpyAsync(b0, hB, bytes, cudaMemcpyHostToDevice, s0));
        CK(cudaEventRecord(ebpB, s0));
        cudaSetDevice(1);
        CK(cudaStreamWaitEvent(s1, ebpB, 0));
      };
      sync2(0,s0,1,s1);
      us = bench_min3(f10, 0, s0, 50, 1000); sync2(0,s0,1,s1); g_stg_us[k][1]=us;
      printf("STAGED  1->0 : %8.2f us/op  %8.2f GB/s  (same methodology)\n", us, gbps(bytes,us));
    }
  }

  // ================= Q3: peer-mapped kernel writes =================
  printf("\n===== Q3: PEER-MAPPED KERNEL WRITES (dev1 kernel -> dev0 pointer) =====\n");
  bool pm_ok=false;
  if(!peerOK) printf("SKIPPED (peer access unavailable)\n");
  else {
    cudaSetDevice(0);
    k_fill<<<GRS,256,0,s0>>>(a0, N_SML, 3.f); CK(cudaGetLastError()); CK(cudaStreamSynchronize(s0));
    cudaSetDevice(1);
    k_fill<<<GRS,256,0,s1>>>(a0, N_SML, 7.f);            // kernel on dev1 writing dev0 memory
    cudaError_t le = cudaGetLastError();
    CK(cudaStreamSynchronize(s1));
    cudaSetDevice(0);
    float s = dev_sum(a0, N_SML, sc0, s0);
    pm_ok = (le==cudaSuccess) && fabsf(s - 7.f*N_SML) < 1.f;
    printf("dev1 kernel writing dev0 buffer: launch=%s checksum=%.1f expected=%.1f -> %s\n",
           cudaGetErrorString(le), s, 7.f*N_SML, pm_ok?"OK":"FAIL");
    // ping-pong: dev1 kernel writes dev0 buf, dev0 kernel writes dev1 buf, alternating
    cudaSetDevice(0); CK(cudaEventRecord(pe0, s0)); sync2(0,s0,1,s1);
    g_pp_us = bench_min3([&](int){
      cudaSetDevice(1);
      CK(cudaStreamWaitEvent(s1, pe0, 0));
      k_fill<<<GRS,256,0,s1>>>(a0, N_SML, 1.f); CK(cudaGetLastError());
      CK(cudaEventRecord(pe1, s1));
      cudaSetDevice(0);
      CK(cudaStreamWaitEvent(s0, pe1, 0));
      k_fill<<<GRS,256,0,s0>>>(a1, N_SML, 1.f); CK(cudaGetLastError());
      CK(cudaEventRecord(pe0, s0));
    }, 0, s0, 20, 200);
    sync2(0,s0,1,s1);
    printf("ping-pong 10KB peer kernel-to-kernel writes: %.2f us per round trip\n", g_pp_us);
  }
  printf("PEER_MAPPED_WRITES=%s\n", !peerOK?"SKIPPED":(pm_ok?"WORKS":"FAIL"));

  // ================= Q4a: cross-device events, no capture =================
  printf("\n===== Q4a: CROSS-DEVICE EVENTS WITHOUT CAPTURE (100 alternating reps) =====\n");
  bool cdsync=false;
  { size_t bytes=(size_t)N_SML*2; bool okA=true, okB=true; int shownA=0, shownB=0;
    for(int rep=0; rep<100; rep++){
      float v0=(rep&1)?3.f:5.f, v1=(rep&1)?5.f:3.f;
      cudaSetDevice(0);
      k_fill_delay<<<GRS,256,0,s0>>>(a0,N_SML,v0,SPIN); CK(cudaGetLastError());
      CK(cudaEventRecord(e0x, s0));
      cudaSetDevice(1);
      CK(cudaStreamWaitEvent(s1, e0x, 0));
      CK(cudaMemcpyPeerAsync(b1, 1, a0, 0, bytes, s1));
      CK(cudaStreamSynchronize(s1));
      float sA = dev_sum(b1, N_SML, sc1, s1);
      if(fabsf(sA - v0*N_SML) > 1.f){ okA=false; if(shownA<3){ printf("    rep %d dir 0->1: saw %.1f expected %.1f\n", rep, sA, v0*N_SML); shownA++; } }
      cudaSetDevice(1);
      k_fill_delay<<<GRS,256,0,s1>>>(a1,N_SML,v1,SPIN); CK(cudaGetLastError());
      CK(cudaEventRecord(e1x, s1));
      cudaSetDevice(0);
      CK(cudaStreamWaitEvent(s0, e1x, 0));
      CK(cudaMemcpyPeerAsync(b0, 0, a1, 1, bytes, s0));
      CK(cudaStreamSynchronize(s0));
      float sB = dev_sum(b0, N_SML, sc0, s0);
      if(fabsf(sB - v1*N_SML) > 1.f){ okB=false; if(shownB<3){ printf("    rep %d dir 1->0: saw %.1f expected %.1f\n", rep, sB, v1*N_SML); shownB++; } }
    }
    cdsync = okA && okB;
    printf("dev0 record -> dev1 wait -> dev1 reads dev0 data : %s\n", okA?"WORKS":"FAIL");
    printf("dev1 record -> dev0 wait -> dev0 reads dev1 data : %s\n", okB?"WORKS":"FAIL");
  }
  printf("CROSS_DEVICE_EVENT_STREAM_SYNC=%s\n", cdsync?"WORKS":"FAIL");

  // ================= Q4b: cross-device events inside CUDA graphs =================
  printf("\n===== Q4b: CROSS-DEVICE EVENTS INSIDE GRAPHS =====\n");
  bool graphsOk=false; char werr[512] = "";
  int r1, r2; char err1[256]="", err2[256]="";
  // W1: graph on dev1 waits event recorded on dev0
  r1 = graph_wait_test(s1, s0, 1, 0, a0, b1, sc1, hV, e0x, cudaStreamCaptureModeGlobal, err1, sizeof err1);
  if(r1==1){
    printf("W1 (graph dev1 <- record dev0) Global capture failed: %s\n", err1);
    r1 = graph_wait_test(s1, s0, 1, 0, a0, b1, sc1, hV, e0x, cudaStreamCaptureModeRelaxed, err1, sizeof err1);
    if(r1==0) printf("W1 retry with cudaStreamCaptureModeRelaxed: capture OK, replay CORRECT\n");
    else if(r1==2) printf("W1 retry with Relaxed: capture OK, replay WRONG (stale data)\n");
    else printf("W1 retry with Relaxed failed: %s\n", err1);
  } else printf("W1 (graph dev1 waits dev0-recorded event, Global mode): capture OK, replay %s\n", r1==0?"CORRECT":"WRONG");
  // W2: graph on dev0 waits event recorded on dev1
  r2 = graph_wait_test(s0, s1, 0, 1, a1, b0, sc0, hW, e1x, cudaStreamCaptureModeGlobal, err2, sizeof err2);
  if(r2==1){
    printf("W2 (graph dev0 <- record dev1) Global capture failed: %s\n", err2);
    r2 = graph_wait_test(s0, s1, 0, 1, a1, b0, sc0, hW, e1x, cudaStreamCaptureModeRelaxed, err2, sizeof err2);
    if(r2==0) printf("W2 retry with cudaStreamCaptureModeRelaxed: capture OK, replay CORRECT\n");
    else if(r2==2) printf("W2 retry with Relaxed: capture OK, replay WRONG (stale data)\n");
    else printf("W2 retry with Relaxed failed: %s\n", err2);
  } else printf("W2 (graph dev0 waits dev1-recorded event, Global mode): capture OK, replay %s\n", r2==0?"CORRECT":"WRONG");
  graphsOk = (r1==0) && (r2==0);
  // R: capture-time cross-device record — record into dev0's stream while capturing dev1's stream
  { sync2(1,s1,0,s0); cudaSetDevice(1);
    cudaError_t cb = cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
    if(cb!=cudaSuccess) printf("record-test BeginCapture failed: %s\n", cudaGetErrorString(cb));
    else {
      cudaError_t rr = cudaEventRecord(e0x, s0);
      printf("capture-time cross-device cudaEventRecord(dev0 stream, while capturing dev1, Global): %s\n",
             cudaGetErrorString(rr));
      cudaGraph_t gt=NULL;
      cudaError_t er = cudaStreamEndCapture(s1, &gt);
      printf("  EndCapture after cross-device record: %s\n", cudaGetErrorString(er));
      if(gt) cudaGraphDestroy(gt);
      cudaGetLastError();
    }
  }
  if(!graphsOk){
    if(r1!=0 && err1[0]) snprintf(werr,sizeof werr,"W1: %s",err1);
    if(r2!=0 && err2[0]) snprintf(werr+strlen(werr),sizeof(werr)-strlen(werr),"%sW2: %s", werr[0]?" ; ":"",err2);
    if(!werr[0]) snprintf(werr,sizeof werr,"capture succeeded but replay ordering wrong");
  }
  printf("GRAPHS_CROSS_DEVICE_EVENTS=%s%s%s\n", graphsOk?"WORKS":"UNSUPPORTED", werr[0]?" -- ":"", werr);

  // ================= Q5: allreduce microbench =================
  printf("\n===== Q5: ALLREDUCE x%d, n=%d bf16 (10KB), lockstep exchange+add =====\n", AOPS, N_SML);
  { size_t bytes=(size_t)N_SML*2;
    // ---- eager, staged transport (manual D2H + H2D through pinned host) ----
    { cudaSetDevice(0); k_fill<<<GRS,256,0,s0>>>(a0,N_SML,1.f); CK(cudaGetLastError());
      CK(cudaMemcpyAsync(h0s, a0, bytes, cudaMemcpyDeviceToHost, s0)); CK(cudaEventRecord(p0e, s0));
      cudaSetDevice(1); k_fill<<<GRS,256,0,s1>>>(a1,N_SML,2.f); CK(cudaGetLastError());
      CK(cudaMemcpyAsync(h1s, a1, bytes, cudaMemcpyDeviceToHost, s1)); CK(cudaEventRecord(p1e, s1));
      sync2(0,s0,1,s1);
      auto op = [&](int){
        cudaSetDevice(0);
        CK(cudaStreamWaitEvent(s0, p1e, 0));
        CK(cudaMemcpyAsync(b0, h1s, bytes, cudaMemcpyHostToDevice, s0));
        k_add<<<GRS,256,0,s0>>>(a0, b0, N_SML); CK(cudaGetLastError());
        CK(cudaMemcpyAsync(h0s, a0, bytes, cudaMemcpyDeviceToHost, s0));
        CK(cudaEventRecord(p0e, s0));
        cudaSetDevice(1);
        CK(cudaStreamWaitEvent(s1, p0e, 0));
        CK(cudaMemcpyAsync(b1, h0s, bytes, cudaMemcpyHostToDevice, s1));
        k_add<<<GRS,256,0,s1>>>(a1, b1, N_SML); CK(cudaGetLastError());
        CK(cudaMemcpyAsync(h1s, a1, bytes, cudaMemcpyDeviceToHost, s1));
        CK(cudaEventRecord(p1e, s1));
      };
      op(0); sync2(0,s0,1,s1);
      cudaSetDevice(0); float v0 = dev_sum(a0,N_SML,sc0,s0);
      cudaSetDevice(1); float v1 = dev_sum(a1,N_SML,sc1,s1);
      printf("staged eager verify after 1 op: sum0=%.1f (exp %.1f) sum1=%.1f (exp %.1f) -> %s\n",
             v0, 3.f*N_SML, v1, 5.f*N_SML, (fabsf(v0-3.f*N_SML)<1.f && fabsf(v1-5.f*N_SML)<1.f)?"OK":"FAIL");
      float best=1e30f;
      for(int r=0;r<3;r++){
        cudaSetDevice(0);
        cudaEvent_t ta,tb; CKF(cudaEventCreate(&ta)); CKF(cudaEventCreate(&tb));
        CKF(cudaEventRecord(ta, s0));
        for(int i=0;i<AOPS;i++) op(i);
        CK(cudaStreamWaitEvent(s0, p1e, 0));       // cover dev1's final op
        CKF(cudaEventRecord(tb, s0)); CKF(cudaEventSynchronize(tb));
        float ms; CKF(cudaEventElapsedTime(&ms,ta,tb));
        CKF(cudaEventDestroy(ta)); CKF(cudaEventDestroy(tb));
        sync2(0,s0,1,s1);
        if(ms*1000.f/AOPS < best) best = ms*1000.f/AOPS;
      }
      g_eager_stg_us = best;
      printf("EAGER  staged  : %.3f ms per %d ops, %.2f us/op\n", best*AOPS/1000.f, AOPS, best);
    }
    // ---- eager, cudaMemcpyPeerAsync transport (works with or without peer enable) ----
    { cudaSetDevice(0); k_fill<<<GRS,256,0,s0>>>(a0,N_SML,1.f); CK(cudaGetLastError()); CK(cudaEventRecord(p0e, s0));
      cudaSetDevice(1); k_fill<<<GRS,256,0,s1>>>(a1,N_SML,2.f); CK(cudaGetLastError()); CK(cudaEventRecord(p1e, s1));
      sync2(0,s0,1,s1);
      auto op = [&](int){
        cudaSetDevice(0);
        CK(cudaStreamWaitEvent(s0, p1e, 0));
        CK(cudaMemcpyPeerAsync(b0, 0, a1, 1, bytes, s0));
        k_add<<<GRS,256,0,s0>>>(a0, b0, N_SML); CK(cudaGetLastError());
        CK(cudaEventRecord(p0e, s0));
        cudaSetDevice(1);
        CK(cudaStreamWaitEvent(s1, p0e, 0));
        CK(cudaMemcpyPeerAsync(b1, 1, a0, 0, bytes, s1));
        k_add<<<GRS,256,0,s1>>>(a1, b1, N_SML); CK(cudaGetLastError());
        CK(cudaEventRecord(p1e, s1));
      };
      op(0); sync2(0,s0,1,s1);
      cudaSetDevice(0); float v0 = dev_sum(a0,N_SML,sc0,s0);
      cudaSetDevice(1); float v1 = dev_sum(a1,N_SML,sc1,s1);
      printf("peerapi eager verify after 1 op: sum0=%.1f sum1=%.1f -> %s\n", v0, v1,
            (fabsf(v0-3.f*N_SML)<1.f && fabsf(v1-5.f*N_SML)<1.f)?"OK":"FAIL");
      float best=1e30f;
      for(int r=0;r<3;r++){
        cudaSetDevice(0);
        cudaEvent_t ta,tb; CKF(cudaEventCreate(&ta)); CKF(cudaEventCreate(&tb));
        CKF(cudaEventRecord(ta, s0));
        for(int i=0;i<AOPS;i++) op(i);
        CK(cudaStreamWaitEvent(s0, p1e, 0));
        CKF(cudaEventRecord(tb, s0)); CKF(cudaEventSynchronize(tb));
        float ms; CKF(cudaEventElapsedTime(&ms,ta,tb));
        CKF(cudaEventDestroy(ta)); CKF(cudaEventDestroy(tb));
        sync2(0,s0,1,s1);
        if(ms*1000.f/AOPS < best) best = ms*1000.f/AOPS;
      }
      g_eager_peer_us = best;
      printf("EAGER  peerapi: %.3f ms per %d ops, %.2f us/op%s\n", best*AOPS/1000.f, AOPS, best,
             peerOK?"":"  (cudaMemcpyPeerAsync with peer disabled: driver stages internally)");
    }
    // ---- graphs: graph-INTERNAL cross-device event chains (record/wait nodes inside the
    //      paired graphs; waits reference events not yet recorded at capture time) ----
    if(!graphsOk) printf("note: W1/W2 (wait on externally pre-recorded event) failed capture; testing the\n"
                         "      graph-INTERNAL event chain variant (record/wait nodes inside the graphs) anyway\n");
    {
      // reference: 128 eager staged ops from fresh init (1,2)
      cudaSetDevice(0); k_fill<<<GRS,256,0,s0>>>(a0,N_SML,1.f); CK(cudaGetLastError());
      CK(cudaMemcpyAsync(h0s, a0, bytes, cudaMemcpyDeviceToHost, s0)); CK(cudaEventRecord(p0e, s0));
      cudaSetDevice(1); k_fill<<<GRS,256,0,s1>>>(a1,N_SML,2.f); CK(cudaGetLastError());
      CK(cudaMemcpyAsync(h1s, a1, bytes, cudaMemcpyDeviceToHost, s1)); CK(cudaEventRecord(p1e, s1));
      sync2(0,s0,1,s1);
      for(int i=0;i<AOPS;i++){
        cudaSetDevice(0);
        CK(cudaStreamWaitEvent(s0, p1e, 0));
        CK(cudaMemcpyAsync(b0, h1s, bytes, cudaMemcpyHostToDevice, s0));
        k_add<<<GRS,256,0,s0>>>(a0, b0, N_SML); CK(cudaGetLastError());
        CK(cudaMemcpyAsync(h0s, a0, bytes, cudaMemcpyDeviceToHost, s0));
        CK(cudaEventRecord(p0e, s0));
        cudaSetDevice(1);
        CK(cudaStreamWaitEvent(s1, p0e, 0));
        CK(cudaMemcpyAsync(b1, h0s, bytes, cudaMemcpyHostToDevice, s1));
        k_add<<<GRS,256,0,s1>>>(a1, b1, N_SML); CK(cudaGetLastError());
        CK(cudaMemcpyAsync(h1s, a1, bytes, cudaMemcpyDeviceToHost, s1));
        CK(cudaEventRecord(p1e, s1));
      }
      sync2(0,s0,1,s1);
      cudaSetDevice(0); float refA = dev_sum(a0,N_SML,sc0,s0);
      printf("graph reference (eager 128 ops from init 1,2): sum0=%.1f\n", refA);

      // G0 on s0: op0 free (consumes prefilled h1g[1]); op i>0 waits ge1[i-1];
      //   op i: H2D b0<-h1g[(i+1)&1], add, D2H h0g[i&1], record ge0[i].
      // G1 on s1: op i waits ge0[i]; H2D b1<-h0g[i&1], add, D2H h1g[i&1], record ge1[i].
      cudaGraph_t gA=NULL,gBv=NULL; cudaGraphExec_t xA=NULL,xB=NULL;
      bool capOK=false; char gerr[256]="";
      for(int m=0; m<2 && !capOK; m++){
        cudaStreamCaptureMode mode = m==0 ? cudaStreamCaptureModeGlobal : cudaStreamCaptureModeRelaxed;
        sync2(0,s0,1,s1);
        cudaSetDevice(0);
        cudaError_t cb0 = cudaStreamBeginCapture(s0, mode);
        if(cb0!=cudaSuccess){ snprintf(gerr,sizeof gerr,"G0 BeginCapture: %s", cudaGetErrorString(cb0)); continue; }
        cudaError_t t = cudaSuccess;
        for(int i=0;i<AOPS && t==cudaSuccess;i++){
          if(i>0) t = cudaStreamWaitEvent(s0, ge1[i-1], 0);
          if(t==cudaSuccess) t = cudaMemcpyAsync(b0, h1g[(i+1)&1], bytes, cudaMemcpyHostToDevice, s0);
          if(t==cudaSuccess){ k_add<<<GRS,256,0,s0>>>(a0, b0, N_SML); t = cudaGetLastError(); }
          if(t==cudaSuccess) t = cudaMemcpyAsync(h0g[i&1], a0, bytes, cudaMemcpyDeviceToHost, s0);
          if(t==cudaSuccess) t = cudaEventRecord(ge0[i], s0);
        }
        cudaError_t e = cudaStreamEndCapture(s0, &gA);
        if(t!=cudaSuccess || e!=cudaSuccess || gA==NULL){
          snprintf(gerr,sizeof gerr,"G0 capture (%s): body=%s end=%s", m?"Relaxed":"Global",
                   cudaGetErrorString(t), cudaGetErrorString(e));
          cudaGetLastError();
          if(gA){ cudaGraphDestroy(gA); gA=NULL; }
          continue;
        }
        cudaSetDevice(1);
        cudaError_t cb1 = cudaStreamBeginCapture(s1, mode);
        if(cb1!=cudaSuccess){
          snprintf(gerr,sizeof gerr,"G1 BeginCapture (%s): %s", m?"Relaxed":"Global", cudaGetErrorString(cb1));
          cudaGraphDestroy(gA); gA=NULL; continue;
        }
        t = cudaSuccess;
        for(int i=0;i<AOPS && t==cudaSuccess;i++){
          t = cudaStreamWaitEvent(s1, ge0[i], 0);
          if(t==cudaSuccess) t = cudaMemcpyAsync(b1, h0g[i&1], bytes, cudaMemcpyHostToDevice, s1);
          if(t==cudaSuccess){ k_add<<<GRS,256,0,s1>>>(a1, b1, N_SML); t = cudaGetLastError(); }
          if(t==cudaSuccess) t = cudaMemcpyAsync(h1g[i&1], a1, bytes, cudaMemcpyDeviceToHost, s1);
          if(t==cudaSuccess) t = cudaEventRecord(ge1[i], s1);
        }
        e = cudaStreamEndCapture(s1, &gBv);
        if(t!=cudaSuccess || e!=cudaSuccess || gBv==NULL){
          snprintf(gerr,sizeof gerr,"G1 capture (%s): body=%s end=%s", m?"Relaxed":"Global",
                   cudaGetErrorString(t), cudaGetErrorString(e));
          cudaGetLastError();
          if(gBv){ cudaGraphDestroy(gBv); gBv=NULL; }
          cudaGraphDestroy(gA); gA=NULL; continue;
        }
        cudaError_t e0i = cudaGraphInstantiate(&xA, gA, 0), e1i = cudaGraphInstantiate(&xB, gBv, 0);
        if(e0i||e1i){
          snprintf(gerr,sizeof gerr,"instantiate (%s): %s / %s", m?"Relaxed":"Global",
                   cudaGetErrorString(e0i), cudaGetErrorString(e1i));
          if(xA){ cudaGraphExecDestroy(xA); xA=NULL; } if(xB){ cudaGraphExecDestroy(xB); xB=NULL; }
          cudaGraphDestroy(gA); gA=NULL; cudaGraphDestroy(gBv); gBv=NULL; continue;
        }
        capOK = true;
        if(m==1) printf("note: graph pair only capturable in cudaStreamCaptureModeRelaxed\n");
      }
      if(!capOK) printf("GRAPH allreduce capture failed: %s\n", gerr);
      else {
        float best=1e30f; int badsum=0;
        for(int rep=0; rep<3; rep++){
          // reset: a0=1, a1=2; h1g[1] must hold dev1's initial data for G0 op0
          cudaSetDevice(0); k_fill<<<GRS,256,0,s0>>>(a0,N_SML,1.f); CK(cudaGetLastError());
          cudaSetDevice(1); k_fill<<<GRS,256,0,s1>>>(a1,N_SML,2.f); CK(cudaGetLastError());
          CK(cudaMemcpyAsync(h1g[1], a1, bytes, cudaMemcpyDeviceToHost, s1));
          sync2(0,s0,1,s1);
          cudaSetDevice(0);
          cudaEvent_t ta,tb; CKF(cudaEventCreate(&ta)); CKF(cudaEventCreate(&tb));
          CKF(cudaEventRecord(ta, s0));
          CKF(cudaGraphLaunch(xA, s0));
          cudaSetDevice(1); CKF(cudaGraphLaunch(xB, s1));
          CKF(cudaEventRecord(e1x, s1));          // eager terminal record after xB in stream order
          cudaSetDevice(0);
          CK(cudaStreamWaitEvent(s0, e1x, 0));    // reliable end fence (stream semantics)
          CKF(cudaEventRecord(tb, s0)); CKF(cudaEventSynchronize(tb));
          float ms; CKF(cudaEventElapsedTime(&ms,ta,tb));
          CKF(cudaEventDestroy(ta)); CKF(cudaEventDestroy(tb));
          sync2(0,s0,1,s1);
          cudaSetDevice(0); float sA = dev_sum(a0,N_SML,sc0,s0);
          if(sA != refA) badsum++;
          if(rep==0) printf("graph replay verify (fresh round): sum0=%.1f expected %.1f -> %s\n",
                            sA, refA, sA==refA?"OK":"MISMATCH");
          if(ms*1000.f/AOPS < best) best = ms*1000.f/AOPS;
        }
        if(badsum==0){
          g_graph_us = best;
          printf("GRAPH  staged : %.3f ms per %d ops, %.2f us/op  (checksum OK in all 3 rounds)\n",
                 best*AOPS/1000.f, AOPS, best);
        } else {
          printf("GRAPH  staged : %.2f us/op measured BUT %d/3 rounds MISMATCH: graph-internal cross-device\n"
                 "event chains do NOT safely order back-to-back replays (stale signaled events) -> UNSAFE\n",
                 best, badsum);
        }
      }
      if(xA) cudaGraphExecDestroy(xA);  if(gA) cudaGraphDestroy(gA);
      if(xB) cudaGraphExecDestroy(xB);  if(gBv) cudaGraphDestroy(gBv);
    }
  }

  // ================= Q6: mapped pinned cross-device visibility =================
  printf("\n===== Q6: MAPPED PINNED CROSS-DEVICE VISIBILITY =====\n");
  printf("prop.canMapHostMemory: dev0=%d dev1=%d\n", p0p.canMapHostMemory, p1p.canMapHostMemory);
  { bool ptrOK=false, visOK=false;
    cudaError_t ea = cudaHostAlloc(&hMap, MAP_BYTES, cudaHostAllocMapped | cudaHostAllocPortable);
    if(ea != cudaSuccess) printf("cudaHostAlloc(Mapped|Portable, %.1f MB): %s\n", MAP_BYTES/1048576.0, cudaGetErrorString(ea));
    else {
      cudaSetDevice(0); cudaError_t r0 = cudaHostGetDevicePointer(&m0, hMap, 0);
      cudaSetDevice(1); cudaError_t r1 = cudaHostGetDevicePointer(&m1, hMap, 0);
      printf("cudaHostGetDevicePointer: dev0 %s -> %p ; dev1 %s -> %p ; host %p (same VA as host: %d/%d)\n",
             cudaGetErrorString(r0), (void*)m0, cudaGetErrorString(r1), (void*)m1, (void*)hMap,
             m0==(char*)hMap, m1==(char*)hMap);
      ptrOK = (r0==cudaSuccess && r1==cudaSuccess && m0 && m1);
      if(ptrOK){
        visOK = true; int shown=0;
        for(int rep=0; rep<50 && visOK; rep++){        // alternating directions, delayed-write free (volatile+fence)
          float v0=(rep&1)?3.f:5.f, v1=(rep&1)?5.f:3.f;
          cudaSetDevice(0);
          k_write_mapped<<<GRS,256,0,s0>>>((volatile bf16*)(m0+pay_off(0,0)), N_SML, v0); CK(cudaGetLastError());
          CK(cudaEventRecord(e0x, s0));
          cudaSetDevice(1);
          CK(cudaStreamWaitEvent(s1, e0x, 0));
          float sA = dev_sum_mapped((volatile bf16*)(m1+pay_off(0,0)), N_SML, sc1, s1);
          if(fabsf(sA - v0*N_SML) > 1.f){ visOK=false; if(shown<3){ printf("    rep %d 0->1: saw %.1f expected %.1f\n", rep, sA, v0*N_SML); shown++; } }
          cudaSetDevice(1);
          k_write_mapped<<<GRS,256,0,s1>>>((volatile bf16*)(m1+pay_off(1,0)), N_SML, v1); CK(cudaGetLastError());
          CK(cudaEventRecord(e1x, s1));
          cudaSetDevice(0);
          CK(cudaStreamWaitEvent(s0, e1x, 0));
          float sB = dev_sum_mapped((volatile bf16*)(m0+pay_off(1,0)), N_SML, sc0, s0);
          if(fabsf(sB - v1*N_SML) > 1.f){ visOK=false; if(shown<3){ printf("    rep %d 1->0: saw %.1f expected %.1f\n", rep, sB, v1*N_SML); shown++; } }
        }
        printf("cross-visible rounds (50 alternating kernel-write -> kernel-read): %s\n", visOK?"ALL OK":"FAILED");
      }
    }
    g_map_ok = ptrOK && visOK;
    printf("MAPPED_PINNED_CROSS_VISIBLE=%s\n", g_map_ok?"WORKS":"FAIL");
    if(!g_map_ok){
      printf("-> host-pinned mapped transport is dead on this WDDM stack; stopping after Q6 as planned.\n");
      printf("\n===== SUMMARY (QUESTION=ANSWER) =====\n");
      printf("Q6 MAPPED_PINNED_CROSS_VISIBLE=FAIL (Q7-Q10 skipped by design)\n");
      printf("total CUDA errors encountered: %d\n", g_nerr.load());
      return 2;
    }
    // shared resources for Q7-Q10
    cudaSetDevice(0); CKF(cudaMalloc(&acc0,(size_t)NMAX*4)); CKF(cudaMalloc(&go0,32)); CKF(cudaMalloc(&dbg0,sizeof(ull)*(2*AOPS+4)));
    CKF(cudaMalloc(&qs0,(size_t)NMAX*2)); CKF(cudaMalloc(&rs0,(size_t)NMAX*2));
    cudaSetDevice(1); CKF(cudaMalloc(&acc1,(size_t)NMAX*4)); CKF(cudaMalloc(&go1,32)); CKF(cudaMalloc(&dbg1,sizeof(ull)*(2*AOPS+4)));
    CKF(cudaMalloc(&qs1,(size_t)NMAX*2)); CKF(cudaMalloc(&rs1,(size_t)NMAX*2));
    CKF(cudaMallocHost(&hqs0,(size_t)NMAX*2)); CKF(cudaMallocHost(&hqs1,(size_t)NMAX*2));
  }

  // ================= Q7: spin-handshake allreduce, eager, two host threads =================
  printf("\n===== Q7: SPIN-HANDSHAKE ALLREDUCE (eager, one host thread per device) =====\n");
  { const int n7[2]={N_SML,20480}; const char* t7[2]={"10KB","40KB"};
    for(int k=0;k<2;k++){
      int n=n7[k];
      printf("-- n=%d bf16 (%s): bf16-feedback verify, then %d DEPENDENT ops x (warmup+3) --\n", n, t7[k], AOPS);
      spin_ar_run(n,8,1,false,true);
      float us=spin_ar_bench(n,AOPS,false);
      g_q7_us[k][0]=us;
      printf("EAGER spin        %s: %8.2f us/op  (%.3f ms per %d; staged baseline 126.24 us/op -> %.1fx)\n",
             t7[k], us, us*AOPS/1000.f, AOPS, 126.24f/us);
      us=spin_ar_bench(n,AOPS,true);
      g_q7_us[k][1]=us;
      printf("EAGER spin+nsleep %s: %8.2f us/op  (%.3f ms per %d)\n", t7[k], us, us*AOPS/1000.f, AOPS);
    }
  }

  // ================= Q9: graph replay submission cost =================
  printf("\n===== Q9: cudaGraphLaunch SUBMISSION COST (~10 kernel nodes, two threads) =====\n");
  { cudaGraphExec_t x9[2]={NULL,NULL};
    for(int d=0;d<2;d++){
      cudaSetDevice(d); cudaStream_t s=d?s1:s0;
      cudaGraph_t g=NULL;
      CKF(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      for(int k=0;k<10;k++){ k_fill<<<1,256,0,s>>>((bf16*)((d? a1:a0)), 512, 1.f); CK(cudaGetLastError()); }
      cudaError_t ee = cudaStreamEndCapture(s,&g); CKF(ee);
      CKF(cudaGraphInstantiate(&x9[d],g,0)); cudaGraphDestroy(g);
    }
    float res[2]={-1,-1};
    auto w=[&](int d){
      cudaSetDevice(d); cudaStream_t s=d?s1:s0;
      for(int i=0;i<100;i++) CK(cudaGraphLaunch(x9[d],s));      // warmup
      CK(cudaStreamSynchronize(s));
      float best=1e30f;
      for(int r=0;r<3;r++){
        CK(cudaStreamSynchronize(s));
        double t0=now_us();
        for(int i=0;i<1000;i++) CK(cudaGraphLaunch(x9[d],s));   // submission only, no sync inside
        double t1=now_us();
        CK(cudaStreamSynchronize(s));
        if((float)((t1-t0)/1000.0) < best) best=(float)((t1-t0)/1000.0);
      }
      res[d]=best;
    };
    std::thread t0w(w,0), t1w(w,1); t0w.join(); t1w.join();
    g_q9_us[0]=res[0]; g_q9_us[1]=res[1];
    printf("cudaGraphLaunch per launch, 1000 back-to-back per thread: dev0 %.2f us, dev1 %.2f us\n", res[0], res[1]);
    cudaGraphExecDestroy(x9[0]); cudaGraphExecDestroy(x9[1]);
  }

  // ================= Q10: allreduce bandwidth curve =================
  printf("\n===== Q10: ALLREDUCE BANDWIDTH CURVE, mapped-spin vs staged memcpy =====\n");
  { const int n10[5]={5120,20480,81920,327680,1310720};
    const char* t10[5]={"10KB","40KB","160KB","640KB","2.5MB"};
    printf("32 dependent ops x (warmup+3) per size ; eff GB/s counts 2*payload/op ; spin variant per Q7 winner\n");
    printf("size  : MAPPED spin allreduce           | STAGED memcpy allreduce (Q5-style lockstep)\n");
    for(int k=0;k<5;k++){
      bool bo = (k<2) ? (g_q7_us[k][1]>0 && g_q7_us[k][1]<g_q7_us[k][0]) : false;
      int n=n10[k]; size_t S=(size_t)n*2;
      spin_ar_run(n,8,1,false,true);                     // bf16-feedback correctness per size
      float usM=spin_ar_bench(n,32,bo);
      float usS=staged_ar_us(n,32);
      g_q10_spin_us[k]=usM; g_q10_stg_us[k]=usS;
      printf("%-5s : %8.2f us/op %7.2f GB/s | %8.2f us/op %7.2f GB/s\n",
             t10[k], usM, 2.0*S/usM/1000.0, usS, 2.0*S/usS/1000.0);
    }
  }

  // ================= Q8: spin allreduce inside CUDA graphs =================
  // executed last: a deadlock/TDR here would poison the context for every earlier question
  printf("\n===== Q8: SPIN ALLREDUCE INSIDE CUDA GRAPHS (two-thread replay, both launch orders) =====\n");
  { // diagnostic first: 128 chained TINY kernel nodes per graph, no communication -- bounds
    // the per-node graph execution/scheduling cost on WDDM without any spin handshake
    cudaGraphExec_t xd[2]={NULL,NULL};
    for(int d=0;d<2;d++){
      cudaSetDevice(d); cudaStream_t s=d?s1:s0; cudaGraph_t g=NULL;
      CKF(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      for(int i=0;i<AOPS;i++){ k_fill<<<1,256,0,s>>>((bf16*)(d? b1:b0),512,1.f); CK(cudaGetLastError()); }
      CKF(cudaStreamEndCapture(s,&g));
      CKF(cudaGraphInstantiate(&xd[d],g,0)); cudaGraphDestroy(g);
    }
    float best=1e30f;
    for(int r=0;r<4;r++){
      Gate gA,gB;
      double t0=now_us();
      std::thread ta(replay_worker,0,xd[0],s0,nullptr,&gA);
      std::thread tb(replay_worker,1,xd[1],s1,nullptr,&gB);
      ta.join(); tb.join();
      float us=(float)((now_us()-t0)/(double)AOPS);
      if(r>0 && us<best) best=us;
    }
    g_q8_node_us=best;
    printf("diag: %d chained tiny kernel nodes per graph, concurrent two-thread replay: %.2f us/node\n", AOPS, best);
    cudaGraphExecDestroy(xd[0]); cudaGraphExecDestroy(xd[1]);
  }
  { // diag 2: same chained tiny nodes, but each kernel WRITES mapped host memory (volatile+fence):
    // isolates the node-boundary cost of sysmem-writing kernels inside graphs on WDDM
    cudaGraphExec_t xm[2]={NULL,NULL};
    for(int d=0;d<2;d++){
      cudaSetDevice(d); cudaStream_t s=d?s1:s0; cudaGraph_t g=NULL;
      const char* map = d? m1 : m0;
      CKF(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      for(int i=0;i<AOPS;i++){ k_write_mapped<<<GRS,256,0,s>>>((volatile bf16*)(map+pay_off(d,0)), N_SML, 1.f); CK(cudaGetLastError()); }
      CKF(cudaStreamEndCapture(s,&g));
      CKF(cudaGraphInstantiate(&xm[d],g,0)); cudaGraphDestroy(g);
    }
    float best=1e30f;
    for(int r=0;r<4;r++){
      Gate gA,gB;
      double t0=now_us();
      std::thread ta(replay_worker,0,xm[0],s0,nullptr,&gA);
      std::thread tb(replay_worker,1,xm[1],s1,nullptr,&gB);
      ta.join(); tb.join();
      float us=(float)((now_us()-t0)/(double)AOPS);
      if(r>0 && us<best) best=us;
    }
    g_q8_mnode_us=best;
    printf("diag: %d chained mapped-WRITING kernel nodes per graph, concurrent replay: %.2f us/node\n", AOPS, best);
    cudaGraphExecDestroy(xm[0]); cudaGraphExecDestroy(xm[1]);
  }
  { const int n8[2]={N_SML,20480}; const char* t8[2]={"10KB","40KB"};
    for(int k=0;k<2;k++){
      int n=n8[k]; char eb[256]="";
      printf("-- %s: verify graphs (%d bf16-feedback iterations), both launch orders --\n", t8[k], 8);
      cudaGraphExec_t xv0=build_spin_graph(0,s0,0,m0,qs0,acc0,n,8,1,false,eb,sizeof eb);
      cudaGraphExec_t xv1=build_spin_graph(1,s1,1,m1,qs1,acc1,n,8,1,false,eb,sizeof eb);
      bool vOK=false;
      if(!xv0||!xv1) printf("%s verify graphs: capture failed: %s\n", t8[k], eb);
      else {
        double u1=graph_pair_run(xv0,xv1,true ,n,8,1,true);
        double u2=graph_pair_run(xv0,xv1,false,n,8,1,true);
        vOK = (u1>0)&&(u2>0);
        printf("B-then-A: %s ; A-then-B: %s (watchdog/timeout seen: %s)\n",
               u1>0?"completed+verified":"DEADLOCK/FAIL", u2>0?"completed+verified":"DEADLOCK/FAIL",
               (u1>0&&u2>0)?"no":"YES");
      }
      if(xv0) cudaGraphExecDestroy(xv0); if(xv1) cudaGraphExecDestroy(xv1);
      cudaGraphExec_t xt0=build_spin_graph(0,s0,0,m0,qs0,acc0,n,AOPS,0,false,eb,sizeof eb);
      cudaGraphExec_t xt1=build_spin_graph(1,s1,1,m1,qs1,acc1,n,AOPS,0,false,eb,sizeof eb);
      cudaGraphExec_t xb0=build_spin_graph(0,s0,0,m0,qs0,acc0,n,AOPS,0,true ,eb,sizeof eb);
      cudaGraphExec_t xb1=build_spin_graph(1,s1,1,m1,qs1,acc1,n,AOPS,0,true ,eb,sizeof eb);
      if(!xt0||!xt1||!xb0||!xb1) printf("%s timing graphs: capture failed: %s\n", t8[k], eb);
      else {
        auto time_pair=[&](cudaGraphExec_t xa, cudaGraphExec_t xb)->float{
          float bst=1e30f;
          for(int r=0;r<12;r++){                              // long sequence: test DVFS ramp-up
            double us=graph_pair_run(xa,xb,true,n,AOPS,0,false);
            if(r==0||r==4||r==11) printf("      replay round %2d: %7.2f us/op\n", r, us);
            if(us<0) return -1.f;
            if((float)us<bst) bst=(float)us;
          }
          return bst;
        };
        float usT=time_pair(xt0,xt1), usB=time_pair(xb0,xb1);
        float best = (usB>0 && (usT<=0 || usB<usT)) ? usB : usT;
        // hot-clock control: dense compute burn on both GPUs immediately before each replay,
        // to test whether the slow in-graph number is a low-DVFS-P-state artifact
        float hot=1e30f;
        for(int r=0;r<3;r++){
          cudaSetDevice(0); k_burn<<<68,256,0,s0>>>(sc0,300000000LL); CK(cudaGetLastError());
          cudaSetDevice(1); k_burn<<<68,256,0,s1>>>(sc1,300000000LL); CK(cudaGetLastError());
          double us=graph_pair_run(xt0,xt1,true,n,AOPS,0,false);
          if(r==0||r==2) printf("      hot-clock replay %d: %7.2f us/op\n", r, us);
          if(us<0) break;
          if((float)us<hot) hot=(float)us;
        }
        g_q8_ok[k]=vOK&&(best>0); g_q8_us[k]=best; g_q8_ms[k]=best*AOPS/1000.f;
        g_q8_hot_us[k]=hot;
        dump_dbg("in-graph spin-wait (last round)", AOPS, best);
        printf("GRAPH spin %s x%d: tight %.2f us/op, nanosleep %.2f, hot-clock %.2f -> best %.3f ms per %d -> SPIN_ALLREDUCE_IN_GRAPHS=%s\n",
               t8[k], AOPS, usT, usB, hot, g_q8_ms[k], AOPS, g_q8_ok[k]?"WORKS":"DEADLOCK/FAIL");
      }
      if(xt0) cudaGraphExecDestroy(xt0); if(xt1) cudaGraphExecDestroy(xt1);
      if(xb0) cudaGraphExecDestroy(xb0); if(xb1) cudaGraphExecDestroy(xb1);
    }
  }

  // ================= summary =================
  printf("\n===== SUMMARY (QUESTION=ANSWER) =====\n");
  printf("Q1 PEER_ACCESS=%s\n", peerOK?"ENABLED":"UNSUPPORTED");
  const char* ptag = peerOK?"PEER":"PEERAPI";
  for(int k=0;k<2;k++){
    if(g_peer_us[k][0]>0){
      printf("Q2 %s_0TO1_%s=%.2fus_%.2fGBps\n", ptag, g_sztag[k], g_peer_us[k][0], gbps(g_bytes[k],g_peer_us[k][0]));
      printf("Q2 %s_1TO0_%s=%.2fus_%.2fGBps\n", ptag, g_sztag[k], g_peer_us[k][1], gbps(g_bytes[k],g_peer_us[k][1]));
    }
    if(g_stg_us[k][0]>0){
      printf("Q2 STAGED_0TO1_%s=%.2fus_%.2fGBps\n", g_sztag[k], g_stg_us[k][0], gbps(g_bytes[k],g_stg_us[k][0]));
      printf("Q2 STAGED_1TO0_%s=%.2fus_%.2fGBps\n", g_sztag[k], g_stg_us[k][1], gbps(g_bytes[k],g_stg_us[k][1]));
    }
  }
  printf("Q3 PEER_MAPPED_WRITES=%s", !peerOK?"SKIPPED":(pm_ok?"WORKS":"FAIL"));
  if(g_pp_us>0) printf(" pingpong10KB=%.2fus", g_pp_us);
  printf("\n");
  printf("Q4a CROSS_DEVICE_EVENT_STREAM_SYNC=%s\n", cdsync?"WORKS":"FAIL");
  printf("Q4b GRAPHS_CROSS_DEVICE_EVENTS=%s\n", graphsOk?"WORKS":"UNSUPPORTED");
  if(g_eager_stg_us>0) printf("Q5 ALLREDUCE10KB_EAGER_STAGED=%.2fus_op %.3fms_per_128\n", g_eager_stg_us, g_eager_stg_us*AOPS/1000.f);
  if(g_eager_peer_us>0) printf("Q5 ALLREDUCE10KB_EAGER_%s=%.2fus_op %.3fms_per_128\n", ptag, g_eager_peer_us, g_eager_peer_us*AOPS/1000.f);
  if(g_graph_us>0)     printf("Q5 ALLREDUCE10KB_GRAPH_STAGED=%.2fus_op %.3fms_per_128\n", g_graph_us, g_graph_us*AOPS/1000.f);
  else                 printf("Q5 ALLREDUCE10KB_GRAPH=NOT_AVAILABLE_OR_UNSAFE\n");
  printf("Q6 MAPPED_PINNED_CROSS_VISIBLE=%s\n", g_map_ok?"WORKS":"FAIL");
  if(g_map_ok){
    { const char* t7[2]={"10KB","40KB"};
      for(int k=0;k<2;k++){
        if(g_q7_us[k][0]>0) printf("Q7 SPIN_ALLREDUCE_EAGER_%s=%.2fus_op %.3fms_per_128 (staged baseline 126.24us)\n",
                                   t7[k], g_q7_us[k][0], g_q7_us[k][0]*AOPS/1000.f);
        if(g_q7_us[k][1]>0) printf("Q7 SPIN_ALLREDUCE_EAGER_%s_NSLEEP=%.2fus_op %.3fms_per_128\n",
                                   t7[k], g_q7_us[k][1], g_q7_us[k][1]*AOPS/1000.f);
      }
    }
    { const char* t8[2]={"10KB","40KB"};
      for(int k=0;k<2;k++){
        if(g_q8_us[k]>0) printf("Q8 SPIN_ALLREDUCE_IN_GRAPHS_%s=%s %.2fus_op %.3fms_per_128 hot_clock=%.2fus_op\n",
                                t8[k], g_q8_ok[k]?"WORKS":"DEADLOCK", g_q8_us[k], g_q8_ms[k], g_q8_hot_us[k]);
        else             printf("Q8 SPIN_ALLREDUCE_IN_GRAPHS_%s=NOT_CAPTURED\n", t8[k]);
      }
    }
    if(g_q8_node_us>0)  printf("Q8 GRAPH_TINY_NODE_SCHED=%.2fus_per_node (128 chained nodes/graph, no comms)\n", g_q8_node_us);
    if(g_q8_mnode_us>0) printf("Q8 GRAPH_MAPPEDWRITE_NODE_SCHED=%.2fus_per_node (same, but kernels write mapped host mem)\n", g_q8_mnode_us);
    printf("Q9 GRAPH_LAUNCH_10NODE_SUBMISSION_dev0=%.2fus dev1=%.2fus\n", g_q9_us[0], g_q9_us[1]);
    { const int n10[5]={5120,20480,81920,327680,1310720}; const char* t10[5]={"10KB","40KB","160KB","640KB","2.5MB"};
      for(int k=0;k<5;k++){
        if(g_q10_spin_us[k]>0) printf("Q10 MAPPED_SPIN_ALLREDUCE_%s=%.2fus_op %.2fGBps_eff\n",
                                      t10[k], g_q10_spin_us[k], 2.0*(n10[k]*2)/g_q10_spin_us[k]/1000.0);
        if(g_q10_stg_us[k]>0)  printf("Q10 STAGED_ALLREDUCE_%s=%.2fus_op %.2fGBps_eff\n",
                                      t10[k], g_q10_stg_us[k], 2.0*(n10[k]*2)/g_q10_stg_us[k]/1000.0);
      }
    }
  }
  printf("total CUDA errors encountered: %d\n", g_nerr.load());
  return 0;
}
