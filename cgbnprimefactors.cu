// cgbnprimefactors — GPU prime factorization using NVIDIA CGBN.
//
// A CUDA counterpart to the CPU `primefactors` program. It prints the DISTINCT
// prime factors of an unsigned integer (matching the CPU version's output) using
// three cooperating stages:
//
//   1. Host trial division  — peels off small prime factors (< SMALL_BOUND) with
//      plain host bignum arithmetic (no GMP).
//   2. GPU Miller-Rabin      — cgbn_modular_power over many bases in parallel to
//      decide whether a cofactor is prime.
//   3. GPU Pollard's rho     — thousands of instances, each running rho with a
//      distinct constant c, race to split a composite cofactor.
//
// The GPU layer is templated on CGBN width and dispatched at runtime to the
// NARROWEST width that fits each cofactor (128 up to 32768 bit). Profiling showed
// rho_kernel is latency-bound on integer-ALU dependency chains and its cost is
// dominated by limb count, so running a ~50-bit factor through 128-bit math
// instead of 512-bit slashes op-count and register pressure.
//
// DISTRIBUTED (OpenMPI): rho is embarrassingly parallel over the walk constant c.
// Every rank runs the IDENTICAL orchestration loop over an identical work stack —
// trial division, Miller-Rabin, and exact division are deterministic and cheap, so
// replicating them on every rank keeps the ranks in lockstep with no communication.
// The ranks diverge only inside the rho search: each rank (on its own GPU) walks a
// DISJOINT slice of the c-space (slot = attempt*nranks + rank). After each attempt
// an MPI_Allreduce(MIN) elects the lowest-ranked finder and MPI_Bcast hands its
// factor to everyone, so all ranks resume from the same split. Only rank 0 prints.
// k GPUs search k× the constants per attempt — a throughput win, matching rho's
// ~sqrt(k)-from-k-walks nature (see CLAUDE.md), not a hard-semiprime speedup.
//
// Build: make cgbnprimefactors
// Run:   mpirun -np <ranks> ./cgbnprimefactors <bit-width> <unsigned-integer>
//        (each rank binds GPU rank%deviceCount; output on stderr from rank 0)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <string>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <cuda.h>
#include <gmp.h>          // CGBN's host backend (cgbn_mpz.h) requires GMP
#include <cgbn/cgbn.h>

// ---------------------------------------------------------------------------
// Parameters.
// ---------------------------------------------------------------------------
#define MAXBITS    32768             // host bignum width / largest GPU width
#define LIMBS      (MAXBITS / 32)     // 32-bit limbs in a host BN

#define RHO_WARPS     8192U          // target warp count per rho launch; each
                                     // engine sets instances = RHO_WARPS*32/TPI
                                     // so narrow (small-TPI) widths stay saturated
#define RHO_BLOCK     128U           // threads/block (multiple of every TPI used)
#define SMALL_BOUND   100000U        // host trial-division bound for small primes

// Deterministic-ish Miller-Rabin bases (same guarantee class as the CPU version's
// mpz_probab_prime_p — probabilistic for very large inputs).
static const uint32_t MR_BASES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
static const uint32_t MR_NBASES  = sizeof(MR_BASES) / sizeof(MR_BASES[0]);

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_e));                                          \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define CGBN_CHECK(report)                                                      \
  do {                                                                          \
    if (cgbn_error_report_check(report)) {                                      \
      fprintf(stderr, "CGBN error: %s\n", cgbn_error_string(report));           \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// MPI topology, filled in by main(). mpi_nranks == 1 with a single (or no) mpirun
// process, in which case the code runs exactly like the serial version.
static int32_t mpi_rank   = 0;
static int32_t mpi_nranks = 1;

// ===========================================================================
// Host bignum: fixed 512-bit little-endian uint32 limbs. Handles all decimal
// I/O and small-factor trial division; GPU does the heavy arithmetic.
// ===========================================================================
struct BN { uint32_t l[LIMBS]; };

static void bn_zero(BN& v) { memset(v.l, 0, sizeof(v.l)); }
static void bn_from_u32(BN& v, uint32_t x) { bn_zero(v); v.l[0] = x; }

static bool bn_is_zero(const BN& v) {
  for (int32_t i = 0; i < LIMBS; ++i)
    if (v.l[i])
      return false;

  return true;
}

static bool bn_is_one(const BN& v) {
  if (v.l[0] != 1)
    return false;

  for (int32_t i = 1; i < LIMBS; ++i)
    if (v.l[i])
      return false;

  return true;
}

static int32_t bn_cmp(const BN& a, const BN& b) {
  for (int32_t i = LIMBS - 1; i >= 0; --i)
    if (a.l[i] != b.l[i])
      return a.l[i] < b.l[i] ? -1 : 1;

  return 0;
}

static uint32_t bn_bitlen(const BN& v) {
  for (int32_t i = LIMBS - 1; i >= 0; --i)
    if (v.l[i])
      return (uint32_t) i * 32 + (32 - __builtin_clz(v.l[i]));

  return 0;
}

static uint32_t bn_divmod_u32(BN& v, uint32_t d) {
  uint64_t rem = 0;
  for (int32_t i = LIMBS - 1; i >= 0; --i) {
    uint64_t cur = (rem << 32) | v.l[i];
    v.l[i] = (uint32_t) (cur / d);
    rem = cur % d;
  }
  return (uint32_t) rem;
}

static uint32_t bn_mod_u32(const BN& v, uint32_t d) {
  uint64_t rem = 0;
  for (int32_t i = LIMBS - 1; i >= 0; --i)
    rem = ((rem << 32) | v.l[i]) % d;

  return (uint32_t) rem;
}

static BN bn_from_dec(const char* s) {
  BN v; bn_zero(v);
  for (; *s; ++s) {
    if (*s < '0' || *s > '9')
      continue;

    uint64_t carry = (uint64_t) (*s - '0');
    for (int32_t i = 0; i < LIMBS; ++i) {
      uint64_t cur = (uint64_t) v.l[i] * 10 + carry;
      v.l[i] = (uint32_t) cur;
      carry = cur >> 32;
    }

    if (carry) {
      (void) fprintf(stderr, "error: input exceeds the %i-bit width\n", MAXBITS);
      exit(1);
    }
  }
  return v;
}

static std::string bn_to_dec(BN v) {
  if (bn_is_zero(v))
    return "0";

  std::string s;
  while (!bn_is_zero(v))
    s.push_back((char) ('0' + bn_divmod_u32(v, 10)));

  std::reverse(s.begin(), s.end());
  return s;
}

// BN <-> width-specific device memory (value fits in BITS, so upper limbs zero).
template <uint32_t BITS>
static void bn_to_mem(const BN& v, cgbn_mem_t<BITS>& m) {
  const int32_t LB = BITS / 32;

  for (int32_t i = 0; i < LB; ++i)
    m._limbs[i] = (i < LIMBS) ? v.l[i] : 0;
}

template <uint32_t BITS>
static void bn_from_mem(const cgbn_mem_t<BITS>& m, BN& v) {
  int32_t LB = BITS / 32;
  bn_zero(v);

  for (int32_t i = 0; i < LB && i < LIMBS; ++i)
    v.l[i] = m._limbs[i];
}

// ===========================================================================
// Device kernels — templated on CGBN width (BITS) and threads/instance (TPI).
// ===========================================================================
template <uint32_t BITS, uint32_t TPI>
__device__ __forceinline__ void
rho_step(cgbn_env_t<cgbn_context_t<TPI>, BITS> env,
         typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t& r,
         const typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t& x,
         uint32_t c,
         const typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t& M) {
  typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_wide_t w;
  cgbn_sqr_wide(env, w, x);
  cgbn_rem_wide(env, r, w, M);
  cgbn_add_ui32(env, r, r, c);

  if (cgbn_compare(env, r, M) >= 0)
    cgbn_sub(env, r, r, M);
}

// Parallel Pollard's rho (Floyd tortoise/hare): instance i walks M with
// c = c_base+i+1; the first to find 1 < gcd < M claims the shared result slot.
//
// Three "improvements" were implemented, profiled, and REVERTED as net-slower
// here (all recorded in CLAUDE.md):
//   * Brent's batched-gcd variant — ~2.6x slower: cgbn_gcd is not the bottleneck,
//     and its serial q = q*|x-y| product chain worsened the latency stall.
//   * Per-instance ILP (K independent walks) — 2-4x slower: with 65536 instances
//     the walk-level parallelism is already saturated, and packing walks into an
//     instance only raises register pressure.
//   * __launch_bounds__ register-capping to raise occupancy — monotonically
//     slower (75%->100% occ = 8.6s->11s): the compiler's default 52-reg / 75%
//     point is optimal; forcing more warps costs more than it buys. The kernel
//     is throughput-bound at the default allocation, not occupancy-limited.
template <uint32_t BITS, uint32_t TPI>
__global__ void
rho_kernel(cgbn_error_report_t* report, cgbn_mem_t<BITS>* g_M, uint32_t count,
           uint32_t max_iters, uint32_t c_base, int* g_found,
           cgbn_mem_t<BITS>* g_factor) {
  typedef cgbn_context_t<TPI>         context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;

  int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= (int32_t) count)
    return;

  context_t ctx(cgbn_report_monitor, report, instance);
  env_t env = ctx.template env<env_t>();

  typename env_t::cgbn_t M, x, y, diff, g;
  cgbn_load(env, M, g_M);
  cgbn_set_ui32(env, x, 2);
  cgbn_set_ui32(env, y, 2);

  uint32_t c = c_base + (uint32_t) instance + 1;

  for (uint32_t i = 0; i < max_iters; ++i) {
    if ((i & 4095) == 0 && *(volatile int*) g_found) return;

    rho_step<BITS, TPI>(env, x, x, c, M);      // x = f(x)
    rho_step<BITS, TPI>(env, y, y, c, M);      // y = f(f(y))
    rho_step<BITS, TPI>(env, y, y, c, M);

    int32_t cmp = cgbn_compare(env, x, y);
    if (cmp == 0)
      break;
    if (cmp > 0)
      cgbn_sub(env, diff, x, y);
    else
      cgbn_sub(env, diff, y, x);

    cgbn_gcd(env, g, diff, M);

    if (cgbn_equals_ui32(env, g, 1))
      continue;
    if (cgbn_compare(env, g, M) == 0)
      break;

    int32_t win = 0;
    if ((threadIdx.x % TPI) == 0)
      win = (atomicCAS(g_found, 0, 1) == 0);
    win = __shfl_sync(0xffffffff, win, (threadIdx.x / TPI) * TPI);

    if (win)
      cgbn_store(env, g_factor, g);

    return;
  }
}

// Miller-Rabin: instance i tests base g_bases[i]; a witness sets the composite
// flag. M must be odd and greater than every base.
template <uint32_t BITS, uint32_t TPI>
__global__ void
mr_kernel(cgbn_error_report_t* report, cgbn_mem_t<BITS>* g_M, uint32_t* g_bases,
          uint32_t nbases, int* g_composite) {
  typedef cgbn_context_t<TPI>         context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;

  int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= (int32_t) nbases)
    return;

  context_t ctx(cgbn_report_monitor, report, instance);
  env_t env = ctx.template env<env_t>();

  typename env_t::cgbn_t M, Mm1, d, x, a;
  cgbn_load(env, M, g_M);
  cgbn_sub_ui32(env, Mm1, M, 1);

  uint32_t s = 0;                            // M-1 = d * 2^s
  cgbn_set(env, d, Mm1);

  while ((cgbn_get_ui32(env, d) & 1) == 0) {
    cgbn_shift_right(env, d, d, 1); ++s;
  }

  cgbn_set_ui32(env, a, g_bases[instance]);
  cgbn_modular_power(env, x, a, d, M);       // x = a^d mod M
  if (cgbn_equals_ui32(env, x, 1) || cgbn_compare(env, x, Mm1) == 0)
    return;

  for (uint32_t r = 1; r < s; ++r) {
    typename env_t::cgbn_wide_t w;
    cgbn_sqr_wide(env, w, x);
    cgbn_rem_wide(env, x, w, M);
    if (cgbn_compare(env, x, Mm1) == 0)
      return;
  }

  if ((threadIdx.x % TPI) == 0)
    atomicOr(g_composite, 1);
}

// Exact quotient q = M / d on a single instance.
template <uint32_t BITS, uint32_t TPI>
__global__ void
divexact_kernel(cgbn_error_report_t* report, cgbn_mem_t<BITS>* g_M,
                cgbn_mem_t<BITS>* g_d, cgbn_mem_t<BITS>* g_q) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;

  int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= 1)
    return;

  context_t ctx(cgbn_report_monitor, report, instance);
  env_t env = ctx.template env<env_t>();

  typename env_t::cgbn_t M, d, q, r;
  cgbn_load(env, M, g_M);
  cgbn_load(env, d, g_d);
  cgbn_div_rem(env, q, r, M, d);
  cgbn_store(env, g_q, q);
}

// ===========================================================================
// Per-width GPU engine: device buffers + the three operations, all at width BITS.
// ===========================================================================
template <uint32_t BITS, uint32_t TPI>
struct Engine {
  cgbn_error_report_t* report = nullptr;
  cgbn_mem_t<BITS>* d_M = nullptr;
  cgbn_mem_t<BITS>* d_d = nullptr;
  cgbn_mem_t<BITS>* d_q = nullptr;
  cgbn_mem_t<BITS>* d_factor = nullptr;
  uint32_t* d_bases = nullptr;
  int32_t* d_flag = nullptr;

  void init() {
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    CUDA_CHECK(cudaMalloc(&d_M,      sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc(&d_d,      sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc(&d_q,      sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc(&d_factor, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc(&d_flag,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bases,  sizeof(MR_BASES)));
    CUDA_CHECK(cudaMemcpy(d_bases, MR_BASES, sizeof(MR_BASES),
                          cudaMemcpyHostToDevice));
  }

  void put(cgbn_mem_t<BITS>* dst, const BN& v) {
    cgbn_mem_t<BITS> h;
    bn_to_mem<BITS>(v, h);
    CUDA_CHECK(cudaMemcpy(dst, &h, sizeof(h), cudaMemcpyHostToDevice));
  }

  bool is_prime(const BN& M) {
    put(d_M, M);
    int32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice));

    uint32_t threads = MR_NBASES * TPI;
    uint32_t blocks  = (threads + RHO_BLOCK - 1) / RHO_BLOCK;
    mr_kernel<BITS, TPI><<<blocks, RHO_BLOCK>>>(report, d_M, d_bases, MR_NBASES,
                                                d_flag);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    int32_t composite = 0;
    CUDA_CHECK(cudaMemcpy(&composite, d_flag, sizeof(int),
                          cudaMemcpyDeviceToHost));
    return composite == 0;
  }

  // Instances scaled so warp count == RHO_WARPS regardless of TPI (a narrow
  // width with small TPI packs more instances per warp, so it needs more of
  // them to keep every SM busy).
  static constexpr uint32_t INSTANCES = RHO_WARPS * 32U / TPI;

  // One rho attempt at a given c-slot: the INSTANCES walks cover the constants
  // c in [slot*INSTANCES+1, (slot+1)*INSTANCES]. Distinct slots — across MPI
  // ranks and across escalation attempts — therefore search DISJOINT c, so no
  // two GPUs ever repeat a walk. Returns true (filling `factor`) if this GPU
  // split M; the caller (mpi_rho) escalates `iters` and advances `slot`.
  bool rho_once(const BN& M, BN& factor, uint32_t slot, uint32_t iters) {
    put(d_M, M);
    int32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice));

    uint32_t threads = INSTANCES * TPI;
    uint32_t blocks  = (threads + RHO_BLOCK - 1) / RHO_BLOCK;
    uint32_t c_base  = slot * INSTANCES;
    rho_kernel<BITS, TPI><<<blocks, RHO_BLOCK>>>(report, d_M, INSTANCES, iters,
                                                 c_base, d_flag, d_factor);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    int32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (found) {
      cgbn_mem_t<BITS> h;
      CUDA_CHECK(cudaMemcpy(&h, d_factor, sizeof(h), cudaMemcpyDeviceToHost));
      bn_from_mem<BITS>(h, factor);
      return true;
    }
    return false;
  }

  BN divexact(const BN& M, const BN& d) {
    put(d_M, M);
    put(d_d, d);
    divexact_kernel<BITS, TPI><<<1, TPI>>>(report, d_M, d_d, d_q);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    cgbn_mem_t<BITS> h;
    CUDA_CHECK(cudaMemcpy(&h, d_q, sizeof(h), cudaMemcpyDeviceToHost));
    BN q; bn_from_mem<BITS>(h, q);
    return q;
  }
};

// One engine per CGBN width, smallest first. TPI packs limbs one-per-thread up
// to 1024 bits (128/4, 256/8, 512/16, 1024/32); wider widths keep TPI=32 (the
// CGBN max) and grow to 2..32 limbs/thread. WIDTH_LIST(X) expands X(BITS,TPI,
// NAME) for every width, so it drives declarations, init, and dispatch from one
// table — add or remove a width by editing this list only.
#define WIDTH_LIST(X)   \
  X(128,   4,  E128)    \
  X(256,   8,  E256)    \
  X(512,   16, E512)    \
  X(1024,  32, E1024)   \
  X(2048,  32, E2048)   \
  X(4096,  32, E4096)   \
  X(8192,  32, E8192)   \
  X(16384, 32, E16384)  \
  X(32768, 32, E32768)

#define DECL_ENGINE(BITS, TPI, NAME) static Engine<BITS, TPI> NAME;
WIDTH_LIST(DECL_ENGINE)
#undef DECL_ENGINE

#define INIT_ENGINE(BITS, TPI, NAME) NAME.init();
static void gpu_init() {
  // Bind this MPI rank to a GPU. On a multi-GPU host each rank gets its own
  // device; when ranks outnumber GPUs (e.g. several ranks on one box) they
  // round-robin and share — still correct, just contended.
  int32_t ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev > 0) CUDA_CHECK(cudaSetDevice(mpi_rank % ndev));
  WIDTH_LIST(INIT_ENGINE)
}
#undef INIT_ENGINE

// Dispatch each cofactor to the narrowest width whose bit-length fits with 16
// bits of headroom (so rho_step's "+c", c < 2^19, cannot overflow the width).
// main() guarantees bitlen <= MAXBITS-16, so the widest case always matches.
static bool gpu_is_prime(const BN& M) {
  uint32_t b = bn_bitlen(M);
  #define X(BITS, TPI, NAME)                  \
    if (b <= (uint32_t)(BITS) - 16U)          \
      return NAME.is_prime(M);
  WIDTH_LIST(X)
  #undef X
  return E32768.is_prime(M);
}

static bool gpu_rho_once(const BN& M, BN& f, uint32_t slot, uint32_t iters) {
  uint32_t b = bn_bitlen(M);
  #define X(BITS, TPI, NAME)                    \
    if (b <= (uint32_t)(BITS) - 16U)            \
      return NAME.rho_once(M, f, slot, iters);
  WIDTH_LIST(X)
  #undef X
  return E32768.rho_once(M, f, slot, iters);
}

static BN gpu_divexact(const BN& M, const BN& d) {
  uint32_t b = bn_bitlen(M);   // M is the larger operand => picks the right width
  #define X(BITS, TPI, NAME)                    \
    if (b <= (uint32_t)(BITS) - 16U)            \
      return NAME.divexact(M, d);
  WIDTH_LIST(X)
  #undef X
  return E32768.divexact(M, d);
}

// ===========================================================================
// Factorization orchestration.
// ===========================================================================
static std::vector<BN> Factors;   // distinct primes

static void record_factor(const BN& p) {
  for (const BN& f : Factors) if (bn_cmp(f, p) == 0) return;
  Factors.push_back(p);
}

// Distributed Pollard's rho. Each rank searches a disjoint c-slice per attempt
// (slot = attempt*nranks + rank) on its own GPU; ranks synchronize after every
// attempt. The lowest-ranked finder wins (MPI_Allreduce MIN over the candidate
// rank, INT_MAX = "found nothing") and MPI_Bcast hands its factor to all ranks,
// so every rank returns the SAME factor and their work stacks stay identical.
// With mpi_nranks == 1 this degenerates to the plain 7-attempt escalating search.
static bool mpi_rho(const BN& C, BN& factor) {
  uint32_t iters = 200000U;
  for (int32_t attempt = 0; attempt < 7; ++attempt) {
    uint32_t slot = (uint32_t) attempt * (uint32_t) mpi_nranks + (uint32_t) mpi_rank;

    BN local; bn_zero(local);
    int32_t found = gpu_rho_once(C, local, slot, iters) ? 1 : 0;

    int32_t cand   = found ? mpi_rank : INT_MAX;   // elect lowest-ranked finder
    int32_t winner = INT_MAX;
    MPI_Allreduce(&cand, &winner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (winner != INT_MAX) {
      factor = local;   // valid on the winner; the Bcast fills every other rank
      MPI_Bcast(&factor, sizeof(BN), MPI_BYTE, winner, MPI_COMM_WORLD);
      return true;
    }
    iters *= 3;         // no rank split it — escalate walk length in lockstep
  }
  return false;
}

int main(int argc, char* const argv[])
{
  MPI_Init(&argc, (char***) &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nranks);

  if (argc != 3) {
    if (mpi_rank == 0)
      fprintf(stderr, "Usage: cgbnprimefactors <bit-width> <unsigned-integer>\n");
    MPI_Finalize();
    return 1;
  }

  // mpirun replicates argv to every rank, so each parses the input identically —
  // no broadcast needed. Trial division below is likewise deterministic, so all
  // ranks reach the GPU stage with the same working value and factor list.
  const char* NS = argv[2];
  BN N = bn_from_dec(NS);

  // Cap below MAXBITS so the widest rho_step's "+c" cannot overflow the width.
  if (bn_bitlen(N) > MAXBITS - 16) {
    if (mpi_rank == 0)
      fprintf(stderr, "error: input too large for the %d-bit CGBN width\n", MAXBITS);
    MPI_Finalize();
    return 1;
  }

  gpu_init();

  BN M = N;

  // Stage 1 (host): strip small prime factors. Every cofactor left is odd with
  // all prime factors > SMALL_BOUND (hence greater than any Miller-Rabin base).
  if (!bn_is_zero(M) && bn_mod_u32(M, 2) == 0) {
    BN two; bn_from_u32(two, 2); record_factor(two);
    while (bn_mod_u32(M, 2) == 0) bn_divmod_u32(M, 2);
  }
  for (uint32_t p = 3; p <= SMALL_BOUND; p += 2) {
    if (bn_is_one(M)) break;
    if (bn_mod_u32(M, p) == 0) {
      BN bp; bn_from_u32(bp, p); record_factor(bp);
      while (bn_mod_u32(M, p) == 0) bn_divmod_u32(M, p);
    }
  }

  // Stages 2 & 3 (GPU): split remaining cofactor(s) with Miller-Rabin + rho, each
  // dispatched to the narrowest CGBN width that fits.
  std::vector<BN> stack;
  if (!bn_is_one(M) && !bn_is_zero(M)) stack.push_back(M);

  while (!stack.empty()) {
    BN C = stack.back(); stack.pop_back();

    if (gpu_is_prime(C)) { record_factor(C); continue; }

    BN d;
    if (mpi_rho(C, d)) {                    // ranks cooperate; all get the same d
      BN q = gpu_divexact(C, d);
      stack.push_back(d);
      stack.push_back(q);
    } else {
      if (mpi_rank == 0)
        fprintf(stderr, "warning: unable to fully factor composite %s\n",
                bn_to_dec(C).c_str());
      record_factor(C);
    }
  }

  std::sort(Factors.begin(), Factors.end(),
            [](const BN& a, const BN& b) { return bn_cmp(a, b) < 0; });

  // Every rank holds the identical factor list; only rank 0 prints it.
  if (mpi_rank == 0) {
    fprintf(stderr, "---------------------------------\n");
    fprintf(stderr, "Prime Factors of %s:", NS);
    for (const BN& f : Factors) fprintf(stderr, " %s", bn_to_dec(f).c_str());
    fprintf(stderr, "\n---------------------------------\n");
  }

  MPI_Finalize();
  return 0;
}
