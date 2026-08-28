// ecmcgbnprimefactorsmpi — distributed GPU trial-division prime factorization.
//
// A CUDA + CGBN counterpart to cgbnprimefactors.cu with a different strategy: it
// factors an unsigned integer N by TRIAL DIVISION against a precomputed table of
// arbitrary-precision primes read from a file, distributed three ways:
//
//   * OpenMPI  — the divisor interval [1, sqrt(N)] is split between the ranks;
//                each rank owns the file primes that fall in its value sub-range.
//   * pthreads — within a rank, the rank's prime slice is split across host
//                threads; each thread drives its own CUDA stream and buffers.
//   * CUDA/CGBN — the actual big-integer work (N mod p) runs on the GPU. One CGBN
//                instance per prime computes the remainder with cgbn_signed_rem
//                (CGBN's signed two's-complement support, cgbn_signed.h); a zero
//                remainder means p | N.
//
// The primes from the file are ASSUMED PROVEN PRIME (deterministically, not by a
// probabilistic Miller-Rabin test), so a divisor hit is recorded as a prime
// factor directly — no primality test is ever run on a file prime. Whatever
// residual cofactor survives trial division (the case of an incomplete prime
// table, or a prime factor > sqrt(N)) is handed to ECM-GMP (libecm). Only that
// residual is primality-tested; the file primes never are.
//
// Build: make ecmcgbnprimefactorsmpi
// Run:   mpirun -np <ranks> ./ecmcgbnprimefactorsmpi <primes-file> <N> [threads]
//        (output — the distinct prime factors — on stderr from rank 0)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <unistd.h>
#include <pthread.h>
#include <mpi.h>
#include <gmp.h>          // GMP: host bookkeeping, libecm, and CGBN's host backend
#include <ecm.h>          // GMP-ECM (libecm) — residual-cofactor fallback
#include <cuda.h>
#include <cgbn/cgbn.h>   // pulls in cgbn_signed.h (signed two's-complement wrappers)

// ---------------------------------------------------------------------------
// Parameters.
// ---------------------------------------------------------------------------
#define MAXBITS    32768             // host bignum width / largest GPU width
#define LIMBS      (MAXBITS / 32)     // 32-bit limbs in a host BN

#define MOD_BLOCK   128U             // threads/block (a multiple of every TPI used)
#define MOD_BATCH   (1U << 16)       // primes per GPU launch (caps device memory)
#define MR_ROUNDS   40               // Miller-Rabin rounds for the RESIDUAL only

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

// MPI topology, filled in by main(). mpi_nranks == 1 (or no mpirun) searches the
// whole [1, sqrt(N)] interval on one rank — identical to the serial case.
static int32_t mpi_rank   = 0;
static int32_t mpi_nranks = 1;

// ===========================================================================
// Host bignum: fixed MAXBITS little-endian uint32 limbs. Stores primes / N and
// does decimal I/O; a thin bridge converts to/from GMP mpz (for sqrt, interval
// arithmetic, and libecm) and to width-specific device memory (for the GPU).
// ===========================================================================
struct BN { uint32_t l[LIMBS]; };

static void bn_zero(BN& v) { memset(v.l, 0, sizeof(v.l)); }

static bool bn_is_zero(const BN& v) {
  for (int32_t i = 0; i < LIMBS; ++i) if (v.l[i]) return false;
  return true;
}

static int32_t bn_cmp(const BN& a, const BN& b) {
  for (int32_t i = LIMBS - 1; i >= 0; --i)
    if (a.l[i] != b.l[i]) return a.l[i] < b.l[i] ? -1 : 1;
  return 0;
}

static uint32_t bn_bitlen(const BN& v) {
  for (int32_t i = LIMBS - 1; i >= 0; --i)
    if (v.l[i]) return (uint32_t) i * 32 + (32 - __builtin_clz(v.l[i]));
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

static BN bn_from_dec(const char* s) {
  BN v; bn_zero(v);
  for (; *s; ++s) {
    if (*s < '0' || *s > '9') continue;
    uint64_t carry = (uint64_t) (*s - '0');
    for (int32_t i = 0; i < LIMBS; ++i) {
      uint64_t cur = (uint64_t) v.l[i] * 10 + carry;
      v.l[i] = (uint32_t) cur;
      carry = cur >> 32;
    }
    if (carry) {
      fprintf(stderr, "error: value exceeds the %d-bit width\n", MAXBITS);
      exit(1);
    }
  }
  return v;
}

static std::string bn_to_dec(BN v) {
  if (bn_is_zero(v)) return "0";
  std::string s;
  while (!bn_is_zero(v)) s.push_back((char) ('0' + bn_divmod_u32(v, 10)));
  std::reverse(s.begin(), s.end());
  return s;
}

// BN <-> GMP mpz (via decimal — simple and exact; used off the hot path).
static void bn_to_mpz(const BN& v, mpz_t out) {
  std::string s = bn_to_dec(v);
  mpz_set_str(out, s.c_str(), 10);
}

static BN mpz_to_bn(const mpz_t v) {
  char* s = mpz_get_str(nullptr, 10, v);
  BN b = bn_from_dec(s);
  free(s);
  return b;
}

// BN -> width-specific device memory (value fits in BITS, so upper limbs zero).
template <uint32_t BITS>
static void bn_to_mem(const BN& v, cgbn_mem_t<BITS>& m) {
  const int32_t LB = BITS / 32;
  for (int32_t i = 0; i < LB; ++i) m._limbs[i] = (i < LIMBS) ? v.l[i] : 0;
}

// ===========================================================================
// GPU trial-division kernel — one CGBN instance per prime computes N mod p.
// ===========================================================================
template <uint32_t BITS, uint32_t TPI>
__global__ void
mod_kernel(cgbn_error_report_t* report, cgbn_mem_t<BITS>* g_N,
           cgbn_mem_t<BITS>* g_primes, uint32_t count, uint32_t* g_div) {
  typedef cgbn_context_t<TPI>         context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;

  int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= (int32_t) count) return;

  context_t ctx(cgbn_report_monitor, report, instance);
  env_t env = ctx.template env<env_t>();

  typename env_t::cgbn_t N, p, r;
  cgbn_load(env, N, g_N);
  cgbn_load(env, p, &g_primes[instance]);

  // N and p are positive and the width carries >=16 bits of headroom, so the
  // sign bit is clear and the signed remainder equals the true remainder. Using
  // the signed API exercises CGBN's signed two's-complement support directly.
  cgbn_signed_rem(env, r, N, p);

  uint32_t isdiv = cgbn_equals_ui32(env, r, 0) ? 1u : 0u;
  if ((threadIdx.x % TPI) == 0) g_div[instance] = isdiv;   // lane 0 writes
}

// ===========================================================================
// pthread trial-division worker — one host thread per prime slice, each with its
// own CUDA stream and device buffers, launching mod_kernel in MOD_BATCH chunks.
// ===========================================================================
struct WorkerArg {
  const BN*  N;
  const BN*  primes;      // base of the rank's prime array
  size_t     start, end;  // this thread's half-open slice [start, end)
  std::vector<BN>* out;   // thread-local: primes found to divide N
};

template <uint32_t BITS, uint32_t TPI>
static void* worker(void* varg) {
  WorkerArg* a = (WorkerArg*) varg;
  const size_t total = a->end - a->start;
  if (total == 0) return nullptr;

  const uint32_t cap = (total < MOD_BATCH) ? (uint32_t) total : MOD_BATCH;

  cgbn_error_report_t* report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  cgbn_mem_t<BITS>* d_N = nullptr;
  cgbn_mem_t<BITS>* d_primes = nullptr;
  uint32_t*         d_div = nullptr;
  CUDA_CHECK(cudaMalloc(&d_N,      sizeof(cgbn_mem_t<BITS>)));
  CUDA_CHECK(cudaMalloc(&d_primes, (size_t) cap * sizeof(cgbn_mem_t<BITS>)));
  CUDA_CHECK(cudaMalloc(&d_div,    (size_t) cap * sizeof(uint32_t)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cgbn_mem_t<BITS> hN;
  bn_to_mem<BITS>(*a->N, hN);
  CUDA_CHECK(cudaMemcpyAsync(d_N, &hN, sizeof(hN), cudaMemcpyHostToDevice,
                             stream));

  std::vector<cgbn_mem_t<BITS>> hp(cap);
  std::vector<uint32_t>         hdiv(cap);

  for (size_t off = 0; off < total; off += cap) {
    uint32_t n = (uint32_t) std::min((size_t) cap, total - off);

    for (uint32_t i = 0; i < n; ++i)
      bn_to_mem<BITS>(a->primes[a->start + off + i], hp[i]);
    CUDA_CHECK(cudaMemcpyAsync(d_primes, hp.data(),
                               (size_t) n * sizeof(cgbn_mem_t<BITS>),
                               cudaMemcpyHostToDevice, stream));

    uint32_t threads = n * TPI;
    uint32_t blocks  = (threads + MOD_BLOCK - 1) / MOD_BLOCK;
    mod_kernel<BITS, TPI><<<blocks, MOD_BLOCK, 0, stream>>>(report, d_N,
                                                            d_primes, n, d_div);
    CUDA_CHECK(cudaMemcpyAsync(hdiv.data(), d_div,
                               (size_t) n * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CGBN_CHECK(report);

    for (uint32_t i = 0; i < n; ++i)
      if (hdiv[i]) a->out->push_back(a->primes[a->start + off + i]);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_N));
  CUDA_CHECK(cudaFree(d_primes));
  CUDA_CHECK(cudaFree(d_div));
  CUDA_CHECK(cgbn_error_report_free(report));
  return nullptr;
}

// Split this rank's prime slice across `nthreads` pthreads at width BITS/TPI.
template <uint32_t BITS, uint32_t TPI>
static void trial_divide(const BN& N, const std::vector<BN>& primes,
                         int32_t nthreads, std::vector<BN>& found) {
  const size_t np = primes.size();
  if (np == 0) return;
  if (nthreads < 1) nthreads = 1;
  if ((size_t) nthreads > np) nthreads = (int) np;

  std::vector<pthread_t>            tids(nthreads);
  std::vector<WorkerArg>           args(nthreads);
  std::vector<std::vector<BN>>     outs(nthreads);

  const size_t chunk = (np + nthreads - 1) / nthreads;
  for (int32_t t = 0; t < nthreads; ++t) {
    size_t start = (size_t) t * chunk;
    size_t end   = std::min(start + chunk, np);
    if (start > np) start = np;
    args[t] = WorkerArg{ &N, primes.data(), start, end, &outs[t] };
    pthread_create(&tids[t], nullptr, worker<BITS, TPI>, &args[t]);
  }
  for (int32_t t = 0; t < nthreads; ++t) {
    pthread_join(tids[t], nullptr);
    found.insert(found.end(), outs[t].begin(), outs[t].end());
  }
}

// One engine per CGBN width (smallest first), TPI packing limbs one-per-thread up
// to 1024 bits then holding at 32. All N mod p run against the same N, so the
// width is chosen ONCE from N's bit-length, not per prime.
#define WIDTH_LIST(X)   \
  X(128,   4)           \
  X(256,   8)           \
  X(512,   16)          \
  X(1024,  32)          \
  X(2048,  32)          \
  X(4096,  32)          \
  X(8192,  32)          \
  X(16384, 32)          \
  X(32768, 32)

// Dispatch to the narrowest width whose bit-length fits N with 16 bits of
// headroom (so cgbn_signed_rem sees a clear sign bit). main() caps N at
// MAXBITS-16, so the widest case always matches.
static void trial_divide_dispatch(const BN& N, const std::vector<BN>& primes,
                                  int32_t nthreads, std::vector<BN>& found) {
  uint32_t b = bn_bitlen(N);
  #define X(BITS, TPI)                                      \
    if (b <= (uint32_t)(BITS) - 16U) {                      \
      trial_divide<BITS, TPI>(N, primes, nthreads, found);  \
      return;                                               \
    }
  WIDTH_LIST(X)
  #undef X
  trial_divide<32768, 32>(N, primes, nthreads, found);
}

// ===========================================================================
// Distinct-factor bookkeeping.
// ===========================================================================
static std::vector<BN> Factors;

static void record_factor(const BN& p) {
  for (const BN& f : Factors) if (bn_cmp(f, p) == 0) return;
  Factors.push_back(p);
}

static void record_factor_mpz(const mpz_t p) {
  BN b = mpz_to_bn(p);
  record_factor(b);
}

// ===========================================================================
// ECM-GMP residual fallback. Curves follow GMP-ECM's tables (row = a stage-1
// bound B1 and how many curves to try before escalating), targeting ~15..70
// digit factors. Single-threaded: the residual is a rare, small tail of the work.
// ===========================================================================
typedef struct { double B1; unsigned long curves; } ecm_level_t;

static const ecm_level_t ECM_LADDER[] = {
  {        2000.0,     30UL },  {       11000.0,     90UL },
  {       50000.0,    300UL },  {      250000.0,    700UL },
  {     1000000.0,   1800UL },  {     3000000.0,   5100UL },
  {    11000000.0,  10600UL },  {    43000000.0,  19300UL },
  {   110000000.0,  49000UL },  {   260000000.0, 124000UL },
  {   850000000.0, 260000UL },  {  2900000000.0, 520000UL },
};

static const size_t ECM_NLEVELS = sizeof(ECM_LADDER) / sizeof(ECM_LADDER[0]);

static bool ecm_find_factor(mpz_t D, const mpz_t N) {
  ecm_params params;
  ecm_init(params);
  params->verbose = 0;
  gmp_randseed_ui(params->rng, 0x9E3779B9UL);

  mpz_t n, d;                        // ecm_factor's N arg is non-const
  mpz_init_set(n, N);
  mpz_init(d);

  bool found = false;
  for (size_t lvl = 0; lvl < ECM_NLEVELS && !found; ++lvl) {
    const double B1 = ECM_LADDER[lvl].B1;
    for (unsigned long c = 0; c < ECM_LADDER[lvl].curves; ++c) {
      params->B1done = ECM_DEFAULT_B1_DONE;   // force a fresh curve at this B1
      mpz_set_ui(params->x, 0UL);
      mpz_set_ui(params->sigma, 0UL);

      int32_t r = ecm_factor(d, n, B1, params);
      if (ECM_FACTOR_FOUND_P(r) &&
          mpz_cmp_ui(d, 1UL) > 0 && mpz_cmp(d, n) != 0) {
        mpz_set(D, d);
        found = true;
        break;
      }
    }
  }

  ecm_clear(params);
  mpz_clear(n);
  mpz_clear(d);
  return found;
}

// Fully factor a residual cofactor with ECM. The residual is NOT a file prime,
// so a Miller-Rabin primality test (MR_ROUNDS) is used to decide whether it is
// itself prime; ECM splits it otherwise, and the pieces recurse.
static void factor_residual(const mpz_t n) {
  if (mpz_cmp_ui(n, 1UL) == 0) return;

  if (mpz_probab_prime_p(n, MR_ROUNDS) > 0) {   // residual only — never a file prime
    record_factor_mpz(n);
    return;
  }

  mpz_t d, q;
  mpz_init(d);
  mpz_init(q);
  if (ecm_find_factor(d, n)) {
    mpz_divexact(q, n, d);
    factor_residual(d);
    factor_residual(q);
  } else {
    fprintf(stderr, "warning: unable to fully factor residual cofactor %s\n",
            mpz_get_str(nullptr, 10, n));
    record_factor_mpz(n);
  }
  mpz_clear(d);
  mpz_clear(q);
}

int main(int argc, char* const argv[])
{
  MPI_Init(&argc, (char***) &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nranks);

  if (argc < 3 || argc > 4) {
    if (mpi_rank == 0)
      fprintf(stderr,
              "Usage: ecmcgbnprimefactorsmpi <primes-file> <N> [threads]\n");
    MPI_Finalize();
    return 1;
  }

  const char* primes_path = argv[1];
  const char* NS          = argv[2];
  int32_t nthreads = (argc == 4)  ? atoi(argv[3])
                                  : (int32_t) sysconf(_SC_NPROCESSORS_ONLN);
  if (nthreads < 1)
    nthreads = 1;

  // mpirun replicates argv, so every rank parses N and reads the file itself —
  // no broadcast needed. Each rank later keeps only its interval's primes.
  BN N = bn_from_dec(NS);
  if (bn_bitlen(N) > MAXBITS - 16) {
    if (mpi_rank == 0)
      fprintf(stderr, "error: N too large for the %d-bit CGBN width\n", MAXBITS);
    MPI_Finalize();
    return 1;
  }

  // sqrt(N): the trial-division bound (via GMP). Bind each rank to a GPU.
  mpz_t Nz, sqrtNz;
  mpz_init(Nz);
  mpz_init(sqrtNz);
  bn_to_mpz(N, Nz);
  mpz_sqrt(sqrtNz, Nz);
  BN SQRTN = mpz_to_bn(sqrtNz);

  int32_t ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev > 0)
    CUDA_CHECK(cudaSetDevice(mpi_rank % ndev));

  // This rank's half-open value sub-interval [lo, hi) of [1, sqrt(N)]. The last
  // rank absorbs the rounding remainder so the ranks tile [1, sqrt(N)] exactly.
  mpz_t width, lo, hi, one;
  mpz_init(width);
  mpz_init(lo);
  mpz_init(hi);
  mpz_init_set_ui(one, 1UL);
  mpz_fdiv_q_ui(width, sqrtNz, (unsigned long) mpi_nranks);   // floor(sqrt(N)/ranks)
  mpz_mul_ui(lo, width, (unsigned long) mpi_rank);
  mpz_add(lo, lo, one);                                      // lo = rank*width + 1

  if (mpi_rank == mpi_nranks - 1) {
    mpz_add(hi, sqrtNz, one);                               // last rank: up to sqrt(N)
  } else {
    mpz_mul_ui(hi, width, (unsigned long) (mpi_rank + 1));
    mpz_add(hi, hi, one);                                   // hi = (rank+1)*width + 1
  }

  BN LO = mpz_to_bn(lo);
  BN HI = mpz_to_bn(hi);
  mpz_clear(width);
  mpz_clear(lo);
  mpz_clear(hi);
  mpz_clear(one);

  // Read the prime table; keep primes in [max(2,lo), hi) and <= sqrt(N). Every
  // rank reads the whole file but retains only its interval's divisors.
  std::vector<BN> primes;
  {
    std::ifstream in(primes_path);
    if (!in) {
      if (mpi_rank == 0)
        fprintf(stderr, "error: cannot open primes file '%s'\n", primes_path);
      MPI_Finalize();
      return 1;
    }
    std::string tok;
    BN two = bn_from_dec("2");
    while (in >> tok) {
      BN p = bn_from_dec(tok.c_str());
      if (bn_cmp(p, two) < 0)      continue;   // skip 0/1
      if (bn_cmp(p, SQRTN) > 0)    continue;   // beyond the trial-division bound
      if (bn_cmp(p, LO) < 0)       continue;   // below this rank's interval
      if (bn_cmp(p, HI) >= 0)      continue;   // at/above this rank's interval
      primes.push_back(p);
    }
  }

  // GPU trial division, this rank's primes split across pthreads.
  std::vector<BN> found;
  trial_divide_dispatch(N, primes, nthreads, found);

  // Gather every rank's found primes to rank 0 (each prime lives in exactly one
  // rank's interval, so there are no cross-rank duplicates).
  int32_t mycount = (int32_t) found.size();
  std::vector<int> counts(mpi_nranks, 0);
  MPI_Gather(&mycount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<BN> all;
  std::vector<int> bytecounts, displs;
  if (mpi_rank == 0) {
    bytecounts.resize(mpi_nranks);
    displs.resize(mpi_nranks);
    int32_t total = 0;
    for (int32_t i = 0; i < mpi_nranks; ++i) {
      bytecounts[i] = counts[i] * (int32_t) sizeof(BN);
      displs[i]     = total * (int32_t) sizeof(BN);
      total += counts[i];
    }
    all.resize(total);
  }
  MPI_Gatherv(found.data(), mycount * (int) sizeof(BN), MPI_BYTE,
              all.data(), mpi_rank == 0 ? bytecounts.data() : nullptr,
              mpi_rank == 0 ? displs.data() : nullptr, MPI_BYTE, 0,
              MPI_COMM_WORLD);

  // Rank 0: record the small prime factors, divide out their full multiplicity,
  // and hand any residual cofactor to ECM.
  if (mpi_rank == 0) {
    mpz_t cof, p;
    mpz_init_set(cof, Nz);
    mpz_init(p);
    for (const BN& f : all) {
      record_factor(f);
      bn_to_mpz(f, p);
      while (mpz_divisible_p(cof, p)) mpz_divexact(cof, cof, p);
    }
    if (mpz_cmp_ui(cof, 1UL) > 0) factor_residual(cof);   // prime > sqrt(N), or incomplete table
    mpz_clear(cof);
    mpz_clear(p);

    std::sort(Factors.begin(), Factors.end(),
              [](const BN& a, const BN& b) { return bn_cmp(a, b) < 0; });

    fprintf(stderr, "---------------------------------\n");
    fprintf(stderr, "Prime Factors of %s:", NS);
    for (const BN& f : Factors) fprintf(stderr, " %s", bn_to_dec(f).c_str());
    fprintf(stderr, "\n---------------------------------\n");
  }

  mpz_clear(Nz);
  mpz_clear(sqrtNz);
  MPI_Finalize();
  return 0;
}
