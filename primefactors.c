#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <gmp.h>
#include <ecm.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int32_t omp_get_thread_num(void) { return 0; }
#endif

uint32_t Bits = 128U;
mpz_t* Factors = NULL;
uint64_t NFactors = 0UL;
uint64_t FactorSize = 0UL;

// Trial-divide by small odd primes up to this bound before switching to
// Pollard's rho for the remaining (large-prime) part.
#define TRIAL_BOUND (1UL << 20)

// Rounds of Miller-Rabin for probabilistic primality (error < 4^-MR_ROUNDS).
#define MR_ROUNDS 40

// Cap on total Pollard's rho iterations (across c-retries) before giving up and
// falling back to ECM. Rho's expected cost to peel off a prime factor p is
// ~1.18*sqrt(p) iterations, so this cap covers factors up to ~10^13.
#define RHO_MAX_ITERS (1UL << 23)

// Per-level ECM schedule (defined near ecm_find_factor as ECM_LADDER): each row
// is a stage-1 bound B1 and the number of curves to run at it before escalating.

// Record F as a distinct prime factor, growing the array if needed. Duplicates
// are ignored, so the output stays a set of distinct primes.
static void record_factor(const mpz_t F) {
  for (uint64_t I = 0; I < NFactors; ++I)
    if (mpz_cmp(Factors[I], F) == 0)
      return;

  if (NFactors == FactorSize) {
    FactorSize *= 2UL;
    Factors = (mpz_t*) realloc(Factors, FactorSize * sizeof(mpz_t));
  }

  mpz_init_set(Factors[NFactors], F);
  ++NFactors;
}

// Pollard's rho with Floyd cycle detection and g(x) = (x^2 + c) mod N. Stores a
// nontrivial divisor of the odd composite N into D, retrying with successive
// values of c (a failed run collapses to gcd == N). Returns false if the total
// iteration budget RHO_MAX_ITERS is exhausted without a split, so the caller can
// escalate to ECM.
static bool pollard_rho(mpz_t D, const mpz_t N) {
  if (mpz_even_p(N)) {
    mpz_set_ui(D, 2UL);
    return true;
  }

  mpz_t X, Y, C, DIFF;
  mpz_inits(X, Y, C, DIFF, NULL);

  bool found = false;
  uint64_t iters = 0UL;

  for (unsigned long ci = 1UL; !found && iters < RHO_MAX_ITERS; ++ci) {
    mpz_set_ui(X, 2UL);
    mpz_set_ui(Y, 2UL);
    mpz_set_ui(C, ci);
    mpz_set_ui(D, 1UL);

    while (mpz_cmp_ui(D, 1UL) == 0 && iters < RHO_MAX_ITERS) {
      // X = g(X); Y = g(g(Y))  (tortoise and hare)
      mpz_mul(X, X, X); mpz_add(X, X, C); mpz_mod(X, X, N);
      mpz_mul(Y, Y, Y); mpz_add(Y, Y, C); mpz_mod(Y, Y, N);
      mpz_mul(Y, Y, Y); mpz_add(Y, Y, C); mpz_mod(Y, Y, N);

      mpz_sub(DIFF, X, Y);
      mpz_abs(DIFF, DIFF);
      mpz_gcd(D, DIFF, N);
      ++iters;
    }

    // A nontrivial factor is 1 < D < N; D == N means this c failed, retry.
    if (mpz_cmp_ui(D, 1UL) != 0 && mpz_cmp(D, N) != 0)
      found = true;
  }

  mpz_clears(X, Y, C, DIFF, NULL);
  return found;
}

// Shared "stop now" flag for the parallel ECM search. GMP-ECM polls it through
// each curve's stop_asap callback and abandons an in-flight curve once a factor
// has been found on any thread.
static volatile int32_t g_ecm_stop = 0;

static int32_t ecm_stop_asap(void) {
  return g_ecm_stop;
}

// ECM schedule: stage-1 bound B1 and how many curves to attempt at that bound
// before escalating. Bounds and curve counts follow GMP-ECM's tables — each row
// targets a factor size (~15 up to ~70 digits), and `curves` is roughly the
// expected number of curves to find a factor of that size, so the search only
// escalates after a level has had a fair chance. Escalation still finds larger
// factors early (a d-digit factor is "small" relative to a higher level's B1),
// so these counts are effective upper bounds per level, not fixed work.
typedef struct { double B1; unsigned long curves; } ecm_level_t;

static const ecm_level_t ECM_LADDER[] = {
  {        2000.0,     30UL },  // ~15 digits
  {       11000.0,     90UL },  // ~20 digits
  {       50000.0,    300UL },  // ~25 digits
  {      250000.0,    700UL },  // ~30 digits
  {     1000000.0,   1800UL },  // ~35 digits
  {     3000000.0,   5100UL },  // ~40 digits
  {    11000000.0,  10600UL },  // ~45 digits
  {    43000000.0,  19300UL },  // ~50 digits
  {   110000000.0,  49000UL },  // ~55 digits
  {   260000000.0, 124000UL },  // ~60 digits
  {   850000000.0, 260000UL },  // ~65 digits
  {  2900000000.0, 520000UL },  // ~70 digits
};
static const size_t ECM_NLEVELS = sizeof(ECM_LADDER) / sizeof(ECM_LADDER[0]);

// Total curves across the whole ladder (the task-stream length).
static unsigned long ecm_total_curves(void) {
  unsigned long total = 0UL;
  for (size_t i = 0; i < ECM_NLEVELS; ++i)
    total += ECM_LADDER[i].curves;
  return total;
}

// Map a linear task index to the B1 bound of the ladder level it falls in.
static double ecm_task_b1(unsigned long task) {
  unsigned long acc = 0UL;
  for (size_t i = 0; i < ECM_NLEVELS; ++i) {
    acc += ECM_LADDER[i].curves;
    if (task < acc)
      return ECM_LADDER[i].B1;
  }
  return ECM_LADDER[ECM_NLEVELS - 1].B1;
}

// Elliptic Curve Method fallback (GMP-ECM), curves parallelized across cores with
// OpenMP. A single parallel region drains one shared task stream over ECM_LADDER:
// task i maps (via ecm_task_b1) to a bound, low B1 first, so cheap curves are
// claimed first, but there is NO barrier between B1 levels — a thread that
// finishes a curve immediately grabs the next task (possibly at a higher bound),
// letting cheap and expensive curves overlap and keeping every core busy. The
// first thread to split N records the factor under a critical section and trips
// g_ecm_stop so the rest abandon their in-flight curves promptly.
static bool ecm_find_factor(mpz_t D, const mpz_t N) {
  const unsigned long TOTAL_TASKS = ecm_total_curves();

  g_ecm_stop = 0;
  bool found = false;
  unsigned long next_task = 0UL;   // shared cursor into the task stream

  #pragma omp parallel
  {
    ecm_params params;
    ecm_init(params);
    params->verbose = 0;               // silence GMP-ECM's own stdout
    params->stop_asap = ecm_stop_asap; // cooperative early-out

    // Seed this thread's RNG distinctly so threads explore different curves;
    // each thread's rng then advances independently across the curves it runs.
    gmp_randseed_ui(params->rng,
                    0x9E3779B9UL * (unsigned long) (omp_get_thread_num() + 1));

    mpz_t n, d;                        // ecm_factor's N arg is non-const
    mpz_init_set(n, N);
    mpz_init(d);

    while (!g_ecm_stop) {
      unsigned long task;
      #pragma omp atomic capture
      task = next_task++;              // claim the next task

      if (task >= TOTAL_TASKS)         // stream exhausted
        break;

      double B1 = ecm_task_b1(task);

      // Force a fresh random curve at this bound: without resetting B1done the
      // next call at the same B1 would redo no stage-1 work.
      params->B1done = ECM_DEFAULT_B1_DONE;
      mpz_set_ui(params->x, 0UL);
      mpz_set_ui(params->sigma, 0UL);

      int32_t r = ecm_factor(d, n, B1, params);
      if (ECM_FACTOR_FOUND_P(r) &&
          mpz_cmp_ui(d, 1UL) > 0 && mpz_cmp(d, n) != 0) {
        #pragma omp critical
        {
          if (!g_ecm_stop) {
            mpz_set(D, d);
            found = true;
            g_ecm_stop = 1;
          }
        }
      }
    }

    ecm_clear(params);
    mpz_clear(n);
    mpz_clear(d);
  }

  return found;
}

// Recursively factor N with Miller-Rabin (is it prime?), Pollard's rho (split
// it), and ECM (fallback when rho gives up), recording each distinct prime.
static void factor(const mpz_t N) {
  if (mpz_cmp_ui(N, 1UL) == 0)
    return;

  if (mpz_probab_prime_p(N, MR_ROUNDS) > 0) {
    record_factor(N);
    return;
  }

  mpz_t D, Q;
  mpz_inits(D, Q, NULL);

  bool split = pollard_rho(D, N);
  if (!split)
    split = ecm_find_factor(D, N);

  if (!split) {
    // Neither method split this composite within its limits. Record it as-is so
    // the program terminates rather than recursing forever, and flag it.
    (void) fprintf(stderr, "warning: unable to fully factor composite ");
    (void) mpz_out_str(stderr, 10, N);
    (void) fputc('\n', stderr);
    record_factor(N);
    mpz_clears(D, Q, NULL);
    return;
  }

  mpz_divexact(Q, N, D);

  factor(D);
  factor(Q);

  mpz_clears(D, Q, NULL);
}

void prime_factors(mpz_t N) {
  mpz_t M;      // working copy of N, reduced as factors are divided out
  mpz_t SQRT;
  mpz_t NINC;
  mpz_t REM;
  mpz_t ZERO;
  mpz_t ONE;
  mpz_t TWO;
  mpz_t BOUND;

  mpz_init2(M, Bits);
  mpz_init2(SQRT, Bits);
  mpz_init2(NINC, Bits);
  mpz_init2(REM, Bits);
  mpz_init2(ZERO, Bits);
  mpz_init2(ONE, Bits);
  mpz_init2(TWO, Bits);
  mpz_init2(BOUND, Bits);

  mpz_set(M, N);
  mpz_set_ui(ZERO, 0UL);
  mpz_set_ui(ONE, 1UL);
  mpz_set_ui(TWO, 2UL);
  mpz_set_ui(NINC, 3UL);
  mpz_set_ui(BOUND, TRIAL_BOUND);

  // Divide out every power of 2, recording 2 once.
  mpz_tdiv_r(REM, M, TWO);
  if (mpz_cmp(REM, ZERO) == 0) {
    record_factor(TWO);
    do {
      mpz_tdiv_q(M, M, TWO);
      mpz_tdiv_r(REM, M, TWO);
    } while (mpz_cmp(REM, ZERO) == 0);
  }

  // Cheap phase: trial-divide by odd candidates up to min(sqrt(M), BOUND).
  // Because each factor is fully divided out, any candidate that still divides
  // M is necessarily prime, so no separate primality test is needed here.
  mpz_sqrt(SQRT, M);
  while (mpz_cmp(NINC, SQRT) <= 0 && mpz_cmp(NINC, BOUND) <= 0) {
    mpz_tdiv_r(REM, M, NINC);

    if (mpz_cmp(REM, ZERO) == 0) {
      record_factor(NINC);

      do {
        mpz_tdiv_q(M, M, NINC);
        mpz_tdiv_r(REM, M, NINC);
      } while (mpz_cmp(REM, ZERO) == 0);

      mpz_sqrt(SQRT, M);   // M shrank; tighten the search bound
    }

    mpz_add(NINC, TWO, NINC);
  }

  // Expensive phase: any residue > 1 now has only large prime factors. Hand it
  // to Miller-Rabin + Pollard's rho (which also covers M itself being prime).
  if (mpz_cmp(M, ONE) > 0)
    factor(M);

  mpz_clear(BOUND);
  mpz_clear(TWO);
  mpz_clear(ONE);
  mpz_clear(ZERO);
  mpz_clear(REM);
  mpz_clear(NINC);
  mpz_clear(SQRT);
  mpz_clear(M);
}

static int32_t factor_cmp(const void* A, const void* B) {
  return mpz_cmp((mpz_srcptr) A, (mpz_srcptr) B);
}

void print_factors(const char* N) {
  qsort(Factors, NFactors, sizeof(mpz_t), factor_cmp);

  (void) fprintf(stderr, "---------------------------------\n");
  (void) fprintf(stderr, "Prime Factors of %s:", N);
  for (uint64_t I = 0; I < NFactors; ++I) {
    (void) fputc(' ', stderr);
    (void) mpz_out_str(stderr, 10, Factors[I]);
  }
  (void) fprintf(stderr, "\n---------------------------------\n");
}

int main(int argc, char* const argv[])
{
  if (argc != 3) {
    (void) fprintf(stderr, "Usage: prime_factors <bit-width> <unsigned-integer>\n");
    return 1;
  }

  Bits = (uint32_t) strtoul(argv[1], NULL, 10);
  const char* NS = argv[2];

  // Initial size (at least a small floor so the growth logic is well-defined).
  FactorSize = Bits ? Bits : 8UL;
  Factors = (mpz_t*) malloc(FactorSize * sizeof(mpz_t));

  mpz_t N;
  mpz_init2(N, Bits);

  mpz_set_str(N, NS, 10);

  prime_factors(N);
  print_factors(NS);

  mpz_clear(N);

  for (uint64_t I = 0; I < NFactors; ++I)
    mpz_clear(Factors[I]);
  free(Factors);

  return 0;
}
