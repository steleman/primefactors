# GPU prime factorization with NVIDIA CGBN

This is the second half of my hobby Prime Factorization research project. This is the part that actually does the Prime Factorization.

There are three programs that print the **distinct prime factors** of an unsigned integer, sharing a CLI convention and an output format (the factor list is written to **stderr**). `primefactors.c` is the CPU reference. The two CUDA programs use my fork of [NVIDIA CGBN](https://github.com/steleman/nvidia-cgbn) for massively-parallel multiple-precision integer arithmetic on the GPU. Each takes a different
algorithmic route:

| Program | Strategy | Parallelism | Best at |
| --- | --- | --- | --- |
| `primefactors.c` | Trial division + Pollard's rho + ECM | CPU, OpenMP-parallel ECM | the reference; the fastest path for hard factors |
| `cgbnprimefactors.cu` | Pollard's rho + Miller-Rabin | CUDA (per-width engines) + OpenMPI | splitting composites with no known divisor table |
| `ecmcgbnprimefactorsmpi.cu` | Trial division by a prime table + ECM | CUDA + pthreads + OpenMPI | factoring against a precomputed prime table |

The two GPU programs are demonstrations of GPU big-integer parallelism - see **Performance
reality** below. Neither beats the CPU ECM path for hard semiprimes on a single node; the value there is the CGBN parallelism itself.

---

## `primefactors.c` — the CPU reference

The primary implementation: arbitrary precision, all arithmetic through GMP `mpz_t` (unbounded by
machine word size), and the yardstick the GPU programs are checked against. It factors N in two
phases:

1. **Cheap phase** — divides out 2 and then odd candidates up to `min(√M, TRIAL_BOUND)`, fully
   dividing each factor out of the working copy, so any candidate that divides is necessarily
   prime (no primality test needed).
2. **Expensive phase** — the remaining residue (only large prime factors left) is handed to a
   recursion that at each step uses Miller-Rabin (`mpz_probab_prime_p`) to test primality, then
   **Pollard's rho** to split composites, escalating to **ECM** when rho gives up. Pollard's rho
   is bounded by an iteration cap and returns failure when exhausted, so the recursion escalates
   to ECM rather than spinning (rho covers factors up to ~10¹³).

**ECM, curve-parallel across cores (OpenMP).** The Elliptic Curve Method fallback is GMP-ECM
(`libecm`), with curves parallelized over a single **shared task stream**: one `omp parallel`
region drains an atomic cursor over an `ECM_LADDER` — a per-level `{B1, curves}` schedule
following GMP-ECM's tables, each row targeting a factor size from ~15 up to ~70 digits. Low
stage-1 bounds are claimed first, but there is no barrier between levels, so a thread that
finishes a curve immediately grabs the next task instead of waiting. The first thread to split N
records the factor and trips a stop flag that in-flight curves poll, abandoning promptly. Each
thread has its own `ecm_params` with a distinctly seeded RNG (the supported thread-safe usage of
libecm).

Measured wall-clock on a 32-core box (semiprime = two equal-size primes): a 30-digit factor ~10.5 s, 35-digit ~20.6 s, 40-digit ~5.7 min — the cost per +5 digits accelerates (inherent ECM sub-exponential growth), and single-machine ECM tops out around 45 digits. Memory scales with threads × B1 (stage-2 tables are per-thread); cap with `OMP_NUM_THREADS` on tight-memory hosts.

```
%>> gmake
%>> ./primefactors <bit-width> <unsigned-integer>
# e.g.
%>> ./primefactors 128 600851475143
```

`<bit-width>` is an allocation *hint* (initial `mpz_init2` size and factor-array capacity); GMP
auto-grows, so an undersized value only affects reallocation churn, not correctness.

---

## `cgbnprimefactors.cu` — parallel Pollard's rho

Factors N by a three-stage pipeline, with all multiple-precision arithmetic on the device:

1. **Host trial division** peels off small primes (`< 100000`) using a plain `uint32`-limb host
   bignum (no GMP) and does all decimal I/O.
2. **GPU Miller-Rabin** (`mr_kernel`) decides whether a surviving cofactor is prime, running
   `cgbn_modular_power` over many bases in parallel.
3. **GPU Pollard's rho** (`rho_kernel`) launches ~8192 warps of instances, each walking `f(x) =
   x² + c mod N` with a distinct constant `c`; the first instance to find `1 < gcd < N` claims a
   global slot via `atomicCAS` and stores the factor. The host escalates the walk length across
   attempts, splits the quotient with an exact-division kernel, and recurses through a work stack
   until every factor is prime. Output is de-duplicated and sorted.

**Per-cofactor width dispatch.** The kernels are templated on CGBN width and instantiated at nine
widths (`128, 256, 512, 1024, …, 32768` bit) driven by a single `WIDTH_LIST` X-macro. Each
cofactor is dispatched at runtime to the *narrowest* width that fits it — rho's cost is dominated
by limb count, so running a ~100-bit number through 128-bit math instead of 512-bit slashes the
per-iteration op count. Inputs up to ~9860 digits are accepted (the widest engine); rho is only
*practical* for small factors (see below).

**Distributed rho (OpenMPI).** Pollard's rho is embarrassingly parallel over the walk constant
`c`, and that is the axis MPI distributes. The design is **SPMD with a replicated work stack**:
every rank runs the identical orchestration loop on identical data (trial division, Miller-Rabin,
and exact division are deterministic and cheap, so each rank recomputes them locally with zero
communication). Ranks diverge *only* inside the rho search — each rank searches a **disjoint slice
of the `c`-space** (`slot = attempt·nranks + rank`) on its own GPU. After each attempt an
`MPI_Allreduce(MIN)` elects the lowest-ranked finder and `MPI_Bcast` hands its factor to everyone,
so all ranks resume from the same split. Only rank 0 prints. With one rank (or no `mpirun`) it
collapses to the plain serial search.

```
%>> gmake cgbnprimefactors
%>> mpirun -np <ranks> ./cgbnprimefactors <bit-width> <unsigned-integer>
# e.g.
%>> ./cgbnprimefactors 128 8539734222673769370568987281911
```

The `<bit-width>` argument is accepted for CLI parity with the CPU program but is ignored — the
width is chosen automatically per cofactor.

---

## `ecmcgbnprimefactorsmpi.cu` — distributed trial division + ECM

A different strategy: factor N by **trial division against a table of proven primes read from a
file**, distributed three ways, with GMP-ECM as the fallback for whatever is left over.

- **OpenMPI** splits the divisor *value* interval `[1, √N]` into equal sub-intervals — one per
  rank. Each rank keeps only the file primes that land in its `[lo, hi)`. `√N` is computed with
  `mpz_sqrt`. Every rank reads the file and parses the input itself (SPMD, no broadcast); each
  prime lives in exactly one rank's interval, so there are no cross-rank duplicates.
- **pthreads** split each rank's prime slice into contiguous chunks — one host thread per chunk,
  each with its own CUDA stream and device buffers, launching the GPU kernel in bounded batches so
  device memory stays capped regardless of table size.
- **CUDA / CGBN** does the arithmetic: `mod_kernel` runs one CGBN instance per prime computing
  `N mod p` and flagging `p | N`. The width is chosen once from `bitlen(N)` (all remainders share
  the same N). The remainder uses **`cgbn_signed_rem`**, exercising CGBN's signed
  two's-complement support (`cgbn_signed.h`).

**The proven-primes assumption.** Primes from the file are taken as deterministically proven
(not probabilistically tested), so a divisor hit is recorded as a prime factor **with no
primality test** — Miller-Rabin is never run on a file prime. After trial division, rank 0 divides
out each found prime's full multiplicity; any residual cofactor (a prime factor `> √N`, or the
case of an incomplete table) is handed to **GMP-ECM** (`libecm`). Only that residual is
Miller-Rabin tested, to decide whether ECM should try to split it — the file primes never are.

```
%>> gmake ecmcgbnprimefactorsmpi
%>> mpirun -np <ranks> ./ecmcgbnprimefactorsmpi <primes-file> <N> [threads-per-rank]
# e.g.  (primes.txt = whitespace/newline-separated decimal primes)
%>> mpirun -np 4 ./ecmcgbnprimefactorsmpi primes.txt 100718234543 8
```

`[threads-per-rank]` defaults to the number of online CPUs. With one rank it scans the whole
`[1, √N]` interval — identical to serial.

---

## Build

```
%>> gmake cgbnprimefactors         # Pollard's-rho variant
%>> gmake ecmcgbnprimefactorsmpi   # trial-division variant
%>> gmake clean                    # clean up everything
```

Neither target is part of `make all` (which builds only the CUDA-less CPU `primefactors`), so a host without CUDA can still build the reference program.

**Toolchain / Dependencies**

- **CUDA** — built with `nvcc` (`/usr/local/cuda-12.9`), `-arch=sm_89` (RTX 4080). Change
  `GPU_ARCH` in the `Makefile` for another GPU. Also tested with CUDA 13.2 (`/usr/local/cuda-13.2`) on Blackwell (`-arch=sm_120`).
- **CGBN** — header-only, in `/usr/local/include/cgbn`. Its host backend requires GMP, which is
  why `<gmp.h>` is included before `<cgbn/cgbn.h>`. Signed support lives in `cgbn_signed.h` (auto-
  included by `cgbn.h`).
- **GMP** (`-lgmp`) — big-integer host bookkeeping and I/O.
- **GMP-ECM** (`-lecm`, `ecmcgbnprimefactorsmpi` only) — the residual-cofactor fallback.
- **OpenMPI** (`-lmpi`, both) — the Makefile derives its include/lib flags from the `mpicxx`
  wrapper via `--showme`, and embeds the OpenMPI libdir as **RUNPATH** thereby avoiding the use of `LD_LIBRARY_PATH`.
- **pthreads** (`-lpthread`, `ecmcgbnprimefactorsmpi` only).

Requirements, with the versions this was verified against:

| | for | verified with |
| --- | --- | --- |
| GMP | all three | system `libgmp` |
| GMP-ECM (`gmp-ecm-devel`) | all three | 7.0.6 |
| CUDA toolkit | `ecmcgbnprimefactorsmpi` | 12.9 |
| CUDA toolkit | `ecmcgbnprimefactorsmpi` | 13.2 |
| CGBN headers in `/usr/local/include/cgbn` | `ecmcgbnprimefactorsmpi` | - |
| OpenMPI | `ecmcgbnprimefactorsmpi` | 5.0.5 |

## Running under `mpirun`

The OpenMPI library path is recorded in the RUNPATH. `mpirun` itself must be in your `PATH`
(that is the launcher lookup, separate from the library path). If OpenMPI isn't on your default
`PATH`, load its environment module or prepend its `bin`:

```
%>> export PATH=/usr/lib64/openmpi/bin:$PATH
```

On Fedora, you could also do this:

```
%>> module load openmpi/x86_64
```

If you run it with `-np 1` (or no `mpirun` at all) either program behaves exactly like its serial version.

## Output

Both write the distinct prime factors — de-duplicated (no multiplicity) and sorted ascending — to
**stderr**, from rank 0:

```
---------------------------------
Prime Factors of 100718234543: 101 9973 99991
---------------------------------
```

## Testing

This was tested on Fedora 41 and 44 with CUDA 12.9 and 13.2 respectively:

- Laptop with an RTX 4080 Ada (sm_89) and a Core(TM) i9-14900HX.
- Intel NUC 13 Extreme with an RTX PRO 4500 Blackwell (sm_120) and a Core(TM) i9-13900K.

---

## Performance reality

These programs demonstrate massively-parallel GPU factorization; they do **not** beat the CPU for
hard inputs.

- **Pollard's rho** gains only ~√k from k parallel walks and scales O(√factor), so it is
  throughput-oriented (many small factors / many numbers), not a hard-semiprime cracker. On the
  RTX 4080 a 16-digit factor takes ~8.7 s (after width-dispatch and instance-scaling tuning cut
  it from ~37 s) versus ~2 s for CPU ECM on the same number; ≥19-digit factors are impractical.
- **Distribution** across MPI ranks is a throughput win only with **one rank per distinct GPU**
  (a multi-GPU host or a cluster). Oversubscribing a single physical GPU with several ranks is
  *slower* - the ranks time-slice on the one device. Unfortunately I do not have access to a real MPI setup where I could test the performance on a relevant number of nodes.

For real speed on single-node large factors, the CPU GMP-ECM path (`primefactors.c`) is the right tool. The value of these two programs is the CGBN big-integer parallelism itself, and the distributed orchestration around it.

The MPI workload distribution can definitely be improved. I'm thinking of a number of things to do next.

