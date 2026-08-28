CC = /usr/bin/gcc
CFLAGS = -O3 -std=c17 -Wall -Wextra -Wpedantic
CFLAGS += -funroll-loops -ftree-vectorize -ftree-slp-vectorize
CFLAGS += -fopenmp
CPPFLAGS = -D_GNU_SOURCE -D_XOPEN_SOURCE=700
LDFLAGS = -lecm -lgmp -lm -flto=32 -O3

# CUDA / CGBN toolchain for the GPU build.
CUDA_HOME = /usr/local/cuda-12.9
NVCC = $(CUDA_HOME)/bin/nvcc
GPU_ARCH = sm_89

# OpenMPI for the distributed GPU build. Flags are derived from the mpicxx wrapper
# (--showme) so the paths follow the installed package rather than being hardcoded;
# they resolve to -I<mpi include> and -L<mpi libdir> -lmpi. Override MPICXX for a
# different MPI. nvcc takes -I/-L/-l natively, so no -ccbin swap is needed.
MPICXX = /usr/lib64/openmpi/bin/mpicxx
MPI_LIBDIRS  = $(shell $(MPICXX) --showme:libdirs)
MPI_INCFLAGS = $(addprefix -I,$(shell $(MPICXX) --showme:incdirs))
MPI_LIBFLAGS = $(addprefix -L,$(MPI_LIBDIRS)) \
               $(addprefix -l,$(shell $(MPICXX) --showme:libs))

# Embed the OpenMPI libdir(s) (e.g. /usr/lib64/openmpi/lib) as RUNPATH so the MPI
# programs find libmpi at runtime without LD_LIBRARY_PATH. --enable-new-dtags
# emits DT_RUNPATH (the modern tag) rather than the legacy DT_RPATH; -Xlinker
# passes each token straight to the host linker.
MPI_RPATHFLAGS = -Xlinker --enable-new-dtags \
                 $(foreach d,$(MPI_LIBDIRS),-Xlinker -rpath -Xlinker $(d))

NVCCFLAGS = -O3 -arch=$(GPU_ARCH) -I/usr/local/include $(MPI_INCFLAGS)
NVCCLIBS = -lgmp $(MPI_LIBFLAGS) $(MPI_RPATHFLAGS)

PROGRAMS = primefactors

all: $(PROGRAMS)

primefactors: primefactors.o
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@

.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# GPU build (requires CUDA + CGBN). Not part of `all` so a CUDA-less host can
# still build the CPU program.
cgbnprimefactors: cgbnprimefactors.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(NVCCLIBS)

# Distributed GPU trial-division variant: CUDA/CGBN + pthreads + OpenMPI + libecm.
# Also NOT part of `all`. Adds -lecm (before -lgmp; libecm depends on libgmp) and
# -lpthread on top of the shared CUDA/MPI flags.
ecmcgbnprimefactorsmpi: ecmcgbnprimefactorsmpi.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lecm $(NVCCLIBS) -lpthread

clean:
	rm -f primefactors primefactors.o cgbnprimefactors ecmcgbnprimefactorsmpi

