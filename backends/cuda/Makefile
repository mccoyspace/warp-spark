# SPDX-License-Identifier: Apache-2.0
WASTE_SRC ?= ../../src
NVCC ?= nvcc
CC ?= cc
AR ?= ar
CUDA_ARCH ?= native

CPPFLAGS += -I$(WASTE_SRC) -I.
CFLAGS += -O3 -std=c11 -fPIC -Wall -Wextra -Werror
NVCCFLAGS += -O3 -std=c++17 -Xcompiler=-fPIC

LIB = libwaste_cuda_backend.a
OBJS = waste_cuda_policy.o waste_cuda_backend.o

.PHONY: all clean test test-policy
all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $(OBJS)

waste_cuda_policy.o: waste_cuda_policy.c waste_cuda_policy.h \
		waste_cuda_backend.h $(WASTE_SRC)/waste.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -c $< -o $@

waste_cuda_backend.o: waste_cuda_backend.cu waste_cuda_policy.h \
		waste_cuda_backend.h $(WASTE_SRC)/waste.h
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -fmad=false \
		-Xcompiler=-ffp-contract=off -c $< -o $@

test-policy: test_policy.c waste_cuda_policy.c waste_cuda_policy.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off \
		test_policy.c waste_cuda_policy.c -o $@

test: test-policy
	./test-policy

clean:
	rm -f $(OBJS) $(LIB) test-policy
