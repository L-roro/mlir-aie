##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

include ${current_dir}../../../makefile-common
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
matrix_multiplication_dir := ${srcdir}/../../matrix_multiplication
subdir=${srcdir}
targetname=single_core
devicename=npu2

# M, K, N: final GEMM dimensions
M?=64
K?=64
N?=64

# m, k, n: single AIE core GEMM dimensions
m?=64
k?=64
n?=64

b_col_maj?=1

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}

target_suffix?=${M}x${K}x${N}
mlir_target?=build/aie_${target_suffix}.mlir
xclbin_target?=build/final_${target_suffix}.xclbin
insts_target?=build/insts_${target_suffix}.txt
aie_py_src?=${targetname}.py

runargs?=-v 2 --warmup 1 --iters 1
aieargs+=-M $M -K $K -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out}

use_chess?=0

ifneq (${use_chess}, 1)
aiecc_peano_flags=--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}
endif

# KERNEL_CC defines the C compiler used to compile the kernel.
# KERNEL_CFLAGS holds the flags passed to that compiler.
# By default, we use Peano, a.k.a. llvm-aie.
KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++
KERNEL_CFLAGS=${PEANOWRAP2P_FLAGS}

ifeq (${use_chess}, 1)
KERNEL_CC=xchesscc_wrapper
KERNEL_CFLAGS=${CHESSCCWRAP2P_FLAGS}
endif

# ##########################################################################
# Targets
# ##########################################################################

.PHONY: all
all: ${xclbin_target} ${targetname}.exe

build/mm_${m}x${k}x${n}.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${xclbin_target}: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				${aiecc_peano_flags} \
				--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp ${matrix_multiplication_dir}/common.h ${srcdir}/../helper.h
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb -DDTYPE_IN=${dtype_in_cpp} -DDTYPE_OUT=${dtype_out_cpp} -DDTYPE_ACC=${dtype_acc_cpp}" \
		cmake `${getwslpath} ${srcdir}` -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname} -Dsubdir=${subdir}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs}

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
