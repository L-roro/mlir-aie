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
targetname=whole_array
devicename=npu2
kernelsrc:= $(firstword $(wildcard ../*.cc))
hostsrc := $(firstword $(wildcard ../*.cpp))

n_aie_cols?=4
aieargs+=--n-aie-cols ${n_aie_cols}

# M, K, N: final GEMM dimensions
M?=512
K?=512
N?=512

# m, k, n: single AIE core GEMM dimensions
m?=64
k?=64
n?=64

include ${srcdir}/../mmul_makefile_common
