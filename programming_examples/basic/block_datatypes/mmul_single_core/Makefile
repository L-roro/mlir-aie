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
M?=128
K?=128
N?=128

# m, k, n: single AIE core GEMM dimensions
m?=64
k?=64
n?=64

include ${srcdir}/../mmul_makefile_common
