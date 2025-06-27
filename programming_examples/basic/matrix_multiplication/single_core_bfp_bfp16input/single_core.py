# vector_passthrough.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.dialects.aiex import v8bfp16ebs8

# from aie.helpers.taplib.tap import TensorAccessPattern
# from aie.helpers.taplib.tensortiler2d import TensorTiler2D
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.taplib.tensortiler2d import TensorTiler2D
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2


def bfp_passthrough():
    # I am only focused on C now...
    M = 64
    K = 64
    N = 64
    m = 32
    k = 32
    n = 32

    # We have to think about the fact that we are dealing with blocks of 8 elements
    # => Forced to adapt tiling dimensions to this
    # This assumes that the matrices are properly laid out already in memory (no transpose needed)
    # v8bfp_M = M
    # v8bfp_K_A = K // 8
    # v8bfp_K_B = K
    # v8bfp_N = N // 8
    # v8bfp_m = m
    # v8bfp_k_A = k // 8
    # v8bfp_k_B = k
    # v8bfp_n = n // 8
    # r = 8
    # s = 1
    # # t = t // 8

    v8bfp_M = M
    v8bfp_K_A = K
    v8bfp_K_B = K
    v8bfp_N = N
    v8bfp_m = m
    v8bfp_k_A = k
    v8bfp_k_B = k
    v8bfp_n = n
    r = 8
    s = 8
    # t = t // 8

    # TODO: Add checks here for the given dimensions

    # Modified values for c
    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Define tensor types
    # AB_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]] # TODO: Change this later to divide A and B
    # C_ty = np.ndarray[(v8bfp_M * v8bfp_N,), np.dtype[v8bfp16ebs8]]
    # ab_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]] # TODO: Change this later to divide A and B
    # c_ty = np.ndarray[(v8bfp_m * v8bfp_n,), np.dtype[v8bfp16ebs8]]

    AB_ty = np.ndarray[(M * K // 8,), np.dtype[np.int8]] # TODO: Change this later to divide A and B
    C_ty = np.ndarray[(v8bfp_M * v8bfp_N,), np.dtype[np.int8]]
    ab_ty = np.ndarray[(m * k // 8,), np.dtype[np.int8]] # TODO: Change this later to divide A and B
    c_ty = np.ndarray[(v8bfp_m * v8bfp_n,), np.dtype[np.int8]]

    matmul_kernel = Kernel(
        "matmul_testing_kernel", f"mm_{m}x{k}x{n}.o", [ab_ty, ab_ty, c_ty]
    )

    zero_kernel = Kernel("zero_kernel", f"mm_{m}x{k}x{n}.o", [c_ty])

    # AIE-array data movement with object fifos
    inA = ObjectFifo(ab_ty, name="inA")
    # a_dims = [(v8bfp_m // r, r * v8bfp_k_A), (v8bfp_k_A // s, s), (r, v8bfp_k_A), (s, 1)]
    # memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)
    memA = inA.cons().forward(name="memA")

    # Input B
    inB = ObjectFifo(ab_ty, name="inB")
    memB = inB.cons().forward(name="memB")

    # Output C
    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            zero(elem_out)

            # for _ in range_(K_div_k) if K_div_k > 1 else range(1):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matmul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)

            of_c.release(1)

    # Create worker from task
    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    # Debugging:
    # rows_per_block = 8
    # N_div_n = 1
    # print(N // 8)
    # # C_tiles = TensorTiler2D.group_tiler((M, N // 8), (m, n // 8), (rows_per_block // 2, N_div_n))
    # C_tiles = TensorTiler2D.group_tiler((32, 8), (16, 4))
    # print(f"C_tiles: {C_tiles.access_order()}")

    # A_tiles = TensorTiler2D.group_tiler(
    #     (v8bfp_M, v8bfp_K_A), (v8bfp_m, v8bfp_k_A), (2, v8bfp_K_A // v8bfp_k_A)
    # )

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k)
    )

    # print(f"A_tiles: {A_tiles.access_order()}")
    # print("A tiles access order:")
    # print(A_tiles._taps)
    # # print(A_tiles[0].access_order())
    # print(A_tiles._taps[0])
    # # print(a_dims)
    # print(f"AB_ty: {AB_ty}")
    # print(f"ab_ty: {ab_ty}")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(AB_ty, AB_ty, C_ty) as (A, B, C):
        rt.start(worker)
        # rt.fill(inA.prod(), A)
        rt.fill(inA.prod(), A, A_tiles[0])
        rt.fill(inB.prod(), B)

        rt.drain(outC.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


module = bfp_passthrough()
print(module)
