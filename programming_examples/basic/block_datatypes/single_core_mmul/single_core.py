#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul():
    M = 128
    K = 128
    N = 128
    m = 64
    k = 64
    n = 64

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Define tensor types
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]
    a_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    b_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m, n // 8), np.dtype[v8bfp16ebs8]]

    zero_kernel = Kernel(f"zero_kernel", f"mm_{m}x{k}x{n}.o", [c_ty])
    matmul_kernel = Kernel(
        "matmul_testing_kernel",
        f"mm_{m}x{k}x{n}.o",
        [a_ty, b_ty, c_ty],
    )

    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = [(8, 8), (8, 64), (8, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            zero(elem_out)

            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)

            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    rows_per_block = 4

    # Define tensor access patterns for inputs/outputs
    A_tiles = TensorTiler2D.group_tiler(
        (M, K // 8), (m, k // 8), (1, K_div_k), pattern_repeat=N_div_n
    )
    b_tap = TensorTiler2D.group_tiler((K, N // 8), (k, n // 8), (K_div_k, N_div_n))[0]

    C_tiles = TensorTiler2D.group_tiler((M, N // 8), (m, n // 8), (rows_per_block // 2, N_div_n))
    c_index = 0

    # print(f"Tap A 0: {A_tiles[0]}")
    # print(f"Tap A 1: {A_tiles[1]}")
    # print(f"Tap B: {b_tap}")
    # print(f"Tap C 0: {C_tiles[0]}")
    # print(f"Tap C 1: {C_tiles[1]}")

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
            # that's what this loop is for. We can track of this in the task groups for syncing.
            for pingpong in [0, 1]:

                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    # At the very last iteration, we may not need a 'pong' iteration
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    # -- A --
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])

                    # -- B --
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])

                # -- C --
                rt.drain(
                    outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True
                )
                c_index += 1

                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]

        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    module = my_program.resolve_program(SequentialPlacer())
    return module


print(my_matmul())
