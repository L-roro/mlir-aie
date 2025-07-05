#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.localbuffer import LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.dialects.aiex import v8bfp16ebs8


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    args = argparser.parse_args()
    print(my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.n_aie_cols))


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, n_aie_cols):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    r, s, t = 8, 8, 8

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    assert (
        m == 64 and k == 64 and n == 64
    ), """Only 64x64x64 tiles are supported in this example"""

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    # When using more AIE columns than n_aie_rows (4) (applicable to NPU2),
    # restrict the number of shim/mem tiles to n_aie_rows,
    # since we have only n_aie_rows row tiles for matrix A
    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    # When using n_aie_rows (4) or less AIE columns (both NPU and NPU2),
    # the number of shim/mem tiles are equal to n_aie_cols.
    # We use the distribute pattern in object FIFO (see linking for A below),
    # since we have n_aie_rows (4) row tiles for matrix A
    else:
        n_shim_mem_A = n_aie_cols

    # Integer division when n_aie_cols < 4, otherwise set to 1
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    dev_ty = NPU2()

    # Define tensor types
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]
    A_l2_ty = np.ndarray[(m * k // 8 * n_A_tiles_per_shim,), np.dtype[v8bfp16ebs8]]
    B_l2_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(m * n // 8 * n_aie_rows,), np.dtype[v8bfp16ebs8]]
    A_l1_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n // 8), np.dtype[v8bfp16ebs8]]

    # AIE Core Function declarations
    zero_kernel = Kernel(f"zero_kernel", f"mm_{m}x{k}x{n}.o", [C_l1_ty])
    matmul_kernel = Kernel(
        "matmul_vectorized_bfp16",
        f"mm_{m}x{k}x{n}.o",
        [A_l1_ty, B_l1_ty, C_l1_ty, np.int32, np.int32, np.int32],
    )
    shuffle_kernel = Kernel(
        "scalar_shuffle",
        f"mm_{m}x{k}x{n}.o",
        # Assumes all matrices are identical!
        [A_l1_ty, C_l1_ty, np.int16, np.int16, np.int16],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        # If n_shim_mem_A == n_rows, n_A_tiles_per_shim is 1 and
        # this simply links a_l3l2_fifos[i] to a_l2l1_fifos[i] directly,
        # If n_shim_mem_A < n_rows, each column receives multiple rows of
        # tiles; distribute it along rows of AIE cores.
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k // 8 * j for j in range(stop_row - start_row)]
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                placement=Tile(
                    2 * i if n_aie_cols == 8 else i, 1
                ),  # alternate columns in full 4x8 NPU2 case
            )
        )

        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        dims_to_stream = [(8, 8), (8, 64), (8, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        # Output C
        C_l2l3_fifos[col] = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth)
        of_offsets = [m * n // 8 * i for i in range(n_aie_rows)]

        # join along one column
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # Tasks for each worker to perform
    def core_fn(in_a, in_b, out_c, zero, matmul, shuffle):
        bufferA = LocalBuffer(A_l1_ty)
        bufferB = LocalBuffer(B_l1_ty)
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                shuffle(elem_in_a, bufferA, k // 8, m, False)
                shuffle(elem_in_b, bufferB, n // 8, k, False)
                matmul(bufferA, bufferB, elem_out, m, k, n)
                # shuffle(elem_out, bufferA, n // 8, m, True)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    # Set up compute tiles
    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row][col]
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                        shuffle_kernel,
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    # Define tensor access patterns (tiling) for A, B, and C
    A_tiles = TensorTiler2D.group_tiler(
        (M, K // 8),  # Size of A matrix
        (m * n_A_tiles_per_shim, k // 8),  # Size of A (smallest) tile
        (1, K // k),  # Size of "group" of tiles
        # Repeat data so can distribute across whole column
        pattern_repeat=N // n // n_aie_cols,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N // 8),  # Size of B matrix
        (k, n // 8),  # Size of B tile
        # Number of tiles per transfer in each dimension (whole col, partial row)
        tile_group_repeats=(K // k // n_aie_cols, N // n),
        # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
        tile_group_steps=(n_aie_cols, 1),
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N // 8),  # Size of C matrix
        (m * n_aie_rows, n // 8),  # Size of C tile
        # Number of tiles per transfer in each dimension (partial col, partial row)
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        # Collect every n_aie_cols row at a time (mirroring how we sent in B data)
        tile_group_steps=(1, n_aie_cols),
    )
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    # May not have pong iteration in some cases
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    # C Output Transfer:
                    # The smallest transfer unit is a (m*n_aie_rows)-x-(n)-sized sub-tile of the matrix.
                    # Transfer one such tile for every (n_aie_cols)-th column, evenly spaced,
                    # then repeat that (tb_n_rows) times for the next contiguous blocks of rows.
                    # Each shim will start at a different column offset, transferring interleaved
                    # columns. For example, shim 0 may transfer the blocks marked 0 below, and shim 1
                    # may transfer the blocks marked 1.
                    #
                    #             N
                    #      ----------------
                    #     |0011    0011    |
                    #     |0011    0011    |
                    #     |0011    0011    |
                    # M   |0011    0011    |
                    #     |                |
                    #     |                |
                    #     |                |
                    #     |                |
                    #      ----------------
                    rt.drain(
                        C_l2l3_fifos[col].cons(),
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                        placement=Tile(col, 0),
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):

                        # A input transfer:
                        #
                        # The smallest transfer unit is a (m*n_A_tiles_per_shim)-sized sub-tile of the input matrix.
                        # Transfer one such tile for every column, contiguously.
                        # Repeat this transfer with identical tiles a total of (N//n//n_aie_cols) times.
                        # Each shim transfers the tiles for separate rows. For example, shim 0 may transfer the
                        # tiles marked 0 below, and shim 1 may transfer the tiles marked 1.
                        #             K
                        #      ----------------
                        #     |0000000000000000|    (repeated N//n//n_aie_cols times)
                        #     |0000000000000000|
                        #     |1111111111111111|
                        # M   |1111111111111111|
                        #     |                |
                        #     |                |
                        #     |                |
                        #     |                |
                        #      ----------------
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)

                        # always equal to n_aie_rows since we have n_aie_rows row tiles for matrix A
                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                A,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                                placement=Tile(
                                    2 * col if n_aie_cols == 8 else col, 0
                                ),  # alternate columns in full 4x8 NPU2 case
                            )
                        # Use the calculated sizes/strides/offsets to record the data movement
                        # caused by the above call to npu_dma_memcpy_nd.
                        # This line does not change MLIR output at all.

                        # B input transfer:
                        # Transfer the first a (n)-wide block of columns of B,
                        # Then transfer the (n_aie_columns)-th such block, and so on.
                        # Each shim will start at a different column offset.
                        # For example, shim 0 may transfer the tiles marked 0 below,
                        # and shim 1 may transfer the tiles marked 1.
                        #
                        #             N
                        #      ----------------
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        # K   |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #      ----------------
                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                            placement=Tile(col, 0),
                        )

                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    # Create the program from the device type and runtime
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
