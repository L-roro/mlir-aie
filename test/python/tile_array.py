# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.dialects import aie
from aie.extras import types as T
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    find_neighbors,
)
from aie.dialects.aiex import TileArray, Channel
from util import construct_and_print_module

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: broadcast
@construct_and_print_module
def broadcast(module):
    @aie.device(AIEDevice.npu1)
    def npu():
        df = TileArray()
        assert df[[0, 1], 0].shape == (2, 1)
        assert df[[0, 1], 3:].shape == (2, 3)

        fls = df[0, 0] >> df[0, 1]
        # CHECK: "aie.flow"(%0, %1) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
        print(fls)

        print()

        fls = df[[0, 1], 0] >> df[[0, 1], 3:]
        # CHECK: "aie.flow"(%0, %3) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 1 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %4) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 1 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %5) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 1 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%6, %9) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%6, %10) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%6, %11) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
        for f in fls:
            print(f)

        print()

        fls = df[0, 0] >> df[1, 0:3]
        # CHECK: "aie.flow"(%0, %6) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 2 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %7) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 2 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %8) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 2 : i32}> : (index, index) -> ()
        for f in fls:
            print(f)

        print()

        fls = df[0, 0] >> df[[2, 3], 1:]
        # CHECK: "aie.flow"(%0, %13) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %15) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %16) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %17) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %19) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %20) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %21) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %22) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %23) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        for f in fls:
            print(f)

        print()

        fls = df[0, 0].flow(df[[2, 3], 1:], source_annot="bob", dest_annot="alice")
        # CHECK: "aie.flow"(%0, %13) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %15) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %16) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %17) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %19) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %20) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %21) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %22) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %23) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        for f in fls:
            print(f)

        print()

        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 3 : i32}> : (index, index) -> ()
        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        for f in df[2, 2].flows():
            print(f)

        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        for f in df[2, 2].flows(source_annot="bob"):
            print(f)

        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        for f in df[2, 2].flows(dest_annot="alice"):
            print(f)

        # CHECK: "aie.flow"(%0, %14) <{dest_bundle = 1 : i32, dest_channel = 1 : i32, source_bundle = 1 : i32, source_channel = 4 : i32}> {dest_annot = {alice}, source_annot = {bob}} : (index, index) -> ()
        for f in df[2, 2].flows(source_annot="bob", dest_annot="alice"):
            print(f)

        assert len(df[0, 3].flows(source_annot="bob", dest_annot="alice")) == 0

        # CHECK: "aie.flow"(%0, %3) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 1 : i32}> : (index, index) -> ()
        for f in df[0, 3].flows():
            print(f)

        # CHECK: "aie.flow"(%6, %9) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32}> : (index, index) -> ()
        for f in df[1, 3].flows(filter_dest=True):
            print(f)

        # CHECK: module {
        # CHECK:   aie.device(npu1) {
        # CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
        # CHECK:     %{{.*}}tile_0_1 = aie.tile(0, 1)
        # CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
        # CHECK:     %{{.*}}tile_0_3 = aie.tile(0, 3)
        # CHECK:     %{{.*}}tile_0_4 = aie.tile(0, 4)
        # CHECK:     %{{.*}}tile_0_5 = aie.tile(0, 5)
        # CHECK:     %{{.*}}tile_1_0 = aie.tile(1, 0)
        # CHECK:     %{{.*}}tile_1_1 = aie.tile(1, 1)
        # CHECK:     %{{.*}}tile_1_2 = aie.tile(1, 2)
        # CHECK:     %{{.*}}tile_1_3 = aie.tile(1, 3)
        # CHECK:     %{{.*}}tile_1_4 = aie.tile(1, 4)
        # CHECK:     %{{.*}}tile_1_5 = aie.tile(1, 5)
        # CHECK:     %{{.*}}tile_2_0 = aie.tile(2, 0)
        # CHECK:     %{{.*}}tile_2_1 = aie.tile(2, 1)
        # CHECK:     %{{.*}}tile_2_2 = aie.tile(2, 2)
        # CHECK:     %{{.*}}tile_2_3 = aie.tile(2, 3)
        # CHECK:     %{{.*}}tile_2_4 = aie.tile(2, 4)
        # CHECK:     %{{.*}}tile_2_5 = aie.tile(2, 5)
        # CHECK:     %{{.*}}tile_3_0 = aie.tile(3, 0)
        # CHECK:     %{{.*}}tile_3_1 = aie.tile(3, 1)
        # CHECK:     %{{.*}}tile_3_2 = aie.tile(3, 2)
        # CHECK:     %{{.*}}tile_3_3 = aie.tile(3, 3)
        # CHECK:     %{{.*}}tile_3_4 = aie.tile(3, 4)
        # CHECK:     %{{.*}}tile_3_5 = aie.tile(3, 5)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_1, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 1, %{{.*}}tile_0_3, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 1, %{{.*}}tile_0_4, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 1, %{{.*}}tile_0_5, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_1_0, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_1_0, DMA : 0, %{{.*}}tile_1_4, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_1_0, DMA : 0, %{{.*}}tile_1_5, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 2, %{{.*}}tile_1_0, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 2, %{{.*}}tile_1_1, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 2, %{{.*}}tile_1_2, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_2_1, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_2_2, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_2_3, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_2_4, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_2_5, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_3_1, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_3_2, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_3_3, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_3_4, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 3, %{{.*}}tile_3_5, DMA : 0)
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_2_1, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_2_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_2_3, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_2_4, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_2_5, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_3_1, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_3_2, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_3_3, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_3_4, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 4, %{{.*}}tile_3_5, DMA : 1) {dest_annot = {alice}, source_annot = {bob}}
        # CHECK:   }
        # CHECK: }

    print(module)


# CHECK-LABEL: lshift
@construct_and_print_module
def lshift(module):
    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        fls = tiles[2, 1] << tiles[0, [2, 3]]
        # CHECK: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_2_1, DMA : 0)
        # CHECK: aie.flow(%{{.*}}tile_0_3, DMA : 0, %{{.*}}tile_2_1, DMA : 1)

        fls = tiles[2, 1] << tiles[0, [2, 3]]
        # CHECK: aie.flow(%{{.*}}tile_0_2, DMA : 1, %{{.*}}tile_2_1, DMA : 2)
        # CHECK: aie.flow(%{{.*}}tile_0_3, DMA : 1, %{{.*}}tile_2_1, DMA : 3)

    print(npu)


# CHECK-LABEL: locks
@construct_and_print_module
def locks(module):
    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        aie.lock(tiles[0, 1].tile)
        # CHECK: %lock_0_1 = aie.lock(%{{.*}}tile_0_1)
        for l in tiles[0, 1].locks():
            print(l.owner)

        aie.lock(tiles[0, 2].tile)
        aie.lock(tiles[0, 2].tile, annot="bob")
        aie.lock(tiles[0, 3].tile)
        aie.lock(tiles[0, 3].tile, annot="alice")

        # CHECK: %lock_0_2 = aie.lock(%{{.*}}tile_0_2)
        # CHECK: %lock_0_2_0 = aie.lock(%{{.*}}tile_0_2) {annot = {bob}}
        # for l in tiles[0, 2].locks():
        #     print(l.owner)

        # NOCHECK: %lock_0_2_0 = aie.lock(%{{.*}}tile_0_2) {annot = {bob}}
        assert len(tiles[0, 2].locks(annot="bob"))
        # for l in tiles[0, 2].locks(annot="bob"):
        #     print(l.owner)

        assert len(tiles[0, 2].locks(annot="alice")) == 0

        assert len(tiles[0, 3].locks(annot="alice")) == 1
        # CHECK: %lock_0_3_1 = aie.lock(%{{.*}}tile_0_3) {annot = {alice}}
        # for l in tiles[0, 3].locks(annot="alice"):
        #     print(l.owner)

    print(module)


# CHECK-LABEL: neighbors
@construct_and_print_module
def neighbors(module):
    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        # CHECK: Neighbors(north=%[[SSA_15:[0-9]+]] = "aie.tile"() <{col = 2 : i32, row = 3 : i32}> : () -> index, west=%[[SSA_8:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index, south=None)
        print(find_neighbors(tiles[2, 2].tile))

        assert tiles[1:3, 1:3].neighbors().shape == (2, 2)
        # CHECK: tile(col=1, row=1) : Neighbors(north=None, west=None, south=None)
        # CHECK: tile(col=1, row=2) : Neighbors(north=<TileArray: [%[[SSA_9:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 3 : i32}> : () -> index]>, west=<TileArray: [%[[SSA_2:[0-9]+]] = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index]>, south=None)
        # CHECK: tile(col=2, row=1) : Neighbors(north=None, west=<TileArray: [%[[SSA_7:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 1 : i32}> : () -> index]>, south=None)
        for idx, n in np.ndenumerate(tiles[1:3, 1:3].neighbors()):
            print(tiles[1:3, 1:3][idx].tile, ":", n)

        # CHECK: tile(col=1, row=1) : Neighbors(north=<TileArray: [%[[SSA_8:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index]>, west=None, south=<TileArray: [%[[SSA_6:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 0 : i32}> : () -> index]>)
        # CHECK: tile(col=1, row=2) : Neighbors(north=<TileArray: [%[[SSA_9:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 3 : i32}> : () -> index]>, west=<TileArray: [%[[SSA_2:[0-9]+]] = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index]>, south=<TileArray: [%[[SSA_7:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 1 : i32}> : () -> index]>)
        # CHECK: tile(col=2, row=1) : Neighbors(north=<TileArray: [%[[SSA_14:[0-9]+]] = "aie.tile"() <{col = 2 : i32, row = 2 : i32}> : () -> index]>, west=<TileArray: [%[[SSA_7:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 1 : i32}> : () -> index]>, south=<TileArray: [%[[SSA_12:[0-9]+]] = "aie.tile"() <{col = 2 : i32, row = 0 : i32}> : () -> index]>)
        # CHECK: tile(col=2, row=2) : Neighbors(north=<TileArray: [%[[SSA_15:[0-9]+]] = "aie.tile"() <{col = 2 : i32, row = 3 : i32}> : () -> index]>, west=<TileArray: [%[[SSA_8:[0-9]+]] = "aie.tile"() <{col = 1 : i32, row = 2 : i32}> : () -> index]>, south=<TileArray: [%[[SSA_13:[0-9]+]] = "aie.tile"() <{col = 2 : i32, row = 1 : i32}> : () -> index]>)
        for idx, n in np.ndenumerate(tiles[1:3, 1:3].neighbors(logical=False)):
            print(tiles[1:3, 1:3][idx].tile, ":", n)


# CHECK-LABEL: channels_basic
@construct_and_print_module
def channels_basic(module):

    # CHECK-LABEL: test-basic
    print("test-basic")

    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        b = aie.buffer(
            tiles[2, 2].tile, np.ndarray[(10, 10), np.dtype[np.int32]], name="bob"
        )
        c = Channel(tiles[2, 2].tile, b)
        c = Channel(
            tiles[2, 2].tile, shape=(10, 10), dtype=np.int32, buffer_name="alice"
        )

    # CHECK: %bob = aie.buffer(%{{.*}}tile_2_2) {sym_name = "bob"} : memref<10x10xi32>
    # CHECK: %bob_producer_lock = aie.lock(%{{.*}}tile_2_2) {sym_name = "bob_producer_lock"}
    # CHECK: %bob_consumer_lock = aie.lock(%{{.*}}tile_2_2) {sym_name = "bob_consumer_lock"}
    # CHECK: %alice = aie.buffer(%{{.*}}tile_2_2) {sym_name = "alice"} : memref<10x10xi32>
    # CHECK: %alice_producer_lock = aie.lock(%{{.*}}tile_2_2) {sym_name = "alice_producer_lock"}
    # CHECK: %alice_consumer_lock = aie.lock(%{{.*}}tile_2_2) {sym_name = "alice_consumer_lock"}
    print(npu)

    # CHECK-LABEL: test-context-manager
    print("test-context-manager")

    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        c = Channel(
            tiles[2, 2].tile, shape=(10, 10), dtype=np.int32, buffer_name="alice"
        )

        @aie.mem(tiles[2, 2].tile)
        def mem():
            with c.put() as buffer:
                # CHECK: %[[SSA_30:[0-9]+]] = "aie.buffer"(%[[SSA_14]]) <{sym_name = "alice"}> : (index) -> memref<10x10xi32>
                print(buffer.owner)
            aie.end()

        @aie.core(tiles[2, 2].tile)
        def core():
            with c.get() as buffer:
                # CHECK: %[[SSA_30:[0-9]+]] = "aie.buffer"(%[[SSA_14]]) <{sym_name = "alice"}> : (index) -> memref<10x10xi32>
                print(buffer.owner)

    # CHECK: %alice = aie.buffer(%{{.*}}tile_2_2) {sym_name = "alice"} : memref<10x10xi32>
    # CHECK: %alice_producer_lock = aie.lock(%{{.*}}tile_2_2) {sym_name = "alice_producer_lock"}
    # CHECK: %alice_consumer_lock = aie.lock(%{{.*}}tile_2_2) {sym_name = "alice_consumer_lock"}
    # CHECK: %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
    # CHECK:   aie.use_lock(%alice_producer_lock, AcquireGreaterEqual)
    # CHECK:   aie.use_lock(%alice_consumer_lock, Release)
    # CHECK:   aie.end
    # CHECK: }
    # CHECK: %core_2_2 = aie.core(%{{.*}}tile_2_2) {
    # CHECK:   aie.use_lock(%alice_consumer_lock, AcquireGreaterEqual)
    # CHECK:   aie.use_lock(%alice_producer_lock, Release)
    # CHECK:   aie.end
    # CHECK: }
    print(npu)


# CHECK-LABEL: nd_channels
@construct_and_print_module
def nd_channels(module):
    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        shapes = np.array([(10, 10)], dtype="i,i").astype(object)
        c = tiles[2, 2].channel(shape=shapes, dtype=[np.int32])
        # CHECK: <Channel: buffer=MemRef(%[[SSA_30:[0-9]+]], memref<10x10xi32>) producer_lock=Scalar(%[[SSA_31:[0-9]+]] = "aie.lock"(%[[SSA_14]]) <{sym_name = "[[SSA_30]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_32:[0-9]+]] = "aie.lock"(%[[SSA_14]]) <{sym_name = "[[SSA_30]]_consumer_lock"}> : (index) -> index)>
        print(c)
        cs = tiles[2:4, 2:4].channel(shape=shapes, dtype=[np.int32])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) <Channel: buffer=MemRef(%[[SSA_33:[0-9]+]], memref<10x10xi32>) producer_lock=Scalar(%[[SSA_34:[0-9]+]] = "aie.lock"(%[[SSA_14:[0-9]+]]) <{sym_name = "[[SSA_33]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_35:[0-9]+]] = "aie.lock"(%[[SSA_14]]) <{sym_name = "[[SSA_33]]_consumer_lock"}> : (index) -> index)>
        # CHECK: (0, 1) <Channel: buffer=MemRef(%[[SSA_36:[0-9]+]], memref<10x10xi32>) producer_lock=Scalar(%[[SSA_37:[0-9]+]] = "aie.lock"(%[[SSA_15:[0-9]+]]) <{sym_name = "[[SSA_36]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_38:[0-9]+]] = "aie.lock"(%[[SSA_15]]) <{sym_name = "[[SSA_36]]_consumer_lock"}> : (index) -> index)>
        # CHECK: (1, 0) <Channel: buffer=MemRef(%[[SSA_39:[0-9]+]], memref<10x10xi32>) producer_lock=Scalar(%[[SSA_40:[0-9]+]] = "aie.lock"(%[[SSA_20:[0-9]+]]) <{sym_name = "[[SSA_39]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_41:[0-9]+]] = "aie.lock"(%[[SSA_20]]) <{sym_name = "[[SSA_39]]_consumer_lock"}> : (index) -> index)>
        # CHECK: (1, 1) <Channel: buffer=MemRef(%[[SSA_42:[0-9]+]], memref<10x10xi32>) producer_lock=Scalar(%[[SSA_43:[0-9]+]] = "aie.lock"(%[[SSA_21:[0-9]+]]) <{sym_name = "[[SSA_42]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_44:[0-9]+]] = "aie.lock"(%[[SSA_21]]) <{sym_name = "[[SSA_42]]_consumer_lock"}> : (index) -> index)>
        for idx, c in np.ndenumerate(cs):
            print(idx, c)

        shapes = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype="i,i").astype(
            object
        )
        cs = tiles[2:4, 2:4].channel(shape=shapes, dtype=[np.int32])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) <Channel: buffer=MemRef(%[[SSA_45:[0-9]+]], memref<1x2xi32>) producer_lock=Scalar(%[[SSA_46:[0-9]+]] = "aie.lock"(%[[SSA_14:[0-9]+]]) <{sym_name = "[[SSA_45]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_47:[0-9]+]] = "aie.lock"(%[[SSA_14]]) <{sym_name = "[[SSA_45]]_consumer_lock"}> : (index) -> index)>
        # CHECK: (0, 1) <Channel: buffer=MemRef(%[[SSA_48:[0-9]+]], memref<3x4xi32>) producer_lock=Scalar(%[[SSA_49:[0-9]+]] = "aie.lock"(%[[SSA_15:[0-9]+]]) <{sym_name = "[[SSA_48]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_50:[0-9]+]] = "aie.lock"(%[[SSA_15]]) <{sym_name = "[[SSA_48]]_consumer_lock"}> : (index) -> index)>
        # CHECK: (1, 0) <Channel: buffer=MemRef(%[[SSA_51:[0-9]+]], memref<5x6xi32>) producer_lock=Scalar(%[[SSA_52:[0-9]+]] = "aie.lock"(%[[SSA_20:[0-9]+]]) <{sym_name = "[[SSA_51]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_53:[0-9]+]] = "aie.lock"(%[[SSA_20]]) <{sym_name = "[[SSA_51]]_consumer_lock"}> : (index) -> index)>
        # CHECK: (1, 1) <Channel: buffer=MemRef(%[[SSA_54:[0-9]+]], memref<7x8xi32>) producer_lock=Scalar(%[[SSA_55:[0-9]+]] = "aie.lock"(%[[SSA_21:[0-9]+]]) <{sym_name = "[[SSA_54]]_producer_lock"}> : (index) -> index) consumer_lock=Scalar(%[[SSA_56:[0-9]+]] = "aie.lock"(%[[SSA_21]]) <{sym_name = "[[SSA_54]]_consumer_lock"}> : (index) -> index)>
        for idx, c in np.ndenumerate(cs):
            print(idx, c)


# CHECK-LABEL: buffer_test_this_needs_to_distinct_from_all_other_mentions_of_buffer_in_this_file
@construct_and_print_module
def buffer_test_this_needs_to_distinct_from_all_other_mentions_of_buffer_in_this_file(
    module,
):
    @aie.device(AIEDevice.npu1)
    def npu():
        tiles = TileArray()

        shapes = [(10, 10)]
        c = tiles[2, 2].buffer(shape=shapes, dtype=[np.int32])
        # CHECK: MemRef(%[[SSA_30]], memref<10x10xi32>)
        print(c)
        cs = tiles[2:4, 2:4].buffer(shape=shapes, dtype=[np.int32])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) MemRef(%[[SSA_31]], memref<10x10xi32>)
        # CHECK: (0, 1) MemRef(%[[SSA_32]], memref<10x10xi32>)
        # CHECK: (1, 0) MemRef(%[[SSA_33]], memref<10x10xi32>)
        # CHECK: (1, 1) MemRef(%[[SSA_34]], memref<10x10xi32>)
        for idx, c in np.ndenumerate(cs):
            print(idx, c)

        shapes = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        cs = tiles[2:4, 2:4].buffer(shape=shapes, dtype=[np.int32])
        assert cs.shape == (2, 2)

        # CHECK: (0, 0) MemRef(%[[SSA_35]], memref<1x2xi32>)
        # CHECK: (0, 1) MemRef(%[[SSA_36]], memref<3x4xi32>)
        # CHECK: (1, 0) MemRef(%[[SSA_37]], memref<5x6xi32>)
        # CHECK: (1, 1) MemRef(%[[SSA_38]], memref<7x8xi32>)
        for idx, c in np.ndenumerate(cs):
            print(idx, c)
