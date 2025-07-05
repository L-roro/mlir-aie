//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <cstdint>

// This kernel mirrors the one found in https://xilinx.github.io/aie_api/group__group__mmul.html
// Go through them in parallel to understand how the bfp datatype modifies accesses to memory
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void test(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB,
          bfp16ebs8 *__restrict pC) {
  // Here is where I actually want to try to do a transformation
  // aie::interleave_custom(const Vec1 &v1, const Vec2 &v2, Select select...)
}

// There is a CPU version of this function in the helper.h file
void scalarShuffleMatrixForBfp16ebs8(size_t tileWidth, size_t tileHeight, uint8_t *inBfpMatrix,
                                     uint8_t *outBfpMatrix, bool unshuffle = false) {

  tileWidth = tileWidth * 1.125;

  size_t subtileWidth = 8 * 1.125;
  size_t subtileHeight = 8;

  // The main idea is that inputGlobal X and Y are traversing the input matrix in the order we want
  // the elements to be accessed by the core, while outputGlobal X and Y are traversing the tiles in
  // the way they are going to be sent to the accelerator. Essentially, outputGlobal X and Y are
  // just traversing the tiles themselves as if they were contiguous and then going to the next
  // tile.

  size_t tileCountingIndex = 0;
  // Iterate over the subtiles in each tile
  for (size_t subtileStartY = 0; subtileStartY < tileHeight; subtileStartY += subtileHeight) {
    for (size_t subtileStartX = 0; subtileStartX < tileWidth; subtileStartX += subtileWidth) {
      // Iterate over the elements in each subtile
      for (size_t i = 0; i < subtileHeight; ++i) {
        for (size_t j = 0; j < subtileWidth; ++j) {
          size_t inputGlobalX = subtileStartX + j;
          size_t inputGlobalY = subtileStartY + i;
          size_t inputIndex = inputGlobalY * tileWidth + inputGlobalX;

          size_t outputGlobalX = tileCountingIndex % tileWidth;
          size_t outputGlobalY = tileCountingIndex / tileWidth;
          size_t outputIndex = outputGlobalY * tileWidth + outputGlobalX;

          if (!unshuffle) {
            outBfpMatrix[outputIndex] = inBfpMatrix[inputIndex];
          } else {
            outBfpMatrix[inputIndex] = inBfpMatrix[outputIndex];
          }
          tileCountingIndex++;
        }
      }
    }
  }
}

extern "C" {

void scalar_shuffle(uint8_t *pA, uint8_t *pC, size_t tileWidth, size_t tileHeight,
                    bool unshuffle = false) {
  scalarShuffleMatrixForBfp16ebs8(tileWidth, tileHeight, pA, pC, false);
}
void vectorized_shuffle(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                        bfp16ebs8 *__restrict pC) {
  test<8, 8, 8, 8, 8, 8>(pA, pB, pC);

  // Just copy the elements through the scalar unit
  uint8_t *pC8 = (uint8_t *)pC;
  uint8_t *pA8 = (uint8_t *)pA;
  uint8_t *pB8 = (uint8_t *)pB;

  // This is an attempt at using intrinsics from the vector unit to do the shuffling, but I doubt it
  // will be efficient Here are the possible instrinsics for data reshaping:
  // http://cervino-doc/aie_api/aie_api_internal/HEAD/group__group__reshape.html
  // There are other interesting intrinsics, but I could not find any way to take advantage of them
  // (for example, we can load the exponents and mantissas separately)
  // aie::block_vector_input_buffer_stream<bfp16ebs8, 64> inputA(pA);
  // inputA.pop();
  // aie::block_vector<bfp16ebs8, 64> testVec;
  // insert(testVec, )
}
}
