//===- helper.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

// Helper function to generate random floating point numbers with high exponent
// variance (useful for blocked datatypes). Exponents are interpreted as base 2
inline float generateRandomFloatingPoint(std::mt19937 &eng, double minExp, double maxExp) {
  std::uniform_real_distribution<float> distrExp(minExp, maxExp);
  float exponent = distrExp(eng);

  std::uniform_real_distribution<float> distrMantissa(0.0, 1.0);
  float mantissa = distrMantissa(eng);

  return mantissa * std::pow(2.0, exponent);
}

// block - block size
// size  - length of the input array
// array - the array
// returnArray - the array to be filled with the quantized values
// rounding - 0 for zero, 1 for nearest (tie to even)
// verbose - make some noise
// Quantization of an array of floats to bfp16.
// The return array is structured as follows:
// 1. The first byte is the shared exponent (max exponent of the block).
// 2. The next *block* bytes are the quantized values.
inline std::vector<uint8_t> floatToBfp16(int block, int size, float *array, int rounding = 0,
                                  int verbose = 0) {
  std::vector<uint8_t> res(size * 1.125);

  int mbits = 7;
  int start = 0, end, i, j, int8 = 1;
  unsigned int sign, exp, maxExp;
  unsigned int *p, mantissa, mask, value;
  int shift, maxShift;
  int8_t valueInt8;

  while (true) {
    // decide on the block (starting and ending point)
    end = start + block;
    if (end > size)
      end = size;

    // Find max exp
    maxExp = 0;
    for (i = start; i < end; i++) {
      p = (unsigned int *)(array + i);
      exp = *p >> 23;    // Get rid of mantissa
      exp &= 0x000000FF; // Keep the last 8 bit exponent (remove sign)

      if (maxExp < exp)
        maxExp = exp;
    }

    // Round each number
    maxShift = 0;
    for (i = start; i < end; i++) {
      p = (unsigned int *)(array + i);
      if (verbose) {
        printf("%d: value in float = %f\n", i, array[i]);
      }

      // sign, exp, and mantissa
      sign = *p & 0x80000000;     // Sign
      exp = *p >> 23;             // Get rid of mantissa
      exp &= 0x000000FF;          // Keep the last 8 bit exponent (remove sign)
      mantissa = *p & 0x007FFFFF; // 23-bit mantissa
      if (exp)
        mantissa |= 0x00800000; // add the implicit for normal value

      if (exp >= 255)
        continue; // Infinity or NaN remains

      // Calculate shift (bits needs to be zeroed out)
      // At least erase 23 - mbits + 1 (+1 is for making the implicit bit
      // explicit) or more if smaller
      shift = 23 - mbits + 1 + maxExp - exp;
      if (verbose) {
        printf("%d: shift=%d rounding=%d\n", i, shift, rounding);
        printf("%d: AS READ         sign=%d exp=%d mantissa=0x%08x\n", i, sign, exp, mantissa);
      }

      // Calculate rounding
      switch (rounding) {
      case 0:
        break; // do nothing, just truncate
      case 1:
        mantissa += 1 << (shift - 1); // add rounding for nearest
        mask = 1;
        for (j = 0; j <= shift; j++) {
          if (mantissa & mask) {
            if (j < shift)
              break; // some bit is set, not a tie case
            if (j == shift)
              mantissa &= ~mask; // tie case, rounded to odd bits, adjust to even
          }
          mask <<= 1;
        }
        break;
      default:
        break;
      }
      if (verbose) {
        printf("%d: ADDED ROUNDING  sign=%d exp=%d mantissa=0x%08x\n", i, sign, exp, mantissa);
      }
      if (mantissa & 0x01000000) { // rounding carried forward and need to adjust exponent
        if (exp < maxExp) {        // This will not result in shifting of max_exp
          mantissa >>= 1;
          exp += 1;
          shift -= 1;
          if (exp >= 255)
            mantissa = 0;         // overflow, signals infinity, should not happen
        } else {                  // Keep the current scale and round down
          mantissa -= 1 << shift; // Round down instead
        }
      }
      if (verbose) {
        printf("%d: ADJUST CARRY    sign=%d exp=%d mantissa=0x%08x\n", i, sign, exp, mantissa);
      }

      // Perform shift
      if (shift < 32) {
        mantissa >>= shift; // setting bits to zero
        mantissa <<= shift;
      } else {
        mantissa = 0;
      }

      if (verbose) {
        printf("%d: SHIFTED         sign=%d exp=%d mantissa=0x%08x\n", i, sign, exp, mantissa);
      }
      if (mantissa) {
        if (shift < 32)
          valueInt8 = (sign >> 24) | (mantissa >> (17 + maxExp - exp));
        else
          valueInt8 = (sign >> 24);
        if (exp)
          mantissa &= ~0x00800000; // remove implicit bit for normal number
        value = sign | (exp << 23) | mantissa;
      } else {
        valueInt8 = 0;
        value = sign; // Mantissa is rounded to zero, signal zero
      }
      *p = value;
      if (verbose) {
        printf("%d: TO BE WRITTEN   sign=%d exp=%d mantissa=0x%08x\n", i, sign, exp, mantissa);
        printf("%d: value = %f\n", i, *(array + i));
        printf("%d: value_int8 = 0x%08x\n", int8, valueInt8);
        printf("max_exp = %d\n", maxExp);
      }
      res[int8] = valueInt8;
      int8++;

      if (maxShift < shift)
        maxShift = shift;
    }
    res[int8 - 9] = (uint8_t)maxExp;
    int8++;
    start = end;
    if (start >= size)
      break;
  }

  return res;
}

// Convert a bfp16 array to a float.
// Size should be the number of bytes in the input bfp16 array
inline std::vector<float> bfp16ebs8ToFloat(int size, uint8_t *array, int verbose = 0) {
  std::vector<float> res(size / 1.125);

  int block = 8;
  int tempIndx = 0;
  for (int i = 0; i < size; i += block + 1) {
    uint8_t sharedExponent = (uint8_t)array[i];
    float multiplier;
    if (sharedExponent >= 127) {
      multiplier = 1.0 * (1 << (sharedExponent - 127));
    } else {
      multiplier = 1.0 / (1 << (127 - sharedExponent));
    }
    multiplier /= 64.0;
    if (verbose) {
      printf("shared_exponent = %d\n", sharedExponent);
      printf("multiplier = %f\n", multiplier);
    }
    for (int j = 1; j < block + 1; j++) {
      res[tempIndx] = float(array[i + j] * multiplier);
      if (verbose) {
        printf("return_array[%d] = %f\n", tempIndx, res[tempIndx]);
      }
      tempIndx++;
    }
  }

  return res;
}

// Shuffle tiles of 64x64 elements for the matrix
// Width and height are expected to be the number of scalar elements in the matrix
// This function rearranges the 8x8 subtiles into rows so that a single subtile is contiguous in
// memory within each tile.
inline std::vector<uint8_t> shuffleMatrixForBfp16ebs8(size_t width, size_t height,
                                                      std::vector<uint8_t> bfpMatrix,
                                                      bool unshuffle = false) {
  assert(width % 64 == 0 && "Matrix width must be divisible by tile dimension");
  assert(width % 64 == 0 && "Matrix height must be divisible by tile dimension");
  assert(bfpMatrix.size() == (size_t)width * height * 1.125 &&
         "Matrix size must be width*height*1.125");

  width = width * 1.125;
  std::vector<uint8_t> res(width * height);

  size_t tileWidth = 64 * 1.125;
  size_t tileHeight = 64;

  size_t subtileWidth = 8 * 1.125;
  size_t subtileHeight = 8;

  // The main idea is that inputGlobal X and Y are traversing the input matrix in the order we want
  // the elements to be accessed by the core, while outputGlobal X and Y are traversing the tiles in
  // the way they are going to be sent to the accelerator. Essentially, outputGlobal X and Y are
  // just traversing the tiles themselves as if they were contiguous and then going to the next
  // tile.

  // Iterate over the tiles in the matrix
  for (size_t tileStartY = 0; tileStartY < height; tileStartY += tileHeight) {
    for (size_t tileStartX = 0; tileStartX < width; tileStartX += tileWidth) {

      size_t tileCountingIndex = 0;
      // Iterate over the subtiles in each tile
      for (size_t subtileStartY = 0; subtileStartY < tileHeight; subtileStartY += subtileHeight) {
        for (size_t subtileStartX = 0; subtileStartX < tileWidth; subtileStartX += subtileWidth) {

          // Iterate over the elements in each subtile
          for (size_t i = 0; i < subtileHeight; ++i) {
            for (size_t j = 0; j < subtileWidth; ++j) {
              size_t inputGlobalX = tileStartX + subtileStartX + j;
              size_t inputGlobalY = tileStartY + subtileStartY + i;
              size_t inputIndex = inputGlobalY * width + inputGlobalX;

              size_t outputGlobalX = tileStartX + tileCountingIndex % tileWidth;
              size_t outputGlobalY = tileStartY + tileCountingIndex / tileWidth;
              size_t outputIndex = outputGlobalY * width + outputGlobalX;

              if (!unshuffle) {
                res[outputIndex] = bfpMatrix[inputIndex];
              } else {
                res[inputIndex] = bfpMatrix[outputIndex];
              }
              tileCountingIndex++;
            }
          }
        }
      }
    }
  }

  return res;
}

// Pretty print to ostream a bfp16ebs8 array
inline void printBfp16ebs8Array(int arraySize, std::vector<uint8_t> array, int blocksPerLine = 4,
                                int blocksBeforeEmptyLine = 8, std::ostream &ostream = std::cout,
                                int width = 3, const std::string &blockSeparatorStart = " | B",
                                const std::string &blockSeparatorEnd = " - ") {
  for (int i = 0; i < arraySize; i++) {
    if (i % (blocksPerLine * 9) == 0) {
      ostream << "\n";
      if (i % (blocksBeforeEmptyLine * 9) == 0) {
        ostream << "\n";
      }
    }

    if (i % 9 == 0) {
      ostream << blockSeparatorStart << std::setw(width) << i / 9 << " - ";
    }

    ostream << std::setw(4) << int(array[i]);
  }

  ostream << std::endl;
}
