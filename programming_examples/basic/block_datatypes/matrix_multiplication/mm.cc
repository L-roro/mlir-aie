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

template <int M, int N>
void zero_vectorized_v64bfp16ebs8(bfp16ebs8 *__restrict cOut) {
  int const vectorSize = 64;

  const aie::accum<accfloat, vectorSize> acc = aie::zeros<accfloat, vectorSize>();

  aie::block_vector_output_buffer_stream<bfp16ebs8, vectorSize> outStreamC(cOut);

  for (int i = 0; i < M * N / 64; i++) {
    outStreamC << acc.to_vector<bfp16ebs8>();
  }
}

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// This kernel mirrors the one found in https://xilinx.github.io/aie_api/group__group__mmul.html
// Go through them in parallel to understand how the bfp datatype modifies accesses to memory
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_2x2_bfp16(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB,
                                 bfp16ebs8 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  for (unsigned z = 0; z < rowA; z += 2) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC1In(pC);
    pC1In.seek(z * colB);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC2In(pC);
    pC2In.seek((z + 1) * colB);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC1Out(pC);
    pC1Out.seek(z * colB);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC2Out(pC);
    pC2Out.seek((z + 1) * colB);

    for (unsigned j = 0; j < colB; j += 2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA1bfp16(pA);
      pA1bfp16.seek(z * colA);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA2bfp16(pA);
      pA2bfp16.seek((z + 1) * colA);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
      pB1bfp16.seek(j);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
      pB2bfp16.seek(j + 1);

      aie::block_vector<bfp16ebs8, sizeA> A0 = pA1bfp16.pop();
      aie::block_vector<bfp16ebs8, sizeA> A1 = pA2bfp16.pop();
      aie::block_vector<bfp16ebs8, sizeB> B0 = pB1bfp16.pop_seek(colB - 1);
      aie::block_vector<bfp16ebs8, sizeB> B1 = pB2bfp16.pop_seek(colB - 1);

      // Note that unlike the example mentioned above, we need
      // to use a mac to take into account results from previous kernel calls
      // but this is completely unrelated to the block datatype.
      aie::accum<accfloat, sizeC> accC00(pC1In.pop());
      aie::accum<accfloat, sizeC> accC01(pC1In.pop());

      aie::accum<accfloat, sizeC> accC10(pC2In.pop());
      aie::accum<accfloat, sizeC> accC11(pC2In.pop());

      accC00 = mac_8x8_8x8T(A0, B0, accC00);
      accC01 = mac_8x8_8x8T(A0, B1, accC01);
      accC10 = mac_8x8_8x8T(A1, B0, accC10);
      accC11 = mac_8x8_8x8T(A1, B1, accC11);

      for (unsigned i = 1; i < colA; ++i) {
        A0 = pA1bfp16.pop();
        A1 = pA2bfp16.pop();

        B0 = pB1bfp16.pop_seek(colB - 1);
        B1 = pB2bfp16.pop_seek(colB - 1);

        accC00 = mac_8x8_8x8T(A0, B0, accC00);
        accC01 = mac_8x8_8x8T(A0, B1, accC01);
        accC10 = mac_8x8_8x8T(A1, B0, accC10);
        accC11 = mac_8x8_8x8T(A1, B1, accC11);
      }

      pC1Out.push(accC00.template to_vector<bfp16ebs8>());
      pC1Out.push(accC01.template to_vector<bfp16ebs8>());
      pC2Out.push(accC10.template to_vector<bfp16ebs8>());
      pC2Out.push(accC11.template to_vector<bfp16ebs8>());
    }
  }
}

// This kernel is a variation of the conventional matrix multiplications in the repo that uses
// different datatypes for the A and B.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_2x2_bfp16_bf16(const bfloat16 *__restrict pA, const bfp16ebs8 *__restrict pB,
                                      bfloat16 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  for (unsigned z = 0; z < rowA; z += 2) {
    bfloat16 *__restrict pC1 = pC + (z * colB + 0) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB + 0) * sizeC;

    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA1 = pA + (z * colA + 0) * sizeA;
      const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA + 0) * sizeA;

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
      pB1bfp16.seek(j);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
      pB2bfp16.seek(j + 1);

      aie::vector<bfloat16, sizeA> A0 = aie::load_v<sizeA>(pA1);
      pA1 += sizeA;
      aie::vector<bfloat16, sizeA> A1 = aie::load_v<sizeA>(pA2);
      pA2 += sizeA;
      aie::block_vector<bfp16ebs8, sizeB> B0 = pB1bfp16.pop_seek(colB - 1);
      aie::block_vector<bfp16ebs8, sizeB> B1 = pB2bfp16.pop_seek(colB - 1);

      aie::accum<accfloat, sizeC> accC00(aie::load_v<sizeC>(pC1));
      aie::accum<accfloat, sizeC> accC01(aie::load_v<sizeC>(pC1 + sizeC));
      aie::accum<accfloat, sizeC> accC10(aie::load_v<sizeC>(pC2));
      aie::accum<accfloat, sizeC> accC11(aie::load_v<sizeC>(pC2 + sizeC));

      // Convert A0 into bfp16
      aie::accum<accfloat, 64> accA0(A0);
      // Convert A1 into bfp16 through a different VLIW slot (see bfp conversion example)
      aie::accum<accfloat, 64> accA1 =
          mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(), broadcast_one_to_v32bfloat16()));

      accC00 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B0, accC00);
      accC01 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B1, accC01);
      accC10 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B0, accC10);
      accC11 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B1, accC11);

      for (unsigned i = 1; i < colA; ++i) {
        A0 = aie::load_v<sizeA>(pA1);
        pA1 += sizeA;
        A1 = aie::load_v<sizeA>(pA2);
        pA2 += sizeA;

        // Convert A0 into bfp16
        accA0 = A0;
        // Convert A1 into bfp16 through a different VLIW slot (see bfp conversion example)
        accA1 =
            mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(), broadcast_one_to_v32bfloat16()));

        B0 = pB1bfp16.pop_seek(colB - 1);
        B1 = pB2bfp16.pop_seek(colB - 1);

        accC00 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B0, accC00);
        accC01 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B1, accC01);
        accC10 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B0, accC10);
        accC11 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B1, accC11);
      }

      aie::store_v(pC1, accC00.template to_vector<bfloat16>());
      pC1 += sizeC;
      aie::store_v(pC1, accC01.template to_vector<bfloat16>());
      pC1 += sizeC;
      aie::store_v(pC2, accC10.template to_vector<bfloat16>());
      pC2 += sizeC;
      aie::store_v(pC2, accC11.template to_vector<bfloat16>());
      pC2 += sizeC;
    }
  }
}

extern "C" {
void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC) {

  matmul_vectorized_2x2_bfp16<8, 8, 8, 8, 8, 8>(pA, pB, pC);
}

void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA, bfp16ebs8 *__restrict pB,
                                           bfloat16 *__restrict pC) {

  matmul_vectorized_2x2_bfp16_bf16<8, 8, 8, 8, 8, 8>(pA, pB, pC);
}

void zero_kernel(bfp16ebs8 *__restrict cOut) { zero_vectorized_v64bfp16ebs8<64, 64>(cOut); }

void zero_kernel_bf16(bfloat16 *__restrict cOut) { zero_vectorized<bfloat16, 64, 64>(cOut); }
}
