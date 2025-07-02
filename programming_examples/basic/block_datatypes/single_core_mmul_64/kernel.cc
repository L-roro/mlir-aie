//===- kernel.cc ------------------------------------------------*- C++ -*-===//
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

// This kernel mirrors the one found in https://xilinx.github.io/aie_api/group__group__mmul.html
// Go through them in parallel to understand how the bfp datatype modifies accesses to memory
template <unsigned m, unsigned k, unsigned n, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_bfp16(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC) {
  const unsigned size_A = r * s;
  const unsigned size_B = s * t;
  const unsigned size_C = r * t;

  for (unsigned z = 0; z < m; z += 2) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC1_in(pC);
    pC1_in.seek(z * n);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC2_in(pC);
    pC2_in.seek((z + 1) * n);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC1_out(pC);
    pC1_out.seek(z * n);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC2_out(pC);
    pC2_out.seek((z + 1) * n);

    for (unsigned j = 0; j < n; j += 2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA1bfp16(pA);
      pA1bfp16.seek(z * k);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA2bfp16(pA);
      pA2bfp16.seek((z + 1) * k);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
      pB1bfp16.seek(j);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
      pB2bfp16.seek(j + 1);

      aie::block_vector<bfp16ebs8, size_A> A0 = pA1bfp16.pop();
      aie::block_vector<bfp16ebs8, size_A> A1 = pA2bfp16.pop();
      aie::block_vector<bfp16ebs8, size_B> B0 = pB1bfp16.pop_seek(n - 1);
      aie::block_vector<bfp16ebs8, size_B> B1 = pB2bfp16.pop_seek(n - 1);

      aie::accum<accfloat, size_C> acc_C00(pC1_in.pop());
      aie::accum<accfloat, size_C> acc_C01(pC1_in.pop());

      aie::accum<accfloat, size_C> acc_C10(pC2_in.pop());
      aie::accum<accfloat, size_C> acc_C11(pC2_in.pop());

      acc_C00 = mac_8x8_8x8T(A0, B0, acc_C00);
      acc_C01 = mac_8x8_8x8T(A0, B1, acc_C01);
      acc_C10 = mac_8x8_8x8T(A1, B0, acc_C10);
      acc_C11 = mac_8x8_8x8T(A1, B1, acc_C11);

      for (unsigned i = 1; i < k; ++i) {
        A0 = pA1bfp16.pop();
        A1 = pA2bfp16.pop();

        B0 = pB1bfp16.pop_seek(n - 1);
        B1 = pB2bfp16.pop_seek(n - 1);

        acc_C00 = mac_8x8_8x8T(A0, B0, acc_C00);
        acc_C01 = mac_8x8_8x8T(A0, B1, acc_C01);
        acc_C10 = mac_8x8_8x8T(A1, B0, acc_C10);
        acc_C11 = mac_8x8_8x8T(A1, B1, acc_C11);
      }

      pC1_out.push(acc_C00.template to_vector<bfp16ebs8>());
      pC1_out.push(acc_C01.template to_vector<bfp16ebs8>());
      pC2_out.push(acc_C10.template to_vector<bfp16ebs8>());
      pC2_out.push(acc_C11.template to_vector<bfp16ebs8>());
    }
  }
}

extern "C" {
void matmul_testing_kernel(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                           bfp16ebs8 *__restrict pC) {

  matmul_vectorized_bfp16<8, 8, 8, 8, 8, 8>(pA, pB, pC);
}

void zero_kernel(bfp16ebs8 *__restrict cOut) { zero_vectorized_v64bfp16ebs8<64, 64>(cOut); }
}
