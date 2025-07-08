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

extern "C" {

void bf16_to_bfp_conversion(bfloat16 *__restrict inA, bfloat16 *__restrict inB,
                            bfp16ebs8 *__restrict outC) {
  // The conversion for inA to bfp16 is carried out from an accumulator
  aie::vector<bfloat16, 64> vecA = aie::load_v<64>(inA);
  aie::accum<accfloat, 64> accA(vecA);

  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> outStreamC(outC);
  outStreamC << accA.to_vector<bfp16ebs8>();
}

} // extern "C"
