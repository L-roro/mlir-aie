//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Clangd fix, remove
#ifdef _CLANGD
namespace std {
using bfloat16_t = double;
} // namespace std
#endif

#include "../common.h"
#include "helper.h"

using EXTERNAL_DATATYPE = float;
using BFP_C_DATATYPE = uint8_t;

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 1024 * 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
// TODO: This might have to be adjusted for bfp
float abs_tol = matmul_common::get_abs_tol<std::bfloat16_t>();
float rel_tol = matmul_common::get_rel_tol<std::bfloat16_t>();

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  // TODO: Remove this
  do_verify = 0;
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); // srand(time(NULL));

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();
  bool do_verify_stochastic =
      (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  int A_VOLUME = M * K;
  int B_VOLUME = N * K;
  int C_VOLUME = M * N;

  size_t A_SIZE = (A_VOLUME * sizeof(uint8_t)) * 1.125;
  size_t B_SIZE = (B_VOLUME * sizeof(uint8_t)) * 1.125;
  size_t C_SIZE = (C_VOLUME * sizeof(uint8_t)) * 1.125;

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // ------------------------------------------------------
  // Initialize input/output buffer sizes and sync them
  // ------------------------------------------------------

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size * 4, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  // ------------------------------------------------------
  // Generate data for buffers
  // ------------------------------------------------------
  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  BFP_C_DATATYPE *bufA = bo_a.map<BFP_C_DATATYPE *>();
  std::vector<EXTERNAL_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    // AVec[i] = matmul_common::get_random<EXTERNAL_DATATYPE>();
    AVec[i] = i;
    // if (i % N == i / N) {
    //   AVec[i] = 1.0;
    // } else {
    //   AVec[i] = 0.0;
    // }
  }

  BFP_C_DATATYPE *bufB = bo_b.map<BFP_C_DATATYPE *>();
  std::vector<EXTERNAL_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    // BVec[i] = matmul_common::get_random<B_DATATYPE>() * i;
    // Diagonal:
    if (i % N == i / N) {
      BVec[i] = 1.0;
    } else {
      BVec[i] = 0.0;
    }
  }

  std::vector<uint8_t> AVecBlocked(A_VOLUME * 1.125);
  bfp16QuantFloat(8, A_VOLUME, AVec.data(), AVecBlocked.data(), 0, 0);

  // Debugging:
  for (int i = 0; i < A_SIZE; i++) {
    ((uint8_t *)AVecBlocked.data())[i] = i / 9;
  }

  // TODO: B will have to be transposed, ignoring this right now since I am
  // trying out the identity matrix
  std::vector<uint8_t> BVecBlocked(B_VOLUME * 1.125);
  bfp16QuantFloat(8, B_VOLUME, BVec.data(), BVecBlocked.data(), 0, 0);

  // ------------------------------------------------------
  // Write data into buffers
  // ------------------------------------------------------
  memcpy(bufA, AVecBlocked.data(), A_SIZE);
  memcpy(bufB, BVecBlocked.data(), B_SIZE);

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  std::vector<uint8_t> CVecBlocked(C_VOLUME * 1.125);
  memset(bufOut, 0, C_SIZE);

  char *bufTrace = bo_trace.map<char *>();
  if (trace_size > 0)
    memset(bufTrace, 0, trace_size);

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    std::cout << "Verification tolerance " << abs_tol << " absolute, "
              << rel_tol << " relative.\n";
    std::cout << "A = \n";
    matmul_common::print_matrix(AVec, K);
    std::cout << "B = \n";
    matmul_common::print_matrix(BVec, N);
  }

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Run kernel
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;
  float macs = 2.0 * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel (iteration " << iter << ").\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out,
                      bo_tmp1, bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // ------------------------------------------------------
    // Check output
    // ------------------------------------------------------
    // std::cout << "Expected:\n";
    // for (uint32_t i = 0; i < C_SIZE; i++) {
    //   if (i % 36 == 0) {
    //     std::cout << "\n";
    //     if (i % 64 == 0) {
    //       std::cout << "\n";
    //     }
    //   }

    //   if (i % 9 == 0) {
    //     std::cout << " | B" << std::setw(3) << i / 9 << " - ";
    //   }

    //   std::cout << std::setw(4) << int(*(AVecBlocked.data() + i));
    // }

    std::cout << "\nOutput:\n";

    std::vector<EXTERNAL_DATATYPE> CVec(C_VOLUME);
    memcpy(CVecBlocked.data(), bufOut, C_SIZE);

    for (uint32_t i = 0; i < C_SIZE; i++) {
      if (i % 36 == 0) {
        std::cout << "\n";
        if (i % 72 == 0) {
          std::cout << "\n";
        }
      }

      if (i % 9 == 0) {
        std::cout << " | B" << std::setw(3) << i / 9 << " - ";
      }

      std::cout << std::setw(4) << int(*(bufOut + i));
    }

    if (do_verify) {
      memcpy(CVecBlocked.data(), bufOut, C_SIZE);

      for (uint32_t i = 0; i < C_SIZE; i++) {
        if (i % 9 == 0) {
          std::cout << "Block " << i / 9 << "\n";
        }

        std::cout << int(*(bufOut + i)) << std::endl;
      }

      // Do the conversion from bfp16 to float
      bfp16ebs8ToFloat(C_VOLUME * 1.125, CVecBlocked.data(), CVec.data(), 0);
      if (verbosity >= 1) {
        if (do_verify_stochastic) {
          std::cout << "Verifying " << verify_stochastic_n_samples
                    << " random samples against reference matmul ..."
                    << std::endl;
        } else {
          std::cout << "Verifying against reference matmul ..." << std::endl;
        }
      }
      auto vstart = std::chrono::system_clock::now();
      if (do_verify_stochastic) {
        errors = matmul_common::verify_stochastic<
            EXTERNAL_DATATYPE, EXTERNAL_DATATYPE, EXTERNAL_DATATYPE>(
            M, N, K, AVec, BVec, CVec, verify_stochastic_n_samples, verbosity,
            abs_tol, rel_tol, b_col_maj);
      } else {
        errors = matmul_common::verify<EXTERNAL_DATATYPE, EXTERNAL_DATATYPE,
                                       EXTERNAL_DATATYPE>(
            M, N, K, AVec, BVec, CVec, verbosity, abs_tol, rel_tol, b_col_maj);
      }
      auto vstop = std::chrono::system_clock::now();

      matmul_common::print_matrix(CVec, N);

      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << " s." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // Only write out trace of last iteration.
  if (trace_size > 0) {
    matmul_common::write_out_trace((char *)bufTrace, trace_size,
                                   vm["trace_file"].as<std::string>());
  }

  // ------------------------------------------------------
  // Output results
  // ------------------------------------------------------
  std::cout << std::endl
            << "Avg NPU matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }

  std::cout << "\nError count: " << errors;
  if (do_verify_stochastic) {
    std::cout << " (out of " << verify_stochastic_n_samples
              << " random samples)";
  }
  std::cout << "\n\n";

  std::cout << "\nFailed.\n\n";
  return 1;
}
