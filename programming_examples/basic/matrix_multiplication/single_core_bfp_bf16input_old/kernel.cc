#include <aie_api/aie.hpp>

// =========Zero the output array in bfp format=========

// This function has been tested and works as expected so far, no need for extra
// work here. Focus on the rest. This function writes a zero vector of bfp16ebs8
// M and N represent the dimensions of the output matrix (number of scalar
// elements)
template <int M, int N>
void zero_vectorized_v64bfp16ebs8(bfp16ebs8 *__restrict cOut) {
  int const vectorSize = 64;

  // not supported!
  // const aie::bfp_vector<bfp16ebs8, vectorSize> zeros = aie::zeros<bfp16ebs8,
  // r>();

  const aie::accum<accfloat, vectorSize> acc =
      aie::zeros<accfloat, vectorSize>();

  aie::block_vector_output_buffer_stream<bfp16ebs8, vectorSize> outStreamC(
      cOut);
  // TODO: Correct M and N to be the number of scalar elements instead of blocks
  for (int i = 0; i < M * N; i++) {
    outStreamC << acc.to_vector<bfp16ebs8>();
  }
}

// =========Use bfp16ebs8 on bf16 input, nowhere near close to working=========

// template <unsigned m, unsigned k, unsigned n, unsigned r, unsigned s,
//           unsigned t>
// void matmul_vectorized_2x2_bf16_bf16(bfloat16 *__restrict pA,
//                                      bfloat16 *__restrict pB,
//                                      bfloat16 *__restrict pC) {
//   // I am not going to use r, s and t for now and just focus on getting the
//   8x8
//   // example working for now. Generalization may come later
//   // This sizes correspond to the size of one of the submatrices of the
//   core's
//   // tile from the full matrix
//   const unsigned sizeA = r * s; // For example, 8x8 = 64
//   const unsigned sizeB = s * t;
//   const unsigned sizeC = r * t;
//   for (unsigned z = 0; z < m; z += 2) {
//     // We are going to use 2x2 blocks. We want pointers to each one of the
//     // output rows in C
//     bfloat16 *__restrict pC1 = pC + (z * n) * sizeC;
//     bfloat16 *__restrict pC2 = pC + ((z + 1) * n) * sizeC;

//     for (unsigned j = 0; j < n; j += 2) {
//       const bfloat16 *__restrict pA1 = pA + (z * k + 0) * sizeA;
//       const bfloat16 *__restrict pA2 = pA + ((z + 1) * k + 0) * sizeA;
//       const bfloat16 *__restrict pB1 = pB + (0 * n + j) * sizeB;
//       const bfloat16 *__restrict pB2 = pB + (0 * n + (j + 1)) * sizeB;

//       aie::vector<bfloat16, sizeA> A0 = aie::load_v<sizeA>(pA1);
//       pA1 += sizeA;
//       aie::vector<bfloat16, sizeA> A1 = aie::load_v<sizeA>(pA2);
//       pA2 += sizeA;
//       aie::vector<bfloat16, sizeB> B0 = aie::load_v<sizeB>(pB1);
//       pB1 += sizeB * n;
//       aie::vector<bfloat16, sizeB> B1 = aie::load_v<sizeB>(pB2);
//       pB2 += sizeB * n;

//       // Perform bf16 to bfp16 conversions
//       // This is serialized, check slot usage
//       aie::accum<accfloat, 64> accA0(A0);
//       aie::accum<accfloat, 64> accA1(A1);
//       aie::accum<accfloat, 64> accB0(B0);
//       aie::accum<accfloat, 64> accB1(B1);

//       v64bfp16ebs8 A0bfp16 = accA0.template to_vector<bfp16ebs8>();
//       v64bfp16ebs8 A1bfp16 = accA1.template to_vector<bfp16ebs8>();
//       v64bfp16ebs8 B0bfp16 = accB0.template to_vector<bfp16ebs8>();
//       v64bfp16ebs8 B1bfp16 = accB1.template to_vector<bfp16ebs8>();

//       // TODO: Check if the code above is equivalent to to_v64bfp16ebs8(acc)

//       // Load partial results from C buffer for accumulation in-place. The
//       // zero.cc function handles the zeroing of data when a new
//       // accumulation is needed (after the 'K' reduction dimension)
//       aie::accum<accfloat, sizeC> accC00(aie::load_v<sizeC>(pC1));
//       aie::accum<accfloat, sizeC> accC01(aie::load_v<sizeC>(pC1 + sizeC));
//       aie::accum<accfloat, sizeC> accC10(aie::load_v<sizeC>(pC2));
//       aie::accum<accfloat, sizeC> accC11(aie::load_v<sizeC>(pC2 + sizeC));

//       accC00 = mac_8x8_8x8T(A0bfp16, B0bfp16, accC00);
//       accC01 = mac_8x8_8x8T(A0bfp16, B1bfp16, accC01);
//       accC10 = mac_8x8_8x8T(A1bfp16, B0bfp16, accC10);
//       accC11 = mac_8x8_8x8T(A1bfp16, B1bfp16, accC11);

//       for (unsigned i = 1; i < k; ++i) {
//         A0 = aie::load_v<sizeA>(pA1);
//         pA1 += sizeA;
//         A1 = aie::load_v<sizeA>(pA2);
//         pA2 += sizeA;
//         B0 = aie::load_v<sizeB>(pB1);
//         pB1 += sizeB * n;
//         B1 = aie::load_v<sizeB>(pB2);
//         pB2 += sizeB * n;

//         v64bfp16ebs8 A0bfp16 = accA0.template to_vector<bfp16ebs8>();
//         v64bfp16ebs8 A1bfp16 = accA1.template to_vector<bfp16ebs8>();
//         v64bfp16ebs8 B0bfp16 = accB0.template to_vector<bfp16ebs8>();
//         v64bfp16ebs8 B1bfp16 = accB1.template to_vector<bfp16ebs8>();

//         accC00 = mac_8x8_8x8T(A0bfp16, B0bfp16, accC00);
//         accC01 = mac_8x8_8x8T(A0bfp16, B1bfp16, accC01);
//         accC10 = mac_8x8_8x8T(A1bfp16, B0bfp16, accC10);
//         accC11 = mac_8x8_8x8T(A1bfp16, B1bfp16, accC11);
//       }

//       // TODO make shift right here to keep most significat bits
//       // when lowering the output
//       // example below shows how to shift right 10 bits
//       // #define SHIFT 10
//       // aie::store_v(pC1, C00.template to_vector<bfloat16>(SHIFT));
//       aie::store_v(pC1, accC00.template to_vector<bfloat16>());
//       pC1 += sizeC;
//       aie::store_v(pC1, accC01.template to_vector<bfloat16>());
//       pC1 += sizeC;
//       aie::store_v(pC2, accC10.template to_vector<bfloat16>());
//       pC2 += sizeC;
//       aie::store_v(pC2, accC11.template to_vector<bfloat16>());
//       pC2 += sizeC;
//     }
//   }
// }

// =========Make an identical kernel for int16 and bf16=========

// Command: make clean && make run M=16 K=16 N=16 m=16 k=16 n=16 runargs="-v 2 --warmup 1 --iters 1" dtype_in=bf16 dtype_out=bf16

template <typename T, unsigned m, unsigned k, unsigned n, unsigned r,
          unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul(const T *__restrict pA,
                                              const T *__restrict pB,
                                              T *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T, T, accauto>;

  for (unsigned int z = 0; z < m; z += 2) {
    T *__restrict pC1 = pC + (z * n + 0) * MMUL::size_C;
    T *__restrict pC2 = pC + ((z + 1) * n + 0) * MMUL::size_C;

    for (unsigned int j = 0; j < n; j += 2) {
      const T *__restrict pA1 = pA + (z * k + 0) * MMUL::size_A;
      const T *__restrict pA2 = pA + ((z + 1) * k + 0) * MMUL::size_A;
      const T *__restrict pB1 = pB + (0 * n + j) * MMUL::size_B;
      const T *__restrict pB2 = pB + (0 * n + (j + 1)) * MMUL::size_B;

      aie::vector<T, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      pA1 += MMUL::size_A;
      aie::vector<T, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
      pA2 += MMUL::size_A;
      aie::vector<T, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B * n;
      aie::vector<T, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
      pB2 += MMUL::size_B * n;

      aie::vector<T, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
      aie::vector<T, MMUL::size_C> acc_C01 =
          aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
      aie::vector<T, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
      aie::vector<T, MMUL::size_C> acc_C11 =
          aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

      MMUL C00(acc_C00);
      MMUL C01(acc_C01);
      MMUL C10(acc_C10);
      MMUL C11(acc_C11);

      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C10.mac(A1, B0);
      C11.mac(A1, B1);

      for (unsigned int i = 1; i < k; ++i) {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B * n;
        B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B * n;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      aie::store_v(pC1, C00.template to_vector<T>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<T>());
      pC1 += MMUL::size_C;
      aie::store_v(pC2, C10.template to_vector<T>());
      pC2 += MMUL::size_C;
      aie::store_v(pC2, C11.template to_vector<T>());
      pC2 += MMUL::size_C;
    }
  }
}

// =========Debug the array configuration with static values=========
// template <unsigned m, unsigned k, unsigned n,
// unsigned r, unsigned s,
//           unsigned t>
// void matmul_vectorized_2x2_bf16_bf16(bfloat16 *__restrict pA,
//                                      bfloat16 *__restrict pB,
//                                      bfloat16 *__restrict pC) {

//   const unsigned sizeA = r * s; // For example, 8x8 = 64
//   const unsigned sizeB = s * t;
//   const unsigned sizeC = r * t;

//   // I am just trying to make sense of what is going on here:
//   // || p1 p3 | ...
//   // || p2 p4 | ...
//   // ||  ...  |
//   for (unsigned z = 0; z < m; z += 2) {
//     // We are going to use 2x2 blocks. We want pointers to each one of the
//     // output rows in C
//     bfloat16 *__restrict pC1 = pC + (z * n) * sizeC;
//     bfloat16 *__restrict pC2 = pC + ((z + 1) * n) * sizeC;

//     for (unsigned j = 0; j < n; j += 2) {
//       const bfloat16 *__restrict pA1 = pA + (z * k + 0) * sizeA;
//       const bfloat16 *__restrict pA2 = pA + ((z + 1) * k + 0) * sizeA;

//       aie::vector<bfloat16, sizeC> C00 =
//           aie::broadcast<bfloat16, sizeC>(z * 4 + j * 2);
//       aie::vector<bfloat16, sizeC> C01 =
//           aie::broadcast<bfloat16, sizeC>(z * 4 + j * 2 + 1);
//       aie::vector<bfloat16, sizeC> C10 =
//           aie::broadcast<bfloat16, sizeC>(z * 4 + j * 2 + 2);
//       aie::vector<bfloat16, sizeC> C11 =
//           aie::broadcast<bfloat16, sizeC>(z * 4 + j * 2 + 3);

//       aie::store_v(pC1, C00);
//       pC1 += sizeC;
//       aie::store_v(pC1, C01);
//       pC1 += sizeC;
//       aie::store_v(pC2, C10);
//       pC2 += sizeC;
//       aie::store_v(pC2, C11);
//       pC2 += sizeC;
//     }
//   }
// }

// =========Minimize the code breaking for the bf16 multiplication=========
// This is too simple, it works
// Command: make clean && make run M=8 K=8 N=8 m=8 k=8 n=8 runargs="-v 2
// --warmup 1 --iters 1" dtype_in=bf16 dtype_out=bf16

// template <typename T, unsigned r, unsigned s, unsigned t>
// static inline void matmul_vectorized_2x2_mmul(const T *__restrict pA,
//                                               const T *__restrict pB,
//                                               T *__restrict pC) {
//   using MMUL = aie::mmul<r, s, t, T, T, accauto>;

//   aie::vector<T, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA);
//   aie::vector<T, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB);

//   aie::vector<T, MMUL::size_C> accC00 = aie::load_v<MMUL::size_C>(pC);

//   MMUL C00(accC00);

//   C00.mac(A0, B0);

//   aie::store_v(pC, C00.template to_vector<T>());
// }

// =========Minimize the code breaking for the bf16 multiplication=========
// Idea: Simply unroll all the loops and manually multiply the matrix to
// complexify the example above on a very simple 2x2 matrix multiply example 00
// 01 10 11
// Command: make clean && make run M=4 K=8 N=4 m=4 k=8 n=4 runargs="-v 2
// --warmup 1 --iters 1" dtype_in=bf16 dtype_out=bf16

// template <typename T, unsigned r, unsigned s, unsigned t>
// static inline void matmul_vectorized_2x2_mmul(const T *__restrict pA,
//                                               const T *__restrict pB,
//                                               T *__restrict pC) {
//   using MMUL = aie::mmul<r, s, t, T, T, accauto>;

//   aie::vector<T, MMUL::size_A> A00 = aie::load_v<MMUL::size_A>(pA);
//   aie::vector<T, MMUL::size_A> A01 =
//       aie::load_v<MMUL::size_A>(pA + MMUL::size_A);
//   aie::vector<T, MMUL::size_A> A10 =
//       aie::load_v<MMUL::size_A>(pA + 2 * MMUL::size_A);
//   aie::vector<T, MMUL::size_A> A11 =
//       aie::load_v<MMUL::size_A>(pA + 3 * MMUL::size_A);
//   aie::vector<T, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pB);
//   aie::vector<T, MMUL::size_B> B01 =
//       aie::load_v<MMUL::size_B>(pB + MMUL::size_B);
//   aie::vector<T, MMUL::size_B> B10 =
//       aie::load_v<MMUL::size_B>(pB + 2 * MMUL::size_B);
//   aie::vector<T, MMUL::size_B> B11 =
//       aie::load_v<MMUL::size_B>(pB + 3 * MMUL::size_B);

//   aie::vector<T, MMUL::size_C> accC00 = aie::load_v<MMUL::size_C>(pC);
//   aie::vector<T, MMUL::size_C> accC01 =
//       aie::load_v<MMUL::size_C>(pC + MMUL::size_C);
//   aie::vector<T, MMUL::size_C> accC10 =
//       aie::load_v<MMUL::size_C>(pC + 2 * MMUL::size_C);
//   aie::vector<T, MMUL::size_C> accC11 =
//       aie::load_v<MMUL::size_C>(pC + 3 * MMUL::size_C);

//   MMUL C00(accC00);
//   MMUL C01(accC01);
//   MMUL C10(accC10);
//   MMUL C11(accC11);

//   C00.mac(A00, B00);
//   C00.mac(A01, B10);

//   C01.mac(A00, B01);
//   C01.mac(A01, B11);

//   C10.mac(A10, B00);
//   C10.mac(A11, B10);

//   C11.mac(A10, B01);
//   C11.mac(A11, B11);

//   aie::store_v(pC, C00.template to_vector<T>());
//   aie::store_v(pC + MMUL::size_C, C01.template to_vector<T>());
//   aie::store_v(pC + 2 * MMUL::size_C, C10.template to_vector<T>());
//   aie::store_v(pC + 3 * MMUL::size_C, C11.template to_vector<T>());
// }

extern "C" {

// For the bf16 to bfp16ebs8 conversion testing
// void matmul_testing_kernel2(bfloat16 *__restrict pA, bfloat16 *__restrict pB,
//                             bfloat16 *__restrict pC) {
//   matmul_vectorized_2x2_bf16_bf16<4, 4, 4, 8, 8, 8>(pA, pB, pC);
// }

// For the int16 testing
// void matmul_testing_kernel2(int16 *__restrict pA, int16 *__restrict pB,
//                             int16 *__restrict pC) {
//   matmul_vectorized_2x2_mmul<int16, 4, 4, 2, 4, 4, 8>(pA, pB, pC);
// }

// For the bf16 testing
void matmul_testing_kernel2(bfloat16 *__restrict pA, bfloat16 *__restrict pB,
                            bfloat16 *__restrict pC) {
  // matmul_vectorized_2x2_mmul<bfloat16, 4, 2, 4, 4, 8, 4>(pA, pB, pC);
  matmul_vectorized_2x2_mmul<bfloat16, 4, 2, 4, 4, 8, 4>(pA, pB, pC);
}

// For the minimization of code breaking
// void matmul_testing_kernel2(bfloat16 *__restrict pA, bfloat16 *__restrict pB,
//                             bfloat16 *__restrict pC) {
//   matmul_vectorized_2x2_mmul<bfloat16, 8, 8, 8>(pA, pB, pC);
// }

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  zero_vectorized_v64bfp16ebs8<32, 32>(cOut);
}
}
