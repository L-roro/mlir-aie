#include <aie2p_aie_api_compat.h>
#include <aie2p_srs.h>
#include <aie_api/aie.hpp>
#include <aiebase_typedefs.h>

// This function writes a zero vector of bfp16ebs8
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

template <unsigned m, unsigned k, unsigned n, unsigned r, unsigned s,
          unsigned t>
void matmul_vectorized_2x2_bfp16_bfp16(const int8 *__restrict pAi8,
                                       const int8 *__restrict pBi8,
                                       int8 *__restrict pCi8) {
  // 8 x 8 = 64 if this is not 64, then we need to scale the number passed to
  // seek() and pop_seek()
  const unsigned size_A = r * s;
  const unsigned size_B = s * t;
  const unsigned size_C = r * t;
  // m = k = n = 64/8 = 8

  for (unsigned z = 0; z < m; z += 2)
    chess_loop_range(4, ) {
      aie::bfp_vector_input_buffer_stream<bfp16ebs8, 64> pC1bfp16_in(
          (bfp16ebs8 *)pCi8);
      pC1bfp16_in.seek(z * n);
      aie::bfp_vector_input_buffer_stream<bfp16ebs8, 64> pC2bfp16_in(
          (bfp16ebs8 *)pCi8);
      pC2bfp16_in.seek((z + 1) * n);
      aie::bfp_vector_output_buffer_stream<bfp16ebs8, 64> pC1bfp16_out(
          (bfp16ebs8 *)pCi8);
      pC1bfp16_out.seek(z * n);
      aie::bfp_vector_output_buffer_stream<bfp16ebs8, 64> pC2bfp16_out(
          (bfp16ebs8 *)pCi8);
      pC2bfp16_out.seek((z + 1) * n);

      for (unsigned j = 0; j < n; j += 2)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
          aie::bfp_vector_input_buffer_stream<bfp16ebs8, 64> pA1bfp16(
              (bfp16ebs8 *)pAi8);
          pA1bfp16.seek(z * k);
          aie::bfp_vector_input_buffer_stream<bfp16ebs8, 64> pA2bfp16(
              (bfp16ebs8 *)pAi8);
          pA2bfp16.seek((z + 1) * k);

          aie::bfp_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(
              (bfp16ebs8 *)pBi8);
          pB1bfp16.seek(j * k);
          aie::bfp_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(
              (bfp16ebs8 *)pBi8);
          pB2bfp16.seek((j + 1) * k);

          aie::bfp_vector<bfp16ebs8, size_A> A0 = pA1bfp16.pop();
          aie::bfp_vector<bfp16ebs8, size_A> A1 = pA2bfp16.pop();
          aie::bfp_vector<bfp16ebs8, size_B> B0 = pB1bfp16.pop();
          // pB1bfp16.seek(n-1);
          aie::bfp_vector<bfp16ebs8, size_B> B1 = pB2bfp16.pop();
          // pB2bfp16.seek(n-1);

          aie::bfp_vector<bfp16ebs8, size_C> C00 = pC1bfp16_in.pop();
          aie::accum<accfloat, size_C> acc_C00 =
              aie::accum<accfloat, size_C>(C00);
          aie::bfp_vector<bfp16ebs8, size_C> C01 = pC1bfp16_in.pop();
          aie::accum<accfloat, size_C> acc_C01 =
              aie::accum<accfloat, size_C>(C01);
          // pC1bfp16_in.seek(-2);

          aie::bfp_vector<bfp16ebs8, size_C> C10 = pC2bfp16_in.pop();
          aie::accum<accfloat, size_C> acc_C10 =
              aie::accum<accfloat, size_C>(C10);
          aie::bfp_vector<bfp16ebs8, size_C> C11 = pC2bfp16_in.pop();
          aie::accum<accfloat, size_C> acc_C11 =
              aie::accum<accfloat, size_C>(C11);
          // pC2bfp16_in.seek(-2);

          // v64accfloat 	mac_8x8_8x8T_conf (v64bfp16ebs8 a, int sgn_x,
          // v64bfp16ebs8 b, int sgn_y, v64accfloat acc, int zero_acc, int
          // sub_mul, int sub_acc1)
          acc_C00 = mac_8x8_8x8T_conf(A0, 1, B0, 1, acc_C00, 0, 0, 0);
          acc_C01 = mac_8x8_8x8T_conf(A0, 1, B1, 1, acc_C01, 0, 0, 0);
          acc_C10 = mac_8x8_8x8T_conf(A1, 1, B0, 1, acc_C10, 0, 0, 0);
          acc_C11 = mac_8x8_8x8T_conf(A1, 1, B1, 1, acc_C11, 0, 0, 0);

          for (unsigned i = 1; i < k; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              A0 = pA1bfp16.pop();
              A1 = pA2bfp16.pop();

              B0 = pB1bfp16.pop();
              // pB1bfp16.seek(n-1);
              B1 = pB2bfp16.pop();
              // pB2bfp16.seek(n-1);

              acc_C00 = mac_8x8_8x8T_conf(A0, 1, B0, 1, acc_C00, 0, 0, 0);
              acc_C01 = mac_8x8_8x8T_conf(A0, 1, B1, 1, acc_C01, 0, 0, 0);
              acc_C10 = mac_8x8_8x8T_conf(A1, 1, B0, 1, acc_C10, 0, 0, 0);
              acc_C11 = mac_8x8_8x8T_conf(A1, 1, B1, 1, acc_C11, 0, 0, 0);
            }

          C00 = acc_C00.template to_vector<bfp16ebs8>();
          pC1bfp16_out.push(C00);
          C01 = acc_C01.template to_vector<bfp16ebs8>();
          pC1bfp16_out.push(C01);

          C10 = acc_C10.template to_vector<bfp16ebs8>();
          pC2bfp16_out.push(C10);
          C11 = acc_C11.template to_vector<bfp16ebs8>();
          pC2bfp16_out.push(C11);
        }
    }
}

void matmul_testing_kernel(bfloat16 *__restrict pA, bfloat16 *__restrict pB,
                           bfloat16 *__restrict pC) {
  matmul_vectorized_2x2_bfp16_bfp16<32, 32, 32, 8, 8, 8>((int8 *)pA, (int8 *)pB,
                                                         (int8 *)pC);
}
