#include "mkl.h"
#include "mkl_solvers_ee.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include "tqdm.hh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

int main() {

  int i, j;

  std::vector<MKL_INT> A_row_index = {0, 0, 1, 1, 2};
  std::vector<MKL_INT> A_col_index = {0, 1, 1, 2, 2};
  std::vector<double> A_values = {2, -1, 2, -1, 2};

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_matrix_t Ai;
  sparse_matrix_t AiCOO;
  mkl_sparse_d_create_coo(&AiCOO, indexing, 3, 3, 5, A_row_index.data(),
                          A_col_index.data(), A_values.data());
  mkl_sparse_convert_csr(AiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Ai);

  std::vector<MKL_INT> x_row_index = {0, 1, 2};
  std::vector<MKL_INT> x_col_index = {0, 0, 0};
  std::vector<double> x_values = {2, -1, 2};

  sparse_matrix_t xi;
  sparse_matrix_t xiCOO;
  mkl_sparse_d_create_coo(&xiCOO, indexing, 3, 1, 3, x_row_index.data(),
                          x_col_index.data(), x_values.data());
  mkl_sparse_convert_csr(xiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &xi);

  std::vector<MKL_INT> y_row_index = {0, 0, 0};
  std::vector<MKL_INT> y_col_index = {0, 1, 2};
  std::vector<double> y_values = {2, -1, 2};

  sparse_matrix_t yi;
  sparse_matrix_t yiCOO;
  mkl_sparse_d_create_coo(&yiCOO, indexing, 1, 3, 3, y_row_index.data(),
                          y_col_index.data(), y_values.data());
  mkl_sparse_convert_csr(yiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &yi);

  sparse_matrix_t Ay;
  sparse_matrix_t xAy;
  mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, Ai, xi, &Ay);
  mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, yi, Ay, &xAy);

  int rows, cols, *rows_start, *rows_end, *col_index;
  double *val;

  mkl_sparse_d_export_csr(xAy, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);

  std::cout << "rows: " << rows << " cols: " << cols << std::endl;
  std::cout << val[0] << " " << val[1] << " " << val[2] << std::endl;
  return 0;
}
