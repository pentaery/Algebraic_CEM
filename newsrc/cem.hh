#pragma once

#include "mkl_spblas.h"
#include "mkl_types.h"

#include "mfem.hpp"
#include "mkl.h"
#include "mkl_cblas.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

class CEM {
public:
  CEM();
  ~CEM();

  void getDatafromMFEM(const char *mesh_file, int order = 1);
  double solveFromLAndReleaseA();

  MKL_INT numVertices() const { return nvtxs; }

private:
  void destroyIfAllocated(sparse_matrix_t &matrix);

  sparse_matrix_t matA;
  sparse_matrix_t matL;

  std::vector<double> vecRHS;
  std::vector<double> vecSOL;

  // Backing storage for MKL CSR handle for Laplacian matrix L from MFEM data.
  std::vector<MKL_INT> matL_rows_start;
  std::vector<MKL_INT> matL_rows_end;
  std::vector<MKL_INT> matL_col_index;
  std::vector<double> matL_values;

  sparse_index_base_t indexing;
  MKL_INT rows;
  MKL_INT cols;
  MKL_INT nvtxs;
};
