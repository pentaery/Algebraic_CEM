#pragma once

#include "mkl_spblas.h"
#include "mkl_types.h"

#include "metis.h"
#include "mfem.hpp"
#include "mkl.h"
#include "mkl_cblas.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class CEM {
public:
  CEM();
  explicit CEM(int k0_);
  CEM(idx_t nparts_, int overlap_, int k0_);
  ~CEM();

  void getDatafromMFEM(const char *mesh_file, int order = 1);
  void graphPartition();
  void findNeighbours();
  void formAUX();
  void exportNeighboursData() const;
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

  idx_t nparts;
  int overlap;
  int k0;
  double cStar;
  std::vector<idx_t> part;

  std::vector<std::set<idx_t>> vertices;
  // 每个子域的点集
  std::vector<std::set<idx_t>> neighbours;
  // 每个子域的邻居子域集合
  std::vector<std::set<idx_t>> overlapping;
  // 每个子域的重叠子域集合

  std::vector<idx_t> globalTolocal;
  // 每个全局点对应的局部点编号
  std::vector<idx_t> count;
  // 每个子域的局部点数量

  std::vector<std::unordered_set<idx_t>> verticesCEM;
  // 每个子域的点集（重叠后）
  std::vector<std::unordered_map<idx_t, idx_t>> globalTolocalCEM;
  // 每个子域的全局点编号到局部点编号的映射
  std::vector<std::vector<idx_t>> localtoGlobalCEM;
  // 每个子域的局部点编号到全局点编号的映射

  std::vector<std::vector<double>> eigenvector;
  std::vector<std::vector<double>> eigenvalue;
};
