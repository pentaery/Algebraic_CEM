#include "cem.hh"

namespace {
double f_rhs(const mfem::Vector &x) {
  const double pi = M_PI;
  return 2.0 * pi * pi * std::sin(pi * x(0)) * std::sin(pi * x(1));
}
} // namespace

CEM::CEM()
    : matA(nullptr), matL(nullptr), indexing(SPARSE_INDEX_BASE_ZERO), rows(0),
      cols(0), nvtxs(0) {}

CEM::~CEM() {
  destroyIfAllocated(matA);
  destroyIfAllocated(matL);
}

void CEM::destroyIfAllocated(sparse_matrix_t &matrix) {
  if (matrix != nullptr) {
    mkl_sparse_destroy(matrix);
    matrix = nullptr;
  }
}

void CEM::getDatafromMFEM(const char *mesh_file, int order) {
  std::ifstream imesh(mesh_file);
  if (!imesh) {
    throw std::runtime_error(std::string("Can not open mesh file: ") +
                             mesh_file);
  }

  mfem::Mesh mesh(imesh, 1, 1);
  const int dim = mesh.Dimension();

  mfem::H1_FECollection fec(order, dim);
  mfem::FiniteElementSpace fespace(&mesh, &fec);

  mfem::Array<int> ess_tdof_list;
  if (mesh.bdr_attributes.Size()) {
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  mfem::LinearForm b(&fespace);
  mfem::FunctionCoefficient f_coef(f_rhs);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(f_coef));
  b.Assemble();

  mfem::GridFunction x(&fespace);
  x = 0.0;

  mfem::BilinearForm a(&fespace);
  mfem::ConstantCoefficient one(1.0);
  a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
  a.Assemble();

  mfem::SparseMatrix A;
  mfem::Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  A.SortColumnIndices();
  nvtxs = static_cast<MKL_INT>(A.Height());
  rows = nvtxs;
  cols = nvtxs;

  vecRHS.assign(B.Size(), 0.0);
  for (int i = 0; i < B.Size(); ++i) {
    vecRHS[i] = B(i);
  }
  vecSOL.assign(nvtxs, 0.0);

  const int *I = A.GetI();
  const int *J = A.GetJ();
  const double *Data = A.GetData();

  const int nnz = A.NumNonZeroElems();
  for (int i = 0; i < std::min(100, nvtxs + 1); ++i) {
    std::cout << I[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < std::min(100, nnz); ++i) {
    std::cout << J[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < std::min(100, nnz); ++i) {
    std::cout << Data[i] << " ";
  }
  std::cout << std::endl;

  // Build and store L in CSR directly from MFEM's A-CSR.
  const auto max_mkl_int = std::numeric_limits<MKL_INT>::max();
  std::vector<char> has_diag(static_cast<size_t>(nvtxs), 0);
  std::vector<double> row_sum(static_cast<size_t>(nvtxs), 0.0);

  for (MKL_INT row = 0; row < nvtxs; ++row) {
    const int row_i = static_cast<int>(row);
    if (I[row_i] < 0 || I[row_i + 1] < 0) {
      throw std::runtime_error("Negative CSR row pointer from MFEM.");
    }
    if (static_cast<long long>(I[row_i]) > max_mkl_int ||
        static_cast<long long>(I[row_i + 1]) > max_mkl_int) {
      throw std::runtime_error("CSR row pointer exceeds MKL_INT range.");
    }

    for (int idx = I[row_i]; idx < I[row_i + 1]; ++idx) {
      if (J[idx] < 0) {
        throw std::runtime_error("Negative CSR column index from MFEM.");
      }
      if (static_cast<long long>(J[idx]) > max_mkl_int) {
        throw std::runtime_error("CSR column index exceeds MKL_INT range.");
      }
      row_sum[static_cast<size_t>(row)] += Data[idx];
      if (J[idx] == row_i) {
        has_diag[static_cast<size_t>(row)] = 1;
      }
    }
  }

  MKL_INT added_diag = 0;
  for (MKL_INT row = 0; row < nvtxs; ++row) {
    if (!has_diag[static_cast<size_t>(row)]) {
      ++added_diag;
    }
  }

  const MKL_INT l_nnz = static_cast<MKL_INT>(nnz) + added_diag;
  matL_rows_start.assign(static_cast<size_t>(nvtxs), 0);
  matL_rows_end.assign(static_cast<size_t>(nvtxs), 0);
  matL_col_index.assign(static_cast<size_t>(l_nnz), 0);
  matL_values.assign(static_cast<size_t>(l_nnz), 0.0);

  MKL_INT out = 0;
  for (MKL_INT row = 0; row < nvtxs; ++row) {
    const int row_i = static_cast<int>(row);
    matL_rows_start[static_cast<size_t>(row)] = out;

    const bool row_has_diag = has_diag[static_cast<size_t>(row)] != 0;
    bool inserted_diag = false;
    const double l_diag = row_sum[static_cast<size_t>(row)];

    for (int idx = I[row_i]; idx < I[row_i + 1]; ++idx) {
      const MKL_INT col = static_cast<MKL_INT>(J[idx]);

      if (!row_has_diag && !inserted_diag && col > row) {
        matL_col_index[static_cast<size_t>(out)] = row;
        matL_values[static_cast<size_t>(out)] = l_diag;
        ++out;
        inserted_diag = true;
      }

      matL_col_index[static_cast<size_t>(out)] = col;
      if (col == row) {
        matL_values[static_cast<size_t>(out)] = l_diag;
        inserted_diag = true;
      } else {
        matL_values[static_cast<size_t>(out)] = -Data[idx];
      }
      ++out;
    }

    if (!inserted_diag) {
      matL_col_index[static_cast<size_t>(out)] = row;
      matL_values[static_cast<size_t>(out)] = l_diag;
      ++out;
    }

    matL_rows_end[static_cast<size_t>(row)] = out;
  }

  if (out != l_nnz) {
    throw std::runtime_error("Internal error while building L CSR.");
  }

  // getDatafromMFEM now stores only L; A is assembled during
  // solveFromLAndReleaseA().
  destroyIfAllocated(matA);
  destroyIfAllocated(matL);
  const sparse_status_t create_status = mkl_sparse_d_create_csr(
      &matL, indexing, nvtxs, nvtxs, matL_rows_start.data(),
      matL_rows_end.data(), matL_col_index.data(), matL_values.data());

  for (int i = 0; i < 100; ++i) {
    std::cout << matL_rows_start[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << matL_col_index[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << matL_values[i] << " ";
  }
  std::cout << std::endl;
  if (create_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("mkl_sparse_d_create_csr failed with status " +
                             std::to_string(static_cast<int>(create_status)));
  }

  std::cout << "MFEM assembled reduced system with nvtxs=" << nvtxs
            << " and nnz=" << nnz << std::endl;
}

double CEM::solveFromLAndReleaseA() {
  auto start = std::chrono::high_resolution_clock::now();

  if (matL == nullptr) {
    throw std::runtime_error(
        "matL is not initialized. Call getDatafromMFEM() first.");
  }
  if (nvtxs <= 0) {
    throw std::runtime_error("Invalid system size for PARDISO solve.");
  }

  destroyIfAllocated(matA);

  try {
    MKL_INT *l_rows_start = nullptr;
    MKL_INT *l_rows_end = nullptr;
    MKL_INT *l_col_index = nullptr;
    double *l_val = nullptr;
    const sparse_status_t export_l_status =
        mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &l_rows_start,
                                &l_rows_end, &l_col_index, &l_val);
    if (export_l_status != SPARSE_STATUS_SUCCESS) {
      throw std::runtime_error(
          "mkl_sparse_d_export_csr(matL) failed with status " +
          std::to_string(static_cast<int>(export_l_status)));
    }

    // Calculate NNZ for matA and fill row pointers.
    // Each row of matA will have 1 diagonal element + all upper triangular
    // elements from L.
    MKL_INT nnz_A = 0;
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      nnz_A++; // The diagonal entry (sum of row L)
      for (MKL_INT j = l_rows_start[i]; j < l_rows_end[i]; ++j) {
        if (l_col_index[j] > i) {
          nnz_A++; // Upper triangular off-diagonal entry
        }
      }
    }

    std::vector<MKL_INT> matA_rows_start(static_cast<size_t>(nvtxs), 0);
    std::vector<MKL_INT> matA_rows_end(static_cast<size_t>(nvtxs), 0);
    std::vector<MKL_INT> matA_col_index(static_cast<size_t>(nnz_A), 0);
    std::vector<double> matA_values(static_cast<size_t>(nnz_A), 0.0);

    auto releaseMatACSR = [&]() {
      std::vector<MKL_INT>().swap(matA_rows_start);
      std::vector<MKL_INT>().swap(matA_rows_end);
      std::vector<MKL_INT>().swap(matA_col_index);
      std::vector<double>().swap(matA_values);
    };

    MKL_INT current_idx = 0;
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      matA_rows_start[i] = current_idx;

      // Reserve space for diagonal entry at the beginning of the row
      const MKL_INT diag_loc = current_idx;
      matA_col_index[current_idx] = i;
      matA_values[current_idx] = 0.0;
      current_idx++;

      for (MKL_INT j = l_rows_start[i]; j < l_rows_end[i]; ++j) {
        // Accumulate to diagonal sum: A(i,i) += L(i,j)
        matA_values[diag_loc] += l_val[j];

        // Add off-diagonal entry: A(i,j) = -L(i,j) for j > i
        if (l_col_index[j] > i) {
          matA_col_index[current_idx] = l_col_index[j];
          matA_values[current_idx] = -l_val[j];
          current_idx++;
        }
      }
      matA_rows_end[i] = current_idx;
    }

    if (current_idx != nnz_A) {
      throw std::runtime_error("Internal error constructing matA CSR.");
    }

    MKL_INT *rows_start = matA_rows_start.data();
    MKL_INT *rows_end = matA_rows_end.data();
    MKL_INT *col_index = matA_col_index.data();
    double *val = matA_values.data();

    MKL_INT perm[64], iparm[64];
    void *pt[64];
    MKL_INT error;
    MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
    MKL_INT nrhs = 1, msglv1 = 0;
    for (int i = 0; i < 64; ++i) {
      pt[i] = 0;
      iparm[i] = 0;
      perm[i] = 0;
    }
    iparm[34] = 1;
    iparm[0] = 1;
    iparm[1] = 3;

    if (vecSOL.size() != static_cast<size_t>(nvtxs)) {
      vecSOL.assign(nvtxs, 0.0);
    }

    std::vector<MKL_INT> ia(static_cast<size_t>(nvtxs) + 1, 0);
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      ia[static_cast<size_t>(i)] = rows_start[i];
    }
    ia[static_cast<size_t>(nvtxs)] = rows_end[nvtxs - 1];

    if (ia.front() != 0) {
      throw std::runtime_error(
          "Invalid CSR ia[0]; expected 0 for 0-based indexing.");
    }
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      if (ia[static_cast<size_t>(i)] > ia[static_cast<size_t>(i + 1)]) {
        throw std::runtime_error(
            "Invalid CSR row pointer: ia is not monotonic.");
      }
    }

    // Build upper-triangular CSR arrays for mtype=2.
    std::vector<MKL_INT> ia_upper(static_cast<size_t>(nvtxs) + 1, 0);
    std::vector<MKL_INT> ja_upper;
    std::vector<double> a_upper;

    ja_upper.reserve(static_cast<size_t>(rows_end[nvtxs - 1]));
    a_upper.reserve(static_cast<size_t>(rows_end[nvtxs - 1]));

    for (MKL_INT i = 0; i < nvtxs; ++i) {
      ia_upper[static_cast<size_t>(i)] = static_cast<MKL_INT>(ja_upper.size());
      for (MKL_INT idx = rows_start[i]; idx < rows_end[i]; ++idx) {
        if (col_index[idx] >= i) { // keep upper triangle including diagonal
          ja_upper.push_back(col_index[idx]);
          a_upper.push_back(val[idx]);
        }
      }
    }
    ia_upper[static_cast<size_t>(nvtxs)] =
        static_cast<MKL_INT>(ja_upper.size());

    if (ia_upper.front() != 0) {
      throw std::runtime_error(
          "Invalid CSR ia[0]; expected 0 for 0-based indexing.");
    }
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      if (ia_upper[static_cast<size_t>(i)] >
          ia_upper[static_cast<size_t>(i + 1)]) {
        throw std::runtime_error(
            "Invalid CSR row pointer: ia is not monotonic.");
      }
    }

    for (int i = 0; i < 100; ++i) {
      std::cout << ia_upper[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 100; ++i) {
      std::cout << ja_upper[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 100; ++i) {
      std::cout << a_upper[i] << " ";
    }
    std::cout << std::endl;

    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, a_upper.data(),
            ia_upper.data(), ja_upper.data(), perm, &nrhs, iparm, &msglv1,
            vecRHS.data(), vecSOL.data(), &error);

    std::cout << "PARDISO phase " << phase << " completed with error code "
              << error << std::endl;
    if (error != 0) {
      std::cout << "PARDISO error: " << error << std::endl;
    }

    phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, a_upper.data(),
            ia_upper.data(), ja_upper.data(), perm, &nrhs, iparm, &msglv1,
            vecRHS.data(), vecSOL.data(), &error);

    std::vector<double> Ax(static_cast<size_t>(nvtxs), 0.0);
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      for (MKL_INT j = rows_start[i]; j < rows_end[i]; ++j) {
        Ax[static_cast<size_t>(i)] +=
            val[j] * vecSOL[static_cast<size_t>(col_index[j])];
      }
    }
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      Ax[static_cast<size_t>(i)] -= vecRHS[static_cast<size_t>(i)];
    }

    const int inc = 1;
    const double rhs_norm = cblas_dnrm2(nvtxs, vecRHS.data(), inc);
    const double res_norm = cblas_dnrm2(nvtxs, Ax.data(), inc);
    const double rel_res = (rhs_norm == 0.0) ? -1.0 : (res_norm / rhs_norm);

    releaseMatACSR();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Direct solve completed in " << duration.count() << " ms"
              << std::endl;

    return rel_res;
  } catch (...) {
    throw;
  }
}
