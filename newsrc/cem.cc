#include "cem.hh"

namespace {
double f_rhs(const mfem::Vector &x) {
  const double pi = M_PI;
  return 2.0 * pi * pi * std::sin(pi * x(0)) * std::sin(pi * x(1));
}
} // namespace

CEM::CEM()
    : matA(nullptr), matL(nullptr), indexing(SPARSE_INDEX_BASE_ZERO), rows(0),
      cols(0), nvtxs(0), nparts(10), overlap(2), k0(3), cStar(1.0) {}

CEM::CEM(int nparts_, int overlap_, int k0_)
    : matA(nullptr), matL(nullptr), indexing(SPARSE_INDEX_BASE_ZERO), rows(0),
      cols(0), nvtxs(0), nparts(nparts_), overlap(overlap_), k0(k0_),
      cStar(1.0) {
  if (nparts <= 0) {
    throw std::runtime_error("nparts must be positive.");
  }
  if (overlap < 0) {
    throw std::runtime_error("overlap must be non-negative.");
  }
  if (k0 <= 0) {
    throw std::runtime_error("k0 must be positive.");
  }
}

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
  cStar = static_cast<double>(nvtxs);
  part.assign(static_cast<size_t>(nvtxs), 0);

  vecRHS.assign(B.Size(), 0.0);
  for (int i = 0; i < B.Size(); ++i) {
    vecRHS[i] = B(i);
  }
  vecSOL.assign(nvtxs, 0.0);

  const int *I = A.GetI();
  const int *J = A.GetJ();
  const double *Data = A.GetData();

  const int nnz = A.NumNonZeroElems();

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

  if (create_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("mkl_sparse_d_create_csr failed with status " +
                             std::to_string(static_cast<int>(create_status)));
  }

  std::cout << "MFEM assembled reduced system with nvtxs=" << nvtxs
            << " and nnz=" << nnz << std::endl;
}

void CEM::graphPartition() {
  if (matL == nullptr) {
    throw std::runtime_error(
        "matL is not initialized. Call getDatafromMFEM() first.");
  }
  if (nvtxs <= 0) {
    throw std::runtime_error("Invalid nvtxs for graph partitioning.");
  }
  if (nparts <= 0) {
    throw std::runtime_error("nparts must be positive for graph partitioning.");
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase I: Graph Partitioning======" << std::endl;

  MKL_INT *rows_start = nullptr;
  MKL_INT *rows_end = nullptr;
  MKL_INT *col_index = nullptr;
  double *val = nullptr;

  const sparse_status_t export_status = mkl_sparse_d_export_csr(
      matL, &indexing, &rows, &cols, &rows_start, &rows_end, &col_index, &val);
  if (export_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        "mkl_sparse_d_export_csr(matL) failed with status " +
        std::to_string(static_cast<int>(export_status)));
  }

  std::vector<idx_t> xadj(static_cast<size_t>(nvtxs) + 1, 0);
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    if (rows_start[i] < 0) {
      throw std::runtime_error(
          "Invalid CSR row pointer (negative rows_start).");
    }
    xadj[static_cast<size_t>(i)] = static_cast<idx_t>(rows_start[i]);
  }
  xadj[static_cast<size_t>(nvtxs)] = static_cast<idx_t>(rows_end[nvtxs - 1]);

  const MKL_INT nnz = rows_end[nvtxs - 1];
  std::vector<idx_t> adjncy(static_cast<size_t>(nnz), 0);
  for (MKL_INT i = 0; i < nnz; ++i) {
    if (col_index[i] < 0) {
      throw std::runtime_error(
          "Invalid CSR column index (negative col_index).");
    }
    adjncy[static_cast<size_t>(i)] = static_cast<idx_t>(col_index[i]);
  }

  idx_t nVertices = static_cast<idx_t>(nvtxs);
  idx_t ncon = 1;
  idx_t objval = 0;
  std::vector<idx_t> part_local(static_cast<size_t>(nvtxs), 0);

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;

  const int metis_status = METIS_PartGraphKway(
      &nVertices, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,
      &nparts, nullptr, nullptr, options, &objval, part_local.data());

  if (metis_status != METIS_OK) {
    throw std::runtime_error("METIS_PartGraphKway failed with code " +
                             std::to_string(metis_status));
  }

  part = std::move(part_local);

  std::ofstream outfile("../../partition.txt");
  if (!outfile.is_open()) {
    throw std::runtime_error("Cannot open ../../partition.txt for writing.");
  }
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    outfile << part[static_cast<size_t>(i)] << ' ';
  }
  outfile << '\n';
  outfile.close();

  std::cout << "Objective for the partition is " << objval << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished with " << duration.count()
            << " ms======" << std::endl;
}

void CEM::findNeighbours() {
  if (matL == nullptr) {
    throw std::runtime_error(
        "matL is not initialized. Call getDatafromMFEM() first.");
  }
  if (nvtxs <= 0) {
    throw std::runtime_error("Invalid nvtxs for neighbour construction.");
  }
  if (nparts <= 0) {
    throw std::runtime_error("nparts must be positive.");
  }
  if (part.size() != static_cast<size_t>(nvtxs)) {
    throw std::runtime_error(
        "Partition vector size mismatch. Call graphPartition() first.");
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::cout
      << "======Phase II: Construct the neighours for the CEM method======"
      << std::endl;

  vertices.assign(static_cast<size_t>(nparts), std::set<idx_t>{});
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    const idx_t p = part[static_cast<size_t>(i)];
    if (p < 0 || p >= nparts) {
      throw std::runtime_error("Invalid partition id in part[].");
    }
    vertices[static_cast<size_t>(p)].insert(static_cast<idx_t>(i));
  }

  neighbours.assign(static_cast<size_t>(nparts), std::set<idx_t>{});

  MKL_INT *rows_start = nullptr;
  MKL_INT *rows_end = nullptr;
  MKL_INT *col_index = nullptr;
  double *val = nullptr;
  const sparse_status_t export_status = mkl_sparse_d_export_csr(
      matL, &indexing, &rows, &cols, &rows_start, &rows_end, &col_index, &val);
  if (export_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        "mkl_sparse_d_export_csr(matL) failed with status " +
        std::to_string(static_cast<int>(export_status)));
  }

  for (MKL_INT i = 0; i < nvtxs; ++i) {
    for (MKL_INT j = rows_start[i]; j < rows_end[i]; ++j) {
      const MKL_INT col = col_index[j];
      if (col < 0 || col >= nvtxs) {
        throw std::runtime_error("Invalid column index in matL CSR.");
      }
      const idx_t pi = part[static_cast<size_t>(i)];
      const idx_t pj = part[static_cast<size_t>(col)];
      if (pi != pj) {
        neighbours[static_cast<size_t>(pi)].insert(pj);
      }
    }
  }

  overlapping.assign(static_cast<size_t>(nparts), std::set<idx_t>{});
  if (overlap > 0) {
    for (idx_t j = 0; j < nparts; ++j) {
      auto &overlap_j = overlapping[static_cast<size_t>(j)];
      const auto &nbr_j = neighbours[static_cast<size_t>(j)];
      overlap_j.insert(nbr_j.begin(), nbr_j.end());
      overlap_j.insert(j);
    }
  }

  for (int i = 1; i < overlap; ++i) {
    for (idx_t j = 0; j < nparts; ++j) {
      std::set<idx_t> temp;
      for (const auto &element : overlapping[static_cast<size_t>(j)]) {
        const auto &nbr_e = neighbours[static_cast<size_t>(element)];
        temp.insert(nbr_e.begin(), nbr_e.end());
      }
      auto &overlap_j = overlapping[static_cast<size_t>(j)];
      overlap_j.insert(temp.begin(), temp.end());
    }
  }

  globalTolocal.assign(static_cast<size_t>(nvtxs), 0);
  count.assign(static_cast<size_t>(nparts), 0);
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    const idx_t p = part[static_cast<size_t>(i)];
    globalTolocal[static_cast<size_t>(i)] = count[static_cast<size_t>(p)];
    ++count[static_cast<size_t>(p)];
  }

  verticesCEM.assign(static_cast<size_t>(nparts), std::unordered_set<idx_t>{});
  globalTolocalCEM.assign(static_cast<size_t>(nparts),
                          std::unordered_map<idx_t, idx_t>{});
  for (idx_t i = 0; i < nparts; ++i) {
    idx_t local_idx = 0;
    for (const auto &element : overlapping[static_cast<size_t>(i)]) {
      const auto &vset = vertices[static_cast<size_t>(element)];
      verticesCEM[static_cast<size_t>(i)].insert(vset.begin(), vset.end());
      for (const auto &element2 : vset) {
        globalTolocalCEM[static_cast<size_t>(i)].insert(
            {element2, local_idx++});
      }
    }
  }

  localtoGlobalCEM.assign(static_cast<size_t>(nparts), std::vector<idx_t>{});
  for (idx_t i = 0; i < nparts; ++i) {
    idx_t index = 0;
    auto &local_to_global = localtoGlobalCEM[static_cast<size_t>(i)];
    local_to_global.resize(verticesCEM[static_cast<size_t>(i)].size());
    for (const auto &element : overlapping[static_cast<size_t>(i)]) {
      for (const auto &element2 : vertices[static_cast<size_t>(element)]) {
        if (index >= static_cast<idx_t>(local_to_global.size())) {
          throw std::runtime_error(
              "Inconsistent local-to-global size while forming overlap map.");
        }
        local_to_global[static_cast<size_t>(index++)] = element2;
      }
    }
    if (index != static_cast<idx_t>(local_to_global.size())) {
      throw std::runtime_error(
          "Incomplete local-to-global mapping while forming overlap map.");
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished with " << duration.count()
            << " ms======" << std::endl;
}

void CEM::formAUX() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase III: Construct the Auxiliary space======"
            << std::endl;

  if (matL == nullptr) {
    throw std::runtime_error(
        "matL is not initialized. Call getDatafromMFEM() first.");
  }
  if (k0 <= 0) {
    throw std::runtime_error("k0 must be positive before formAUX().");
  }
  if (cStar <= 0.0) {
    throw std::runtime_error("cStar must be positive before formAUX().");
  }
  if (part.size() != static_cast<size_t>(nvtxs)) {
    throw std::runtime_error(
        "Partition vector size mismatch. Call graphPartition() first.");
  }
  if (count.size() != static_cast<size_t>(nparts)) {
    throw std::runtime_error(
        "Partition local counts are missing. Call findNeighbours() first.");
  }

  MKL_INT *rows_start = nullptr;
  MKL_INT *rows_end = nullptr;
  MKL_INT *col_index = nullptr;
  double *val = nullptr;
  const sparse_status_t export_status = mkl_sparse_d_export_csr(
      matL, &indexing, &rows, &cols, &rows_start, &rows_end, &col_index, &val);
  if (export_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        "mkl_sparse_d_export_csr(matL) failed with status " +
        std::to_string(static_cast<int>(export_status)));
  }

  char which = 'S';
  MKL_INT pm[128];
  mkl_sparse_ee_init(pm);
  pm[7] = 1;
  pm[8] = 1;

  std::vector<std::vector<MKL_INT>> Ai_col_index(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Ai_row_index(static_cast<size_t>(nparts));
  std::vector<std::vector<double>> Ai_values(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Si_col_index(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Si_row_index(static_cast<size_t>(nparts));
  std::vector<std::vector<double>> Si_values(static_cast<size_t>(nparts));

  for (MKL_INT i = 0; i < nvtxs; ++i) {
    for (MKL_INT j = rows_start[i]; j < rows_end[i]; ++j) {
      const MKL_INT col = col_index[j];
      if (col < 0 || col >= nvtxs) {
        throw std::runtime_error("Invalid column index in matL CSR.");
      }

      const idx_t pi = part[static_cast<size_t>(i)];
      const idx_t pj = part[static_cast<size_t>(col)];
      if (pi < 0 || pi >= nparts || pj < 0 || pj >= nparts) {
        throw std::runtime_error("Invalid partition id while building AUX.");
      }

      if (pi == pj) {
        const size_t part_id = static_cast<size_t>(pi);
        const MKL_INT local_i = globalTolocal[static_cast<size_t>(i)];
        if (col != i) {
          const MKL_INT local_col = globalTolocal[static_cast<size_t>(col)];

          Ai_row_index[part_id].push_back(local_i);
          Ai_col_index[part_id].push_back(local_i);
          Ai_values[part_id].push_back(val[j]);

          Ai_row_index[part_id].push_back(local_i);
          Ai_col_index[part_id].push_back(local_col);
          Ai_values[part_id].push_back(-val[j]);

          Si_col_index[part_id].push_back(local_i);
          Si_row_index[part_id].push_back(local_i);
          Si_values[part_id].push_back(val[j] / (cStar * cStar * 2.0));
        } else {
          Ai_row_index[part_id].push_back(local_i);
          Ai_col_index[part_id].push_back(local_i);
          Ai_values[part_id].push_back(val[j]);

          Si_col_index[part_id].push_back(local_i);
          Si_row_index[part_id].push_back(local_i);
          Si_values[part_id].push_back(val[j] / (cStar * cStar));
        }
      }
    }
  }

  eigenvalue.resize(static_cast<size_t>(nparts));
  eigenvector.resize(static_cast<size_t>(nparts));
  for (idx_t i = 0; i < nparts; ++i) {
    const size_t part_id = static_cast<size_t>(i);
    const MKL_INT count_i = count[part_id];
    if (count_i < 0) {
      throw std::runtime_error("Negative local vertex count in formAUX().");
    }
    eigenvalue[part_id].assign(static_cast<size_t>(k0), 0.0);
    eigenvector[part_id].assign(static_cast<size_t>(k0 * count_i), 0.0);
  }

  std::vector<double> res(static_cast<size_t>(nparts), 0.0);

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  descr.mode = SPARSE_FILL_MODE_UPPER;

  for (idx_t i = 0; i < nparts; ++i) {
    const size_t part_id = static_cast<size_t>(i);
    const MKL_INT count_i = count[part_id];
    if (count_i <= 0) {
      std::cout << "part: " << i << " skipped in formAUX because count is zero."
                << std::endl;
      continue;
    }

    sparse_matrix_t AiCOO = nullptr;
    sparse_matrix_t SiCOO = nullptr;
    sparse_matrix_t Ai = nullptr;
    sparse_matrix_t Si = nullptr;

    const sparse_status_t ai_coo_status = mkl_sparse_d_create_coo(
        &AiCOO, indexing, count_i, count_i,
        static_cast<MKL_INT>(Ai_values[part_id].size()),
        Ai_row_index[part_id].data(), Ai_col_index[part_id].data(),
        Ai_values[part_id].data());
    if (ai_coo_status != SPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("mkl_sparse_d_create_coo(Ai) failed.");
    }

    const sparse_status_t si_coo_status = mkl_sparse_d_create_coo(
        &SiCOO, indexing, count_i, count_i,
        static_cast<MKL_INT>(Si_values[part_id].size()),
        Si_row_index[part_id].data(), Si_col_index[part_id].data(),
        Si_values[part_id].data());
    if (si_coo_status != SPARSE_STATUS_SUCCESS) {
      mkl_sparse_destroy(AiCOO);
      throw std::runtime_error("mkl_sparse_d_create_coo(Si) failed.");
    }

    const sparse_status_t ai_csr_status =
        mkl_sparse_convert_csr(AiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Ai);
    const sparse_status_t si_csr_status =
        mkl_sparse_convert_csr(SiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Si);
    mkl_sparse_destroy(AiCOO);
    mkl_sparse_destroy(SiCOO);
    if (ai_csr_status != SPARSE_STATUS_SUCCESS ||
        si_csr_status != SPARSE_STATUS_SUCCESS) {
      if (Ai != nullptr) {
        mkl_sparse_destroy(Ai);
      }
      if (Si != nullptr) {
        mkl_sparse_destroy(Si);
      }
      throw std::runtime_error(
          "mkl_sparse_convert_csr failed for AUX local matrices.");
    }

    int k = 0;
    const sparse_status_t gv_status = mkl_sparse_d_gv(
        &which, pm, Ai, descr, Si, descr, k0, &k, eigenvalue[part_id].data(),
        eigenvector[part_id].data(), &res[part_id]);

    mkl_sparse_destroy(Ai);
    mkl_sparse_destroy(Si);

    if (gv_status != SPARSE_STATUS_SUCCESS) {
      std::cout << "======error in mkl_sparse_d_gv: "
                << static_cast<int>(gv_status) << "======" << std::endl;
    }
    if (k < k0) {
      std::cout << "===========Not enough eigenvalues in part " << i
                << "===========" << std::endl;
    }
    std::cout << "part: " << i << " residual: " << res[part_id]
              << " Smallest eigenvalue: " << eigenvalue[part_id][0]
              << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finish solving eigen problem in each coarse element with "
            << duration.count() << " ms======" << std::endl;
}

void CEM::exportNeighboursData() const {
  if (nvtxs <= 0 || nparts <= 0) {
    throw std::runtime_error(
        "Invalid nvtxs/nparts for neighbours data export.");
  }
  if (part.size() != static_cast<size_t>(nvtxs)) {
    throw std::runtime_error(
        "Partition vector size mismatch in neighbours data export.");
  }
  if (vertices.size() != static_cast<size_t>(nparts) ||
      overlapping.size() != static_cast<size_t>(nparts) ||
      verticesCEM.size() != static_cast<size_t>(nparts)) {
    throw std::runtime_error(
        "findNeighbours data is incomplete. Call findNeighbours() first.");
  }

  const std::string base_dir = "../../data";

  {
    std::ofstream meta(base_dir + "/neighbours_meta.txt");
    if (!meta.is_open()) {
      throw std::runtime_error("Cannot open neighbours_meta.txt for writing.");
    }
    meta << "nvtxs " << nvtxs << '\n';
    meta << "nparts " << nparts << '\n';
    meta << "overlap " << overlap << '\n';
  }

  {
    std::ofstream part_file(base_dir + "/neighbours_part.txt");
    if (!part_file.is_open()) {
      throw std::runtime_error("Cannot open neighbours_part.txt for writing.");
    }
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      part_file << part[static_cast<size_t>(i)] << ' ';
    }
    part_file << '\n';
  }

  for (idx_t i = 0; i < nparts; ++i) {
    const size_t part_idx = static_cast<size_t>(i);

    {
      std::ofstream core_file(base_dir + "/neighbours_vertices_" +
                              std::to_string(i) + ".txt");
      if (!core_file.is_open()) {
        throw std::runtime_error("Cannot open neighbours_vertices file.");
      }
      for (const auto &v : vertices[part_idx]) {
        core_file << v << ' ';
      }
      core_file << '\n';
    }

    {
      std::ofstream overlap_parts_file(base_dir +
                                       "/neighbours_overlapping_parts_" +
                                       std::to_string(i) + ".txt");
      if (!overlap_parts_file.is_open()) {
        throw std::runtime_error(
            "Cannot open neighbours_overlapping_parts file.");
      }
      for (const auto &p : overlapping[part_idx]) {
        overlap_parts_file << p << ' ';
      }
      overlap_parts_file << '\n';
    }

    {
      std::vector<idx_t> overlap_vertices(verticesCEM[part_idx].begin(),
                                          verticesCEM[part_idx].end());
      std::sort(overlap_vertices.begin(), overlap_vertices.end());

      std::ofstream overlap_vertices_file(
          base_dir + "/neighbours_verticesCEM_" + std::to_string(i) + ".txt");
      if (!overlap_vertices_file.is_open()) {
        throw std::runtime_error("Cannot open neighbours_verticesCEM file.");
      }
      for (const auto &v : overlap_vertices) {
        overlap_vertices_file << v << ' ';
      }
      overlap_vertices_file << '\n';
    }

    {
      std::ofstream local_to_global_file(base_dir +
                                         "/neighbours_localtoGlobalCEM_" +
                                         std::to_string(i) + ".txt");
      if (!local_to_global_file.is_open()) {
        throw std::runtime_error(
            "Cannot open neighbours_localtoGlobalCEM file.");
      }
      for (const auto &v : localtoGlobalCEM[part_idx]) {
        local_to_global_file << v << ' ';
      }
      local_to_global_file << '\n';
    }
  }

  std::cout << "Exported neighbours data to ../../data/" << std::endl;
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

    // Build upper-triangular CSR of A directly from CSR of L for PARDISO.
    // Mapping: A(i,i)=sum_j L(i,j), A(i,j)=-L(i,j) for j>i.
    std::vector<MKL_INT> ia_upper(static_cast<size_t>(nvtxs) + 1, 0);
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      MKL_INT row_nnz = 1; // diagonal
      for (MKL_INT j = l_rows_start[i]; j < l_rows_end[i]; ++j) {
        if (l_col_index[j] > i) {
          ++row_nnz;
        }
      }
      ia_upper[static_cast<size_t>(i + 1)] =
          ia_upper[static_cast<size_t>(i)] + row_nnz;
    }

    const MKL_INT upper_nnz = ia_upper[static_cast<size_t>(nvtxs)];
    std::vector<MKL_INT> ja_upper(static_cast<size_t>(upper_nnz), 0);
    std::vector<double> a_upper(static_cast<size_t>(upper_nnz), 0.0);

    for (MKL_INT i = 0; i < nvtxs; ++i) {
      MKL_INT write = ia_upper[static_cast<size_t>(i)];

      // Diagonal entry first.
      ja_upper[static_cast<size_t>(write)] = i;
      double diag_value = 0.0;
      for (MKL_INT j = l_rows_start[i]; j < l_rows_end[i]; ++j) {
        diag_value += l_val[j];
      }
      a_upper[static_cast<size_t>(write)] = diag_value;
      ++write;

      for (MKL_INT j = l_rows_start[i]; j < l_rows_end[i]; ++j) {
        const MKL_INT col = l_col_index[j];
        if (col > i) {
          ja_upper[static_cast<size_t>(write)] = col;
          a_upper[static_cast<size_t>(write)] = -l_val[j];
          ++write;
        }
      }

      if (write != ia_upper[static_cast<size_t>(i + 1)]) {
        throw std::runtime_error(
            "Internal error constructing upper-triangular A CSR.");
      }
    }

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

    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, a_upper.data(),
            ia_upper.data(), ja_upper.data(), perm, &nrhs, iparm, &msglv1,
            vecRHS.data(), vecSOL.data(), &error);

    if (error != 0) {
      std::cout << "PARDISO error: " << error << std::endl;
    }

    phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, a_upper.data(),
            ia_upper.data(), ja_upper.data(), perm, &nrhs, iparm, &msglv1,
            vecRHS.data(), vecSOL.data(), &error);

    std::vector<double> Ax(static_cast<size_t>(nvtxs), 0.0);
    // a_upper/ia_upper/ja_upper stores only upper triangle for a symmetric A.
    // Expand symmetric contributions so Ax uses the full matrix action.
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      for (MKL_INT idx = ia_upper[static_cast<size_t>(i)];
           idx < ia_upper[static_cast<size_t>(i + 1)]; ++idx) {
        const MKL_INT col = ja_upper[static_cast<size_t>(idx)];
        const double aij = a_upper[static_cast<size_t>(idx)];

        Ax[static_cast<size_t>(i)] += aij * vecSOL[static_cast<size_t>(col)];
        if (col != i) {
          Ax[static_cast<size_t>(col)] += aij * vecSOL[static_cast<size_t>(i)];
        }
      }
    }
    for (MKL_INT i = 0; i < nvtxs; ++i) {
      Ax[static_cast<size_t>(i)] -= vecRHS[static_cast<size_t>(i)];
    }

    const int inc = 1;
    const double rhs_norm = cblas_dnrm2(nvtxs, vecRHS.data(), inc);
    const double res_norm = cblas_dnrm2(nvtxs, Ax.data(), inc);
    const double rel_res = (rhs_norm == 0.0) ? -1.0 : (res_norm / rhs_norm);
    std::cout << "Direct solve residual (abs): " << res_norm
              << ", residual (rel): " << rel_res << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Direct solve completed in " << duration.count() << " ms"
              << std::endl;

    return res_norm;
  } catch (...) {
    throw;
  }
}
