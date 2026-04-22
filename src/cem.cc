#include "cem.hh"
#include <fem/coefficient.hpp>

namespace {
double f_rhs(const mfem::Vector &x) {
  const double pi = M_PI;
  return 2.0 * pi * pi * std::sin(pi * x(0)) * std::sin(pi * x(1));
}

double diffusion_coeff_channel(const mfem::Vector &x) {
  constexpr double k_inside = 1.0e4;
  constexpr double k_outside = 1.0;
  constexpr double half_width = 0.03;
  constexpr double centers[3] = {0.25, 0.5, 0.75};

  const double px = x(0);
  const double py = x(1);

  // Union of 3 horizontal and 3 vertical channels on [0,1]^2.
  for (int i = 0; i < 3; ++i) {
    const double c = centers[i];
    if (std::fabs(py - c) <= half_width || std::fabs(px - c) <= half_width) {
      return k_inside;
    }
  }

  return k_outside;
}

double diffusion_coeff_inclusions(const mfem::Vector &x) {
  constexpr double k_background = 1.0;
  constexpr double k_channel = 1.0e3;
  constexpr double k_fracture = 2.5e3;
  constexpr double k_low = 2.0e-1;

  constexpr double horizontal_centers[5] = {0.12, 0.30, 0.50, 0.70, 0.88};
  constexpr double vertical_centers[4] = {0.20, 0.40, 0.60, 0.80};

  struct Inclusion {
    double cx;
    double cy;
    double rx;
    double ry;
    double k;
  };

  constexpr Inclusion inclusions[] = {
      {0.18, 0.18, 0.05, 0.03, 8.0e3}, {0.33, 0.76, 0.04, 0.06, 6.0e3},
      {0.57, 0.27, 0.06, 0.04, 4.0e3}, {0.76, 0.68, 0.05, 0.05, 9.0e3},
      {0.86, 0.18, 0.03, 0.05, 3.0e3}, {0.48, 0.50, 0.025, 0.025, k_low},
  };

  const double px = x(0);
  const double py = x(1);

  // Wavy channels: large contrast and non-aligned interfaces.
  for (double c : horizontal_centers) {
    const double wiggle = 0.015 * std::sin(8.0 * M_PI * px + 6.0 * c);
    if (std::fabs(py - (c + wiggle)) <= 0.018) {
      return k_channel;
    }
  }

  for (double c : vertical_centers) {
    const double wiggle = 0.015 * std::cos(7.0 * M_PI * py + 5.0 * c);
    if (std::fabs(px - (c + wiggle)) <= 0.018) {
      return k_channel;
    }
  }

  // Two diagonal fractures with medium-high conductivity.
  if (std::fabs(py - (0.10 + 0.75 * px)) <= 0.012 ||
      std::fabs(py - (0.92 - 0.68 * px)) <= 0.012) {
    return k_fracture;
  }

  // Elliptic inclusions (mixed high and low contrasts).
  for (const auto &inc : inclusions) {
    const double dx = (px - inc.cx) / inc.rx;
    const double dy = (py - inc.cy) / inc.ry;
    if (dx * dx + dy * dy <= 1.0) {
      return inc.k;
    }
  }

  // Fine-scale checkerboard patch near the center.
  if (px > 0.35 && px < 0.65 && py > 0.35 && py < 0.65) {
    const int ix = static_cast<int>((px - 0.35) / 0.05);
    const int iy = static_cast<int>((py - 0.35) / 0.05);
    return ((ix + iy) % 2 == 0) ? 2.0e2 : 5.0;
  }

  return k_background;
}

double diffusion_coeff_multichannel_complex(const mfem::Vector &x) {
  constexpr double k_background = 1.0;
  constexpr double k_high = 1.0e1;

  const double px = x(0);
  const double py = x(1);

  // Three parallel high-conductivity channels near each boundary.
  constexpr double edge_centers[3] = {0.03, 0.06, 0.09};
  constexpr double edge_half_width = 0.0075;
  for (double c : edge_centers) {
    const bool near_left = std::fabs(px - c) <= edge_half_width;
    const bool near_right = std::fabs(px - (1.0 - c)) <= edge_half_width;
    const bool near_bottom = std::fabs(py - c) <= edge_half_width;
    const bool near_top = std::fabs(py - (1.0 - c)) <= edge_half_width;
    if (near_left || near_right || near_bottom || near_top) {
      return k_high;
    }
  }

  // Three concentric annular channels around the center.
  constexpr double cx = 0.50;
  constexpr double cy = 0.50;
  constexpr double ring_radii[3] = {0.11, 0.16, 0.21};
  constexpr double ring_half_width = 0.0085;

  const double dx = px - cx;
  const double dy = py - cy;
  const double r = std::sqrt(dx * dx + dy * dy);

  for (double r0 : ring_radii) {
    if (std::fabs(r - r0) <= ring_half_width) {
      return k_high;
    }
  }

  return k_background;
}
} // namespace

CEM::CEM()
    : matA(nullptr), matL(nullptr), matR(nullptr),
      indexing(SPARSE_INDEX_BASE_ZERO), rows(0), cols(0), nvtxs(0), nparts(10),
      overlap(2), k0(3), cStar(1.0) {}

CEM::CEM(int k0_)
    : matA(nullptr), matL(nullptr), matR(nullptr),
      indexing(SPARSE_INDEX_BASE_ZERO), rows(0), cols(0), nvtxs(0), nparts(10),
      overlap(2), k0(k0_), cStar(1.0) {
  if (k0 <= 0) {
    throw std::runtime_error("k0 must be positive.");
  }
}

CEM::CEM(int nparts_, int overlap_, int k0_)
    : matA(nullptr), matL(nullptr), matR(nullptr),
      indexing(SPARSE_INDEX_BASE_ZERO), rows(0), cols(0), nvtxs(0),
      nparts(nparts_), overlap(overlap_), k0(k0_), cStar(1.0) {
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
  destroyIfAllocated(matR);
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
  double (*diffusion_fn)(const mfem::Vector &) = diffusion_coeff_inclusions;
  // Alternatives: diffusion_coeff_channel or diffusion_coeff_inclusions or
  // diffusion_coeff_multichannel_complex.
  mfem::FunctionCoefficient diffusion_coef(diffusion_fn);
  a.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion_coef));
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

  std::cout << "======Partition sizes (local counts)======" << std::endl;
  MKL_INT count_max = 0;
  MKL_INT count_min = std::numeric_limits<MKL_INT>::max();
  for (idx_t i = 0; i < nparts; ++i) {
    const MKL_INT c = count[static_cast<size_t>(i)];
    if (c > count_max) {
      count_max = c;
    }
    if (c < count_min) {
      count_min = c;
    }
  }
  std::cout << "Max partition size: " << count_max
            << ", Min partition size: " << count_min << std::endl;

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

  std::vector<std::vector<std::vector<std::pair<MKL_INT, double>>>> Ai_rows(
      static_cast<size_t>(nparts));
  std::vector<std::vector<std::vector<std::pair<MKL_INT, double>>>> Si_rows(
      static_cast<size_t>(nparts));
  for (idx_t i = 0; i < nparts; ++i) {
    const size_t part_id = static_cast<size_t>(i);
    const MKL_INT count_i = count[part_id];
    if (count_i < 0) {
      throw std::runtime_error("Negative local vertex count in formAUX().");
    }
    Ai_rows[part_id].resize(static_cast<size_t>(count_i));
    Si_rows[part_id].resize(static_cast<size_t>(count_i));
  }

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
        const MKL_INT count_i = count[part_id];
        if (local_i < 0 || local_i >= count_i) {
          throw std::runtime_error(
              "Invalid local row index while building AUX matrices.");
        }

        auto &ai_row = Ai_rows[part_id][static_cast<size_t>(local_i)];
        auto &si_row = Si_rows[part_id][static_cast<size_t>(local_i)];

        if (col != i) {
          const MKL_INT local_col = globalTolocal[static_cast<size_t>(col)];
          if (local_col < 0 || local_col >= count_i) {
            throw std::runtime_error(
                "Invalid local column index while building AUX matrices.");
          }

          ai_row.push_back({local_i, val[j]});
          ai_row.push_back({local_col, -val[j]});

          si_row.push_back({local_i, val[j] / (cStar * cStar * 2.0)});
        } else {
          ai_row.push_back({local_i, val[j]});

          si_row.push_back({local_i, val[j] / (cStar * cStar)});
        }
      }
    }
  }

  std::vector<std::vector<MKL_INT>> Ai_rows_start(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Ai_rows_end(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Ai_col_index(static_cast<size_t>(nparts));
  std::vector<std::vector<double>> Ai_values(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Si_rows_start(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Si_rows_end(static_cast<size_t>(nparts));
  std::vector<std::vector<MKL_INT>> Si_col_index(static_cast<size_t>(nparts));
  std::vector<std::vector<double>> Si_values(static_cast<size_t>(nparts));

  auto buildLocalCSR =
      [](std::vector<std::vector<std::pair<MKL_INT, double>>> &rows_data,
         std::vector<MKL_INT> &rows_start_out,
         std::vector<MKL_INT> &rows_end_out,
         std::vector<MKL_INT> &col_index_out, std::vector<double> &values_out) {
        const MKL_INT nrows = static_cast<MKL_INT>(rows_data.size());
        rows_start_out.assign(static_cast<size_t>(nrows), 0);
        rows_end_out.assign(static_cast<size_t>(nrows), 0);

        MKL_INT nnz = 0;
        for (MKL_INT row = 0; row < nrows; ++row) {
          auto &entries = rows_data[static_cast<size_t>(row)];
          std::sort(entries.begin(), entries.end(),
                    [](const std::pair<MKL_INT, double> &lhs,
                       const std::pair<MKL_INT, double> &rhs) {
                      return lhs.first < rhs.first;
                    });

          size_t write = 0;
          for (size_t idx = 0; idx < entries.size(); ++idx) {
            if (write > 0 && entries[write - 1].first == entries[idx].first) {
              entries[write - 1].second += entries[idx].second;
            } else {
              entries[write++] = entries[idx];
            }
          }
          entries.resize(write);

          rows_start_out[static_cast<size_t>(row)] = nnz;
          nnz += static_cast<MKL_INT>(entries.size());
          rows_end_out[static_cast<size_t>(row)] = nnz;
        }

        col_index_out.assign(static_cast<size_t>(nnz), 0);
        values_out.assign(static_cast<size_t>(nnz), 0.0);
        MKL_INT out = 0;
        for (MKL_INT row = 0; row < nrows; ++row) {
          const auto &entries = rows_data[static_cast<size_t>(row)];
          for (const auto &entry : entries) {
            col_index_out[static_cast<size_t>(out)] = entry.first;
            values_out[static_cast<size_t>(out)] = entry.second;
            ++out;
          }
        }
      };

  for (idx_t i = 0; i < nparts; ++i) {
    const size_t part_id = static_cast<size_t>(i);
    buildLocalCSR(Ai_rows[part_id], Ai_rows_start[part_id],
                  Ai_rows_end[part_id], Ai_col_index[part_id],
                  Ai_values[part_id]);
    buildLocalCSR(Si_rows[part_id], Si_rows_start[part_id],
                  Si_rows_end[part_id], Si_col_index[part_id],
                  Si_values[part_id]);
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

  std::vector<std::vector<double>> res(
      static_cast<size_t>(nparts),
      std::vector<double>(static_cast<size_t>(k0), 0.0));

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  descr.mode = SPARSE_FILL_MODE_UPPER;

#pragma omp parallel for schedule(static) num_threads(80)
  for (idx_t i = 0; i < nparts; ++i) {
    const size_t part_id = static_cast<size_t>(i);
    const MKL_INT count_i = count[part_id];
    if (count_i <= 0) {
      std::cout << "part: " << i << " skipped in formAUX because count is zero."
                << std::endl;
      continue;
    }

    sparse_matrix_t Ai = nullptr;
    sparse_matrix_t Si = nullptr;

    if (Ai_rows_start[part_id].size() != static_cast<size_t>(count_i) ||
        Ai_rows_end[part_id].size() != static_cast<size_t>(count_i) ||
        Si_rows_start[part_id].size() != static_cast<size_t>(count_i) ||
        Si_rows_end[part_id].size() != static_cast<size_t>(count_i)) {
      throw std::runtime_error(
          "Internal error: invalid local CSR row pointer size in formAUX().");
    }

    for (MKL_INT row = 0; row < count_i; ++row) {
      const MKL_INT ai_start = Ai_rows_start[part_id][static_cast<size_t>(row)];
      const MKL_INT ai_end = Ai_rows_end[part_id][static_cast<size_t>(row)];
      const MKL_INT si_start = Si_rows_start[part_id][static_cast<size_t>(row)];
      const MKL_INT si_end = Si_rows_end[part_id][static_cast<size_t>(row)];
      if (ai_start > ai_end || si_start > si_end) {
        throw std::runtime_error("Internal error: non-monotonic local CSR row "
                                 "pointers in formAUX().");
      }
      for (MKL_INT idx = ai_start; idx < ai_end; ++idx) {
        const MKL_INT c = Ai_col_index[part_id][static_cast<size_t>(idx)];
        if (c < 0 || c >= count_i) {
          throw std::runtime_error(
              "Invalid local Ai column index in formAUX().");
        }
      }
      for (MKL_INT idx = si_start; idx < si_end; ++idx) {
        const MKL_INT c = Si_col_index[part_id][static_cast<size_t>(idx)];
        if (c < 0 || c >= count_i) {
          throw std::runtime_error(
              "Invalid local Si column index in formAUX().");
        }
      }
    }

    MKL_INT *ai_col_ptr =
        Ai_col_index[part_id].empty() ? nullptr : Ai_col_index[part_id].data();
    double *ai_val_ptr =
        Ai_values[part_id].empty() ? nullptr : Ai_values[part_id].data();
    MKL_INT *si_col_ptr =
        Si_col_index[part_id].empty() ? nullptr : Si_col_index[part_id].data();
    double *si_val_ptr =
        Si_values[part_id].empty() ? nullptr : Si_values[part_id].data();

    const sparse_status_t ai_csr_status = mkl_sparse_d_create_csr(
        &Ai, indexing, count_i, count_i, Ai_rows_start[part_id].data(),
        Ai_rows_end[part_id].data(), ai_col_ptr, ai_val_ptr);
    const sparse_status_t si_csr_status = mkl_sparse_d_create_csr(
        &Si, indexing, count_i, count_i, Si_rows_start[part_id].data(),
        Si_rows_end[part_id].data(), si_col_ptr, si_val_ptr);

    if (ai_csr_status != SPARSE_STATUS_SUCCESS ||
        si_csr_status != SPARSE_STATUS_SUCCESS) {
      if (Ai != nullptr) {
        mkl_sparse_destroy(Ai);
      }
      if (Si != nullptr) {
        mkl_sparse_destroy(Si);
      }
      throw std::runtime_error(
          "mkl_sparse_d_create_csr failed for AUX local matrices.");
    }

    MKL_INT k = 0;

    const sparse_status_t gv_status = mkl_sparse_d_gv(
        &which, pm, Ai, descr, Si, descr, k0, &k, eigenvalue[part_id].data(),
        eigenvector[part_id].data(), res[part_id].data());

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
    // std::cout << "part: " << i << " residual: " << res[part_id][0]
    //           << " Smallest eigenvalue: " << eigenvalue[part_id][0]
    //           << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finish solving eigen problem in each coarse element with "
            << duration.count() << " ms======" << std::endl;
}

void CEM::formCEM() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase IV: Construct the CEM basis======" << std::endl;

  if (matL == nullptr) {
    throw std::runtime_error(
        "matL is not initialized. Call getDatafromMFEM() first.");
  }
  if (nvtxs <= 0 || nparts <= 0 || k0 <= 0) {
    throw std::runtime_error("Invalid nvtxs/nparts/k0 in formCEM().");
  }
  if (cStar <= 0.0) {
    throw std::runtime_error("cStar must be positive in formCEM().");
  }
  if (vertices.size() != static_cast<size_t>(nparts) ||
      overlapping.size() != static_cast<size_t>(nparts) ||
      verticesCEM.size() != static_cast<size_t>(nparts) ||
      globalTolocalCEM.size() != static_cast<size_t>(nparts) ||
      localtoGlobalCEM.size() != static_cast<size_t>(nparts)) {
    throw std::runtime_error(
        "Neighbour metadata missing. Call findNeighbours() first.");
  }
  if (count.size() != static_cast<size_t>(nparts) ||
      globalTolocal.size() != static_cast<size_t>(nvtxs)) {
    throw std::runtime_error("Local indexing metadata mismatch in formCEM().");
  }
  if (eigenvector.size() != static_cast<size_t>(nparts) ||
      eigenvalue.size() != static_cast<size_t>(nparts)) {
    throw std::runtime_error(
        "Auxiliary space missing. Call formAUX() before formCEM().");
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
  // 这里先计算了sData，后续计算S_i矩阵的时候直接用sData乘就好了，不需要每次都去遍历CSR了
  const double cstar2 = cStar * cStar;
  std::vector<double> sData(static_cast<size_t>(nvtxs), 0.0);
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    for (MKL_INT j = rows_start[i]; j < rows_end[i]; ++j) {
      if (col_index[j] == i) {
        sData[static_cast<size_t>(i)] += val[j] / cstar2;
      } else {
        sData[static_cast<size_t>(i)] += val[j] / (2.0 * cstar2);
      }
    }
  }

  // sMatrix用来存储每个子域的S_i矩阵，后续计算CEM基底的时候直接用sMatrix乘就好了，不需要每次都去遍历CSR了
  std::vector<std::vector<double>> sMatrix(static_cast<size_t>(nparts));
  for (idx_t p = 0; p < nparts; ++p) {
    const size_t part_id = static_cast<size_t>(p);
    const MKL_INT local_n = count[part_id];
    if (local_n < 0) {
      throw std::runtime_error("Negative local count found in formCEM().");
    }

    const size_t local_n_sz = static_cast<size_t>(local_n);
    sMatrix[part_id].assign(local_n_sz * local_n_sz, 0.0);

    const auto &vecs = eigenvector[part_id];
    if (vecs.size() != static_cast<size_t>(k0) * local_n_sz) {
      throw std::runtime_error(
          "eigenvector size mismatch. Ensure formAUX() finished correctly.");
    }

    for (const auto &g1 : vertices[part_id]) {
      const MKL_INT l1 = globalTolocal[static_cast<size_t>(g1)];
      if (l1 < 0 || l1 >= local_n) {
        throw std::runtime_error("Invalid local index l1 in formCEM().");
      }
      for (const auto &g2 : vertices[part_id]) {
        const MKL_INT l2 = globalTolocal[static_cast<size_t>(g2)];
        if (l2 < 0 || l2 >= local_n) {
          throw std::runtime_error("Invalid local index l2 in formCEM().");
        }

        double entry = 0.0;
        for (int e = 0; e < k0; ++e) {
          const double v1 = vecs[static_cast<size_t>(e) * local_n_sz +
                                 static_cast<size_t>(l1)];
          const double v2 = vecs[static_cast<size_t>(e) * local_n_sz +
                                 static_cast<size_t>(l2)];
          entry += v1 * v2;
        }
        entry *=
            sData[static_cast<size_t>(g1)] * sData[static_cast<size_t>(g2)];

        sMatrix[part_id][static_cast<size_t>(l1) * local_n_sz +
                         static_cast<size_t>(l2)] = entry;
      }
    }
  }

  auto s_matrix_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> s_duration = s_matrix_end - start;
  std::cout << "======Finish forming all S_i in " << s_duration.count()
            << " ms======" << std::endl;

  cemBasis.assign(static_cast<size_t>(nparts), std::vector<double>{});
  std::cout
      << "======Start calculating CEM Basis in each overlapping area======"
      << std::endl;

  auto buildLocalCSR =
      [](std::vector<std::vector<std::pair<MKL_INT, double>>> &rows_data,
         std::vector<MKL_INT> &ia, std::vector<MKL_INT> &rows_start_out,
         std::vector<MKL_INT> &rows_end_out, std::vector<MKL_INT> &ja,
         std::vector<double> &a) {
        const MKL_INT nrows = static_cast<MKL_INT>(rows_data.size());
        ia.assign(static_cast<size_t>(nrows) + 1, 0);

        for (MKL_INT row = 0; row < nrows; ++row) {
          auto &entries = rows_data[static_cast<size_t>(row)];
          std::sort(entries.begin(), entries.end(),
                    [](const std::pair<MKL_INT, double> &lhs,
                       const std::pair<MKL_INT, double> &rhs) {
                      return lhs.first < rhs.first;
                    });

          size_t write = 0;
          for (size_t idx = 0; idx < entries.size(); ++idx) {
            if (write > 0 && entries[write - 1].first == entries[idx].first) {
              entries[write - 1].second += entries[idx].second;
            } else {
              entries[write++] = entries[idx];
            }
          }
          entries.resize(write);

          ia[static_cast<size_t>(row + 1)] =
              ia[static_cast<size_t>(row)] +
              static_cast<MKL_INT>(entries.size());
        }

        const MKL_INT nnz = ia[static_cast<size_t>(nrows)];
        rows_start_out.assign(static_cast<size_t>(nrows), 0);
        rows_end_out.assign(static_cast<size_t>(nrows), 0);
        for (MKL_INT row = 0; row < nrows; ++row) {
          rows_start_out[static_cast<size_t>(row)] =
              ia[static_cast<size_t>(row)];
          rows_end_out[static_cast<size_t>(row)] =
              ia[static_cast<size_t>(row + 1)];
        }

        ja.assign(static_cast<size_t>(nnz), 0);
        a.assign(static_cast<size_t>(nnz), 0.0);
        MKL_INT out = 0;
        for (MKL_INT row = 0; row < nrows; ++row) {
          const auto &entries = rows_data[static_cast<size_t>(row)];
          for (const auto &entry : entries) {
            ja[static_cast<size_t>(out)] = entry.first;
            a[static_cast<size_t>(out)] = entry.second;
            ++out;
          }
        }
      };

#pragma omp parallel for schedule(static) num_threads(80)
  for (idx_t p = 0; p < nparts; ++p) {
    const size_t part_id = static_cast<size_t>(p);
    const auto &overlap_vertices =
        verticesCEM[part_id]; // 该子域重叠区域的全局顶点集合
    const auto &g2l_overlap =
        globalTolocalCEM[part_id]; // 该子域重叠区域的全局顶点到局部索引的映射
    const auto &l2g_overlap =
        localtoGlobalCEM[part_id]; // 该子域重叠区域的局部索引到全局顶点的映射

    const MKL_INT n_overlap = static_cast<MKL_INT>(l2g_overlap.size());
    if (n_overlap == 0) {
      cemBasis[part_id].assign(0, 0.0);
      std::cout << "part: " << p << " skipped in formCEM because overlap is 0"
                << std::endl;
      continue;
    }
    if (overlap_vertices.size() != static_cast<size_t>(n_overlap) ||
        g2l_overlap.size() != static_cast<size_t>(n_overlap)) {
      throw std::runtime_error(
          "Inconsistent overlap mapping size in formCEM().");
    }

    std::vector<std::vector<std::pair<MKL_INT, double>>> local_rows(
        static_cast<size_t>(n_overlap));
    // 先把重叠区域内的L矩阵行按照CEM的方式组装到local_rows中，后续再把S_i矩阵的贡献加上去

    for (const auto &g_row : overlap_vertices) {
      const auto row_it = g2l_overlap.find(g_row);
      if (row_it == g2l_overlap.end()) {
        throw std::runtime_error("Missing overlap row mapping in formCEM().");
      }
      const MKL_INT l_row = static_cast<MKL_INT>(row_it->second);
      if (l_row < 0 || l_row >= n_overlap) {
        throw std::runtime_error("Invalid overlap local row in formCEM().");
      }

      // 遍历matL的CSR结构，找到全局行g_row对应的非零列g_col和数值val[j]，如果g_col在重叠区域内，就根据CEM的组装方式把对应的行列和数值加入local_rows中

      for (MKL_INT j = rows_start[g_row]; j < rows_end[g_row]; ++j) {
        const MKL_INT g_col = col_index[j];
        if (g_col < 0 || g_col >= nvtxs) {
          throw std::runtime_error("Invalid global column in formCEM().");
        }
        if (overlap_vertices.find(static_cast<idx_t>(g_col)) !=
            overlap_vertices.end()) {
          const auto col_it = g2l_overlap.find(static_cast<idx_t>(g_col));
          if (col_it == g2l_overlap.end()) {
            throw std::runtime_error(
                "Missing overlap column mapping in formCEM().");
          }
          const MKL_INT l_col = static_cast<MKL_INT>(col_it->second);
          if (l_col < 0 || l_col >= n_overlap) {
            throw std::runtime_error(
                "Invalid overlap local column in formCEM().");
          }

          if (l_col <= l_row) {
            local_rows[static_cast<size_t>(l_row)].push_back({l_row, val[j]});
          } else {
            local_rows[static_cast<size_t>(l_row)].push_back({l_row, val[j]});
            local_rows[static_cast<size_t>(l_row)].push_back({l_col, -val[j]});
          }
        } else {
          local_rows[static_cast<size_t>(l_row)].push_back({l_row, val[j]});
        }
      }
    }

    // 把S_i矩阵的贡献加到local_rows中
    for (const auto &ov_part : overlapping[part_id]) {
      const size_t ov_part_id = static_cast<size_t>(ov_part);
      const MKL_INT ov_local_n = count[ov_part_id];
      if (ov_local_n < 0) {
        throw std::runtime_error("Negative part size in overlap set.");
      }
      const size_t ov_local_n_sz = static_cast<size_t>(ov_local_n);
      const auto &ov_vertices = vertices[ov_part_id];
      for (const auto &g1 : ov_vertices) {
        const auto l1_it = g2l_overlap.find(g1);
        if (l1_it == g2l_overlap.end()) {
          throw std::runtime_error(
              "Missing overlap local row for S_i assembly.");
        }
        const MKL_INT l1 = static_cast<MKL_INT>(l1_it->second);
        const MKL_INT c1 = globalTolocal[static_cast<size_t>(g1)];
        if (c1 < 0 || c1 >= ov_local_n) {
          throw std::runtime_error("Invalid core local index c1 in formCEM().");
        }

        for (const auto &g2 : ov_vertices) {
          const auto l2_it = g2l_overlap.find(g2);
          if (l2_it == g2l_overlap.end()) {
            throw std::runtime_error(
                "Missing overlap local column for S_i assembly.");
          }
          const MKL_INT l2 = static_cast<MKL_INT>(l2_it->second);
          if (l1 <= l2) {
            const MKL_INT c2 = globalTolocal[static_cast<size_t>(g2)];
            if (c2 < 0 || c2 >= ov_local_n) {
              throw std::runtime_error(
                  "Invalid core local index c2 in formCEM().");
            }
            const double sij =
                sMatrix[ov_part_id][static_cast<size_t>(c1) * ov_local_n_sz +
                                    static_cast<size_t>(c2)];
            local_rows[static_cast<size_t>(l1)].push_back({l2, sij});
          }
        }
      }
    }

    std::vector<MKL_INT> ia;
    std::vector<MKL_INT> rows_start_local;
    std::vector<MKL_INT> rows_end_local;
    std::vector<MKL_INT> ja;
    std::vector<double> a;
    buildLocalCSR(local_rows, ia, rows_start_local, rows_end_local, ja, a);

    sparse_matrix_t Ai = nullptr;
    MKL_INT *ja_ptr = ja.empty() ? nullptr : ja.data();
    double *a_ptr = a.empty() ? nullptr : a.data();
    const sparse_status_t create_status = mkl_sparse_d_create_csr(
        &Ai, indexing, n_overlap, n_overlap, rows_start_local.data(),
        rows_end_local.data(), ja_ptr, a_ptr);
    if (create_status != SPARSE_STATUS_SUCCESS) {
      throw std::runtime_error(
          "mkl_sparse_d_create_csr failed for local CEM problem.");
    }

    std::vector<double> rhs(
        static_cast<size_t>(n_overlap) * static_cast<size_t>(k0), 0.0);
    const MKL_INT local_n = count[part_id];
    const size_t local_n_sz = static_cast<size_t>(local_n);
    for (int j = 0; j < k0; ++j) {
      for (const auto &g : vertices[part_id]) {
        const auto it = g2l_overlap.find(g);
        if (it == g2l_overlap.end()) {
          throw std::runtime_error(
              "Missing overlap mapping while forming CEM rhs.");
        }
        const MKL_INT overlap_local = static_cast<MKL_INT>(it->second);
        const MKL_INT core_local = globalTolocal[static_cast<size_t>(g)];
        if (core_local < 0 || core_local >= local_n) {
          throw std::runtime_error(
              "Invalid core local index while forming CEM rhs.");
        }
        rhs[static_cast<size_t>(j) * static_cast<size_t>(n_overlap) +
            static_cast<size_t>(overlap_local)] =
            sData[static_cast<size_t>(g)] *
            eigenvector[part_id][static_cast<size_t>(j) * local_n_sz +
                                 static_cast<size_t>(core_local)];
      }
    }

    cemBasis[part_id].assign(
        static_cast<size_t>(n_overlap) * static_cast<size_t>(k0), 0.0);

    MKL_INT perm[64];
    MKL_INT iparm[64];
    void *pt[64];
    for (int idx = 0; idx < 64; ++idx) {
      pt[idx] = 0;
      iparm[idx] = 0;
      perm[idx] = 0;
    }

    iparm[34] = 1;
    iparm[0] = 1;

    MKL_INT error = 0;
    MKL_INT maxfct = 1;
    MKL_INT mnum = 1;
    MKL_INT mtype = 2;
    MKL_INT phase = 13;
    MKL_INT msglvl = 0;
    MKL_INT nrhs = static_cast<MKL_INT>(k0);

    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n_overlap, a.data(), ia.data(),
            ja.data(), perm, &nrhs, iparm, &msglvl, rhs.data(),
            cemBasis[part_id].data(), &error);

    if (error != 0) {
      mkl_sparse_destroy(Ai);
      throw std::runtime_error("PARDISO failed in formCEM() with error " +
                               std::to_string(static_cast<int>(error)));
    }

    phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n_overlap, a.data(), ia.data(),
            ja.data(), perm, &nrhs, iparm, &msglvl, rhs.data(),
            cemBasis[part_id].data(), &error);

    mkl_sparse_destroy(Ai);

    if (error != 0) {
      throw std::runtime_error(
          "PARDISO release failed in formCEM() with error " +
          std::to_string(static_cast<int>(error)));
    }

    // std::cout << "part: " << p << " overlap_size: " << n_overlap
    //           << " rhs: " << nrhs << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished CEM basis construction with " << duration.count()
            << " ms======" << std::endl;
}

void CEM::formMatR() {
  auto start = std::chrono::high_resolution_clock::now();

  if (nvtxs <= 0 || nparts <= 0 || k0 <= 0) {
    throw std::runtime_error("Invalid nvtxs/nparts/k0 in formMatR().");
  }
  if (cemBasis.size() != static_cast<size_t>(nparts) ||
      localtoGlobalCEM.size() != static_cast<size_t>(nparts)) {
    throw std::runtime_error(
        "CEM basis or overlap map missing. Call formCEM() first.");
  }

  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;
  row_indx.reserve(static_cast<size_t>(k0) * static_cast<size_t>(nparts) * 64);
  col_indx.reserve(static_cast<size_t>(k0) * static_cast<size_t>(nparts) * 64);
  values.reserve(static_cast<size_t>(k0) * static_cast<size_t>(nparts) * 64);

  for (idx_t p = 0; p < nparts; ++p) {
    const size_t part_id = static_cast<size_t>(p);
    const auto &l2g = localtoGlobalCEM[part_id];
    const size_t n_overlap = l2g.size();
    const auto &basis = cemBasis[part_id];
    const size_t expected = static_cast<size_t>(k0) * n_overlap;
    if (basis.size() != expected) {
      throw std::runtime_error(
          "cemBasis size mismatch while forming R matrix.");
    }

    for (int j = 0; j < k0; ++j) {
      const MKL_INT row = static_cast<MKL_INT>(p * static_cast<idx_t>(k0) +
                                               static_cast<idx_t>(j));
      for (size_t k = 0; k < n_overlap; ++k) {
        const MKL_INT col = static_cast<MKL_INT>(l2g[k]);
        if (col < 0 || col >= nvtxs) {
          throw std::runtime_error(
              "Invalid global column index while forming R matrix.");
        }
        row_indx.push_back(row);
        col_indx.push_back(col);
        values.push_back(basis[static_cast<size_t>(j) * n_overlap + k]);
      }
    }
  }

  destroyIfAllocated(matR);
  sparse_matrix_t matR_coo = nullptr;
  const MKL_INT r_rows =
      static_cast<MKL_INT>(nparts) * static_cast<MKL_INT>(k0);
  const sparse_status_t create_status = mkl_sparse_d_create_coo(
      &matR_coo, indexing, r_rows, nvtxs, static_cast<MKL_INT>(values.size()),
      row_indx.data(), col_indx.data(), values.data());
  if (create_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        "mkl_sparse_d_create_coo failed while forming R matrix.");
  }

  const sparse_status_t csr_status =
      mkl_sparse_convert_csr(matR_coo, SPARSE_OPERATION_NON_TRANSPOSE, &matR);
  mkl_sparse_destroy(matR_coo);
  if (csr_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        "mkl_sparse_convert_csr failed while forming R matrix.");
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finish forming the matrix R in " << duration.count()
            << " ms======" << std::endl;
}

void CEM::solveCEM() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase V: Solve the CEM problem======" << std::endl;

  if (matL == nullptr) {
    throw std::runtime_error(
        "matL is not initialized. Call getDatafromMFEM() first.");
  }
  if (matR == nullptr) {
    throw std::runtime_error(
        "matR is not initialized. Call formMatR() before solveCEM().");
  }
  if (vecRHS.size() != static_cast<size_t>(nvtxs)) {
    throw std::runtime_error(
        "RHS size mismatch. Call getDatafromMFEM() before solveCEM().");
  }
  if (nvtxs <= 0 || nparts <= 0 || k0 <= 0) {
    throw std::runtime_error("Invalid nvtxs/nparts/k0 in solveCEM().");
  }

  MKL_INT *rows_start = nullptr;
  MKL_INT *rows_end = nullptr;
  MKL_INT *col_index = nullptr;
  double *val = nullptr;
  const sparse_status_t export_l_status = mkl_sparse_d_export_csr(
      matL, &indexing, &rows, &cols, &rows_start, &rows_end, &col_index, &val);
  if (export_l_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        "mkl_sparse_d_export_csr(matL) failed with status " +
        std::to_string(static_cast<int>(export_l_status)));
  }

  const MKL_INT l_nnz = rows_end[nvtxs - 1];
  std::vector<MKL_INT> A_row_index;
  std::vector<MKL_INT> A_col_index;
  std::vector<double> A_values;
  A_row_index.reserve(static_cast<size_t>(2 * l_nnz));
  A_col_index.reserve(static_cast<size_t>(2 * l_nnz));
  A_values.reserve(static_cast<size_t>(2 * l_nnz));

  for (MKL_INT i = 0; i < nvtxs; ++i) {
    for (MKL_INT j = rows_start[i]; j < rows_end[i]; ++j) {
      A_row_index.push_back(i);
      A_col_index.push_back(i);
      A_values.push_back(val[j]);
      if (col_index[j] != i) {
        A_row_index.push_back(i);
        A_col_index.push_back(col_index[j]);
        A_values.push_back(-val[j]);
      }
    }
  }

  sparse_matrix_t Acoo = nullptr;
  sparse_matrix_t A = nullptr;
  const sparse_status_t a_coo_status = mkl_sparse_d_create_coo(
      &Acoo, indexing, nvtxs, nvtxs, static_cast<MKL_INT>(A_values.size()),
      A_row_index.data(), A_col_index.data(), A_values.data());
  if (a_coo_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("mkl_sparse_d_create_coo failed while forming A.");
  }

  const sparse_status_t a_csr_status =
      mkl_sparse_convert_csr(Acoo, SPARSE_OPERATION_NON_TRANSPOSE, &A);
  mkl_sparse_destroy(Acoo);
  if (a_csr_status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("mkl_sparse_convert_csr failed while forming A.");
  }

  auto end_form_A = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> form_a_duration =
      end_form_A - start;
  std::cout << "======Finish forming the matrix A in "
            << form_a_duration.count() << " ms======" << std::endl;

  sparse_matrix_t RA = nullptr;
  const sparse_status_t ra_status =
      mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, matR, A, &RA);
  if (ra_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(A);
    throw std::runtime_error("mkl_sparse_spmm failed while forming RA.");
  }

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  descr.mode = SPARSE_FILL_MODE_FULL;

  sparse_matrix_t ACEM = nullptr;
  const sparse_status_t acem_status = mkl_sparse_sp2m(
      SPARSE_OPERATION_NON_TRANSPOSE, descr, RA, SPARSE_OPERATION_TRANSPOSE,
      descr, matR, SPARSE_STAGE_FULL_MULT, &ACEM);
  if (acem_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(RA);
    mkl_sparse_destroy(A);
    throw std::runtime_error("mkl_sparse_sp2m failed while forming ACEM.");
  }

  const sparse_status_t order_status = mkl_sparse_order(ACEM);
  if (order_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(RA);
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error("mkl_sparse_order failed for ACEM.");
  }

  const MKL_INT coarse_size =
      static_cast<MKL_INT>(nparts) * static_cast<MKL_INT>(k0);
  std::vector<double> cemRHS(static_cast<size_t>(coarse_size), 0.0);
  const sparse_status_t rhs_status =
      mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matR, descr,
                      vecRHS.data(), 0.0, cemRHS.data());
  if (rhs_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(RA);
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error("mkl_sparse_d_mv failed while forming CEM rhs.");
  }

  auto end_form_acem = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> form_acem_duration =
      end_form_acem - end_form_A;
  std::cout << "======Finish forming the matrix A(CEM) and rhs(CEM) in "
            << form_acem_duration.count() << " ms======" << std::endl;

  mkl_sparse_destroy(RA);

  MKL_INT *rows_start_new = nullptr;
  MKL_INT *rows_end_new = nullptr;
  MKL_INT *col_index_new = nullptr;
  double *val_new = nullptr;
  MKL_INT rows_new = 0;
  MKL_INT cols_new = 0;
  const sparse_status_t export_acem_status = mkl_sparse_d_export_csr(
      ACEM, &indexing, &rows_new, &cols_new, &rows_start_new, &rows_end_new,
      &col_index_new, &val_new);
  if (export_acem_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error(
        "mkl_sparse_d_export_csr(ACEM) failed with status " +
        std::to_string(static_cast<int>(export_acem_status)));
  }

  std::cout << "======In CEM we solve a system with " << rows_new
            << " rows and " << cols_new << " columns" << std::endl;

  MKL_INT perm[64], iparm[64];
  void *pt[64];
  for (int i = 0; i < 64; ++i) {
    pt[i] = 0;
    iparm[i] = 0;
    perm[i] = 0;
  }

  iparm[34] = 1;
  iparm[0] = 1;
  iparm[1] = 3;
  iparm[26] = 1;

  MKL_INT error = 0;
  MKL_INT maxfct = 1;
  MKL_INT mnum = 1;
  MKL_INT mtype = 11;
  MKL_INT phase = 13;
  MKL_INT nrhs = 1;
  MKL_INT msglvl = 0;

  cemSOL.assign(static_cast<size_t>(coarse_size), 0.0);

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &coarse_size, val_new,
          rows_start_new, col_index_new, perm, &nrhs, iparm, &msglvl,
          cemRHS.data(), cemSOL.data(), &error);
  if (error != 0) {
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error("PARDISO failed in solveCEM() with error " +
                             std::to_string(static_cast<int>(error)));
  }

  phase = -1;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &coarse_size, val_new,
          rows_start_new, col_index_new, perm, &nrhs, iparm, &msglvl,
          cemRHS.data(), cemSOL.data(), &error);
  if (error != 0) {
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error(
        "PARDISO release failed in solveCEM() with error " +
        std::to_string(static_cast<int>(error)));
  }

  std::vector<double> vecCEM(static_cast<size_t>(nvtxs), 0.0);
  const sparse_status_t recover_status =
      mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, matR, descr,
                      cemSOL.data(), 0.0, vecCEM.data());
  if (recover_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error(
        "mkl_sparse_d_mv failed while recovering fine-grid CEM solution.");
  }

  // Compute ||A*u_cem - b|| using the same symmetric expansion logic as
  // solveFromLAndReleaseA() for a consistent residual definition.
  MKL_INT *l_rows_start = nullptr;
  MKL_INT *l_rows_end = nullptr;
  MKL_INT *l_col_index = nullptr;
  double *l_val = nullptr;
  const sparse_status_t export_l_for_res_status =
      mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &l_rows_start,
                              &l_rows_end, &l_col_index, &l_val);
  if (export_l_for_res_status != SPARSE_STATUS_SUCCESS) {
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(ACEM);
    throw std::runtime_error(
        "mkl_sparse_d_export_csr(matL) failed while computing CEM residual "
        "with status " +
        std::to_string(static_cast<int>(export_l_for_res_status)));
  }

  std::vector<MKL_INT> ia_upper(static_cast<size_t>(nvtxs) + 1, 0);
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    MKL_INT row_nnz = 1;
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
      mkl_sparse_destroy(A);
      mkl_sparse_destroy(ACEM);
      throw std::runtime_error(
          "Internal error constructing upper-triangular A CSR in solveCEM().");
    }
  }

  std::vector<double> Ax_residual(static_cast<size_t>(nvtxs), 0.0);
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    for (MKL_INT idx = ia_upper[static_cast<size_t>(i)];
         idx < ia_upper[static_cast<size_t>(i + 1)]; ++idx) {
      const MKL_INT col = ja_upper[static_cast<size_t>(idx)];
      const double aij = a_upper[static_cast<size_t>(idx)];

      Ax_residual[static_cast<size_t>(i)] +=
          aij * vecCEM[static_cast<size_t>(col)];
      if (col != i) {
        Ax_residual[static_cast<size_t>(col)] +=
            aij * vecCEM[static_cast<size_t>(i)];
      }
    }
  }
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    Ax_residual[static_cast<size_t>(i)] -= vecRHS[static_cast<size_t>(i)];
  }

  const int inc = 1;
  const double rhs_norm = cblas_dnrm2(nvtxs, vecRHS.data(), inc);
  const double cem_abs_res = cblas_dnrm2(nvtxs, Ax_residual.data(), inc);
  const double cem_rel_res =
      (rhs_norm == 0.0) ? -1.0 : (cem_abs_res / rhs_norm);
  std::cout << "CEM solve residual (abs): " << cem_abs_res
            << ", residual (rel): " << cem_rel_res << std::endl;

  bool has_direct_reference = false;
  for (MKL_INT i = 0; i < nvtxs; ++i) {
    if (std::fabs(vecSOL[static_cast<size_t>(i)]) > 0.0) {
      has_direct_reference = true;
      break;
    }
  }
  if (!has_direct_reference) {
    solveFromLAndReleaseA();
  }

  if (vecSOL.size() == static_cast<size_t>(nvtxs)) {
    std::vector<double> Ax_direct(static_cast<size_t>(nvtxs), 0.0);
    std::vector<double> err_vec = vecCEM;
    std::vector<double> Ax_err(static_cast<size_t>(nvtxs), 0.0);

    const sparse_status_t ax_status =
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
                        vecCEM.data(), 0.0, Ax_direct.data());
    if (ax_status != SPARSE_STATUS_SUCCESS) {
      mkl_sparse_destroy(A);
      mkl_sparse_destroy(ACEM);
      throw std::runtime_error(
          "mkl_sparse_d_mv failed while evaluating CEM energy.");
    }

    const int incx = 1;
    const double norm_energy_direct =
        cblas_ddot(nvtxs, Ax_direct.data(), incx, vecCEM.data(), incx);

    cblas_daxpy(nvtxs, -1.0, vecSOL.data(), incx, err_vec.data(), incx);
    const sparse_status_t ax_err_status =
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr,
                        err_vec.data(), 0.0, Ax_err.data());
    if (ax_err_status != SPARSE_STATUS_SUCCESS) {
      mkl_sparse_destroy(A);
      mkl_sparse_destroy(ACEM);
      throw std::runtime_error(
          "mkl_sparse_d_mv failed while evaluating CEM energy residual.");
    }

    const double norm_energy_residual =
        cblas_ddot(nvtxs, Ax_err.data(), incx, err_vec.data(), incx);

    if (norm_energy_direct > 0.0 && norm_energy_residual >= 0.0) {
      std::cout << "Energy residual is: "
                << std::sqrt(norm_energy_residual) /
                       std::sqrt(norm_energy_direct)
                << std::endl;
    } else {
      std::cout << "Energy residual is unavailable because the reference "
                   "energy norm is non-positive."
                << std::endl;
    }

    const double norm_l2_direct = cblas_dnrm2(nvtxs, vecSOL.data(), incx);
    const double norm_l2_residual = cblas_dnrm2(nvtxs, err_vec.data(), incx);
    if (norm_l2_direct > 0.0) {
      std::cout << "L2 residual is: " << norm_l2_residual / norm_l2_direct
                << std::endl;
    } else {
      std::cout << "L2 residual is unavailable because ||u_direct|| is zero."
                << std::endl;
    }
  } else {
    std::cout << "Reference direct solution vecSOL is not available; "
                 "skip Energy/L2 residual report."
              << std::endl;
  }

  mkl_sparse_destroy(A);
  mkl_sparse_destroy(ACEM);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished solving CEM system with " << duration.count()
            << " ms======" << std::endl;
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

void CEM::solveFromLAndReleaseA() {
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Direct solve completed in " << duration.count() << " ms"
              << std::endl;
    std::cout << "Direct solve residual (abs): " << res_norm
              << ", residual (rel): " << rel_res << std::endl;
  } catch (...) {
    throw;
  }
}
