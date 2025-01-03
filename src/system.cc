#include "system.hh"
#include "mkl_cblas.h"
#include "mkl_spblas.h"
#include <cstdio>
#include <iostream>

#include <ostream>
#include <vector>

void System::formRHSPoisson2d() {
  vecRHS.resize(nvtxs);
  double gridLength = 1.0 / (size + 1);
  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      vecRHS[row * size + col] =
          2 * M_PI * M_PI * sin(M_PI * (row + 1) * gridLength) *
          sin(M_PI * (col + 1) * gridLength) * gridLength * gridLength;
    }
  }
}

void System::formRHS() {
  vecRHS.resize(nvtxs);
  for (int i = 0; i < nvtxs; ++i) {
    vecRHS[i] = 1.0;
  }
}

void System::testPoisson() {
  double gridLength = 1.0 / (size + 1);
  int incx = 1;
  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      vecRHS[row * size + col] = sin(M_PI * (row + 1) * gridLength) *
                                 sin(M_PI * (col + 1) * gridLength);
    }
  }
  double norm1 = cblas_dnrm2(nvtxs, vecRHS.data(), incx);
  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      vecSOL[row * size + col] -= sin(M_PI * (row + 1) * gridLength) *
                                  sin(M_PI * (col + 1) * gridLength);
    }
  }
  double norm2 = cblas_dnrm2(nvtxs, vecSOL.data(), incx);
  std::cout << "Norm of the SOL: " << norm2 / norm1 << std::endl;
}

void System::getDataPoisson2d() {
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;
  values.reserve(5 * size * size);
  for (MKL_INT row = 0; row < size; ++row) {
    for (MKL_INT col = 0; col < size; ++col) {
      MKL_INT index = row * size + col;
      if (col + row == 0 || col + row == 2 * size - 2) {
        row_indx.push_back(index);
        col_indx.push_back(index);
        values.push_back(2.0);
      } else if (col == 0 || col == size - 1 || row == 0 || row == size - 1) {
        row_indx.push_back(index);
        col_indx.push_back(index);
        values.push_back(1.0);
      } else {
        row_indx.push_back(index);
        col_indx.push_back(index);
        values.push_back(0.0);
      }
      if (row > 0) {
        row_indx.push_back(index);
        col_indx.push_back((row - 1) * size + col);
        values.push_back(1.0);
      }
      if (row < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back((row + 1) * size + col);
        values.push_back(1.0);
      }
      if (col > 0) {
        row_indx.push_back(index);
        col_indx.push_back(index - 1);
        values.push_back(1.0);
      }
      if (col < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back(index + 1);
        values.push_back(1.0);
      }
    }
  }
  std::cout << "non-zero elements in L and U: " << values.size() << std::endl;
  // int i;
  // for (i = 0; i < values.size(); ++i) {
  //   std::cout << row_indx[i] << " " << col_indx[i] << " " << values[i]
  //             << std::endl;
  // }
  sparse_matrix_t matB;
  mkl_sparse_d_create_coo(&matB, indexing, nvtxs, nvtxs, values.size(),
                          row_indx.data(), col_indx.data(), values.data());
  mkl_sparse_convert_csr(matB, SPARSE_OPERATION_NON_TRANSPOSE, &matL);
  mkl_sparse_destroy(matB);
}

void System::getData() {
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;



  std::string filename = "../../i.txt";
  std::ifstream infile(filename);
  int index;
  double number;
  while (infile >> index) {
    row_indx.push_back(index);
  }
  infile.close();



  filename = "../../j.txt";
  infile.open(filename);
  while (infile >> index) {
    col_indx.push_back(index);
  }
  infile.close();

  filename = "../../v.txt";
  infile.open(filename);
  while (infile >> number) {
    values.push_back(number);
  }
  infile.close();

  // int i = 0;

  // for (i = 0; i < row_indx.size(); ++i) {
  //   std::cout << row_indx[i] << " ";
  // }

  // std::cout << std::endl;

  // for (i = 0; i < col_indx.size(); ++i) {
  //   std::cout << col_indx[i] << " ";
  // }

  // std::cout << std::endl;

  // for (i = 0; i < values.size(); ++i) {
  //   std::cout << values[i] << " ";
  // }

  // std::cout << std::endl;

  std::cout << "non-zero elements in L and U: " << values.size() << std::endl;
  // int i;
  // for (i = 0; i < values.size(); ++i) {
  //   std::cout << row_indx[i] << " " << col_indx[i] << " " << values[i]
  //             << std::endl;
  // }
  sparse_matrix_t matB;
  mkl_sparse_d_create_coo(&matB, indexing, nvtxs, nvtxs, values.size(),
                          row_indx.data(), col_indx.data(), values.data());
  mkl_sparse_convert_csr(matB, SPARSE_OPERATION_NON_TRANSPOSE, &matL);
  mkl_sparse_destroy(matB);
}

void System::graphPartition() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase I: Graph Partitioning======" << std::endl;
  MKL_INT *rows_start, *rows_end, *col_index;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);


  idx_t ncon = 1;
  idx_t objval;
  idx_t options[METIS_NOPTIONS];
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;
  METIS_PartGraphKway(&nvtxs, &ncon, rows_start, col_index, NULL, NULL, NULL,
                      &nparts, NULL, NULL, NULL, &objval, part);
  std::cout << "Objective for the partition is " << objval << std::endl;

  FILE *fp;
  if ((fp = fopen("../../partition.txt", "wb")) == NULL) {
    printf("cant open the file");
    exit(0);
  }
  for (int i = 0; i < nvtxs; i++) {
    fprintf(fp, "%d ", part[i]);
  }
  fclose(fp);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished with " << duration.count()
            << " ms======" << std::endl;
}

void System::formA() {
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  // for (int i = 0; i < 484; ++i) {
  //   std::cout << rows_start[i] << " ";
  // }
  // std::cout << std::endl;

  std::vector<MKL_INT> A_row_index;
  std::vector<MKL_INT> A_col_index;
  std::vector<double> A_values;
  int i = 0, j = 0;
  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (col_index[j] < i) {
        A_row_index.push_back(i);
        A_col_index.push_back(i);
        A_values.push_back(val[j]);
      } else if (col_index[j] > i) {
        A_row_index.push_back(i);
        A_col_index.push_back(i);
        A_values.push_back(val[j]);
        A_row_index.push_back(i);
        A_col_index.push_back(col_index[j]);
        A_values.push_back(-val[j]);
      } else {
        A_row_index.push_back(i);
        A_col_index.push_back(i);
        A_values.push_back(val[j]);
      }
    }
  }
  sparse_matrix_t matB;
  mkl_sparse_d_create_coo(&matB, indexing, nvtxs, nvtxs, A_values.size(),
                          A_row_index.data(), A_col_index.data(),
                          A_values.data());
  mkl_sparse_convert_csr(matB, SPARSE_OPERATION_NON_TRANSPOSE, &matA);
  mkl_sparse_destroy(matB);
}

void System::solve() {
  auto start = std::chrono::high_resolution_clock::now();
  MKL_INT *rows_start, *rows_end, *col_index;
  mkl_sparse_d_export_csr(matA, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);

  MKL_INT perm[64], iparm[64];
  void *pt[64];
  MKL_INT error;
  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT nrhs = 1, msglv1 = 0;
  int i;
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;
  iparm[1] = 3;
  vecSOL.resize(nvtxs);
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, val, rows_start,
          col_index, perm, &nrhs, iparm, &msglv1, vecRHS.data(), vecSOL.data(),
          &error);
  if (error != 0) {
    std::cout << "error in pardiso: " << error << std::endl;
  }
  phase = -1;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, val, rows_start,
          col_index, perm, &nrhs, iparm, &msglv1, vecRHS.data(), vecSOL.data(),
          &error);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Phase 0: Direct solve with " << duration.count()
            << " ms======" << std::endl;
}

void System::findNeighbours() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout
      << "======Phase II: Construct the neighours for the CEM method======"
      << std::endl;
  int i = 0, j = 0;
  vertices.resize(nparts);
  for (i = 0; i < nvtxs; ++i) {
    vertices[part[i]].insert(i);
  }

  neighbours.resize(nparts);
  MKL_INT *rows_start, *rows_end, *col_index;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (part[i] != part[col_index[j]]) {
        neighbours[part[i]].insert(part[col_index[j]]);
      }
    }
  }
  overlapping.resize(nparts);
  if (overlap > 0) {
    for (j = 0; j < nparts; ++j) {
      overlapping[j].insert(neighbours[j].begin(), neighbours[j].end());
      overlapping[j].insert(j);
    }
  }

  for (i = 1; i < overlap; ++i) {
    for (j = 0; j < nparts; ++j) {
      std::set<MKL_INT> tempset;
      for (const auto &element : overlapping[j]) {
        tempset.insert(neighbours[element].begin(), neighbours[element].end());
      }
      overlapping[j].insert(tempset.begin(), tempset.end());
    }
  }

  globalTolocal.resize(nvtxs);
  count.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    count[i] = 0;
  }
  for (i = 0; i < nvtxs; ++i) {
    globalTolocal[i] = count[part[i]];
    count[part[i]]++;
  }

  verticesCEM.resize(nparts);
  globalTolocalCEM.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    j = 0;
    for (const auto &element : overlapping[i]) {
      verticesCEM[i].insert(vertices[element].begin(), vertices[element].end());
      for (const auto &element2 : vertices[element]) {
        globalTolocalCEM[i].insert({element2, j++});
      }
    }
  }

  localtoGlobalCEM.resize(nparts);
  int index = 0;
  for (i = 0; i < nparts; ++i) {
    index = 0;
    localtoGlobalCEM[i].resize(verticesCEM[i].size());
    for (const auto &element : overlapping[i]) {
      for (const auto &element2 : vertices[element]) {
        localtoGlobalCEM[i][index++] = element2;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished with " << duration.count()
            << " ms======" << std::endl;
}

void System::formAUX() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase III: Construct the Auxiliary space======"
            << std::endl;
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  int i = 0, j = 0;

  char which = 'S';
  MKL_INT pm[128];

  mkl_sparse_ee_init(pm);
  pm[7] = 1;
  pm[8] = 1;

  std::vector<std::vector<MKL_INT>> Ai_col_index;
  std::vector<std::vector<MKL_INT>> Ai_row_index;
  std::vector<std::vector<double>> Ai_values;
  std::vector<std::vector<MKL_INT>> Si_col_index;
  std::vector<std::vector<MKL_INT>> Si_row_index;
  std::vector<std::vector<double>> Si_values;
  Ai_col_index.resize(nparts);
  Ai_row_index.resize(nparts);
  Ai_values.resize(nparts);
  Si_col_index.resize(nparts);
  Si_row_index.resize(nparts);
  Si_values.resize(nparts);

  // for (i = 0; i < nparts; ++i) {
  //   Ai_col_index[i].reserve(nvtxs);
  //   Ai_row_index[i].reserve(nvtxs);
  //   Ai_values[i].reserve(nvtxs);
  //   Si_col_index[i].reserve(nvtxs);
  //   Si_row_index[i].reserve(nvtxs);
  //   Si_values[i].reserve(nvtxs);
  // }

  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (part[i] == part[col_index[j]]) {
        if (col_index[j] != i) {
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[i]);
          Ai_values[part[i]].push_back(val[j]);
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[col_index[j]]);
          Ai_values[part[i]].push_back(-val[j]);

          Si_col_index[part[i]].push_back(globalTolocal[i]);
          Si_row_index[part[i]].push_back(globalTolocal[i]);
          Si_values[part[i]].push_back(val[j] / cStar / cStar / 2);
        } else {
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[i]);
          Ai_values[part[i]].push_back(val[j]);

          Si_col_index[part[i]].push_back(globalTolocal[i]);
          Si_row_index[part[i]].push_back(globalTolocal[i]);
          Si_values[part[i]].push_back(val[j] / cStar / cStar);
        }
      }
    }
  }

  std::vector<sparse_matrix_t> AiCOO;
  std::vector<sparse_matrix_t> SiCOO;
  std::vector<sparse_matrix_t> Ai;
  std::vector<sparse_matrix_t> Si;

  AiCOO.resize(nparts);
  SiCOO.resize(nparts);
  Ai.resize(nparts);
  Si.resize(nparts);

  int k;

  eigenvalue.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    eigenvalue[i].resize(k0);
  }

  eigenvector.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    eigenvector[i].resize(k0 * count[i]);
  }

  double res[nparts];

  matrix_descr descr;

  descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  descr.mode = SPARSE_FILL_MODE_UPPER;

  // #pragma omp parallel for
  for (i = 0; i < nparts; ++i) {
    //   for (j = 0; j < Ai_values[i].size(); ++j) {
    //     std::cout<< Ai_col_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for(j = 0; j < 11; ++j) {
    //     std::cout << Ai_row_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for(j = 0; j < 11; ++j) {
    //     std::cout << Ai_values[i][j] << "  ";
    //   }
    //   std::cout << std::endl;

    //   for (j = 0; j < Si_values[i].size(); ++j) {
    //     std::cout<< Si_col_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for (j = 0; j < Si_values[i].size(); ++j) {
    //     std::cout << Si_row_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for (j = 0; j < Si_values[i].size(); ++j) {
    //     std::cout << Si_values[i][j] << "  ";
    //   }
    //   std::cout << std::endl;

    mkl_sparse_d_create_coo(&AiCOO[i], indexing, count[i], count[i],
                            Ai_values[i].size(), Ai_row_index[i].data(),
                            Ai_col_index[i].data(), Ai_values[i].data());
    mkl_sparse_d_create_coo(&SiCOO[i], indexing, count[i], count[i],
                            Si_values[i].size(), Si_row_index[i].data(),
                            Si_col_index[i].data(), Si_values[i].data());



    mkl_sparse_convert_csr(AiCOO[i], SPARSE_OPERATION_NON_TRANSPOSE, &Ai[i]);
    mkl_sparse_convert_csr(SiCOO[i], SPARSE_OPERATION_NON_TRANSPOSE, &Si[i]);

    mkl_sparse_destroy(AiCOO[i]);
    mkl_sparse_destroy(SiCOO[i]);

    sparse_status_t error =
        mkl_sparse_d_gv(&which, pm, Ai[i], descr, Si[i], descr, k0, &k,
                        eigenvalue[i].data(), eigenvector[i].data(), &res[i]);
    if (error != 0) {
      std::cout << "======error in " << error << "===============" << std::endl;
    }
    if (k < k0) {
      std::cout << "===========Not find enough eigenvalues==========="
                << std::endl;
    }
    std::cout << "part: " << i << " residual: " << res[i]
              << " Smallest eigenvalue: " << eigenvalue[i][0] << std::endl;

    mkl_sparse_destroy(Ai[i]);
    mkl_sparse_destroy(Si[i]);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "======Finish solving eigen problem in each coarse element with "
            << duration.count() << " ms======" << std::endl;
}

void System::formCEM() {
  auto start = std::chrono::high_resolution_clock::now();
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  int i = 0, j = 0, k = 0;

  std::vector<double> sData(nvtxs, 0.0);
  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (col_index[j] == i) {
        sData[i] += val[j] / cStar / cStar;
      } else {
        sData[i] += val[j] / cStar / cStar / 2;
      }
    }
  }

  std::vector<std::vector<double>> sMatrix;
  sMatrix.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    sMatrix[i].resize(vertices[i].size() * vertices[i].size());
    for (const auto &element1 : vertices[i]) {
      for (const auto &element2 : vertices[i]) {
        sMatrix[i][globalTolocal[element1] * vertices[i].size() +
                   globalTolocal[element2]] = 0.0;
        for (j = 0; j < k0; ++j) {
          sMatrix[i][globalTolocal[element1] * vertices[i].size() +
                     globalTolocal[element2]] +=
              eigenvector[i][j * vertices[i].size() + globalTolocal[element1]] *
              eigenvector[i][j * vertices[i].size() + globalTolocal[element2]] *
              sData[element1] * sData[element2];
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finish forming all S_i in " << duration.count()
            << " ms=======" << std::endl;
  // Calculating CEM Basis in each overlapping area
  std::cout
      << "======Start calculating CEM Basis in each overlapping area======"
      << std::endl;
  #pragma omp parallel for
  for (i = 0; i < nparts; ++i) {
    std::vector<MKL_INT> Ai_col_index(2 * verticesCEM[i].size() *
                                          verticesCEM[i].size() /
                                          overlapping[i].size(),
                                      0);
    std::vector<MKL_INT> Ai_row_index(2 * verticesCEM[i].size() *
                                          verticesCEM[i].size() /
                                          overlapping[i].size(),
                                      0);
    std::vector<double> Ai_values(2 * verticesCEM[i].size() *
                                      verticesCEM[i].size() /
                                      overlapping[i].size(),
                                  0);
    int index1 = 0;
    int index2 = 0;
    int index3 = 0;

    // Forming the matrix A_i
    for (const auto &element : verticesCEM[i]) {
      // std::cout << "element: " << element
      //           << " index: " << globalTolocalCEM[i][element] << std::endl;
      for (j = rows_start[element]; j < rows_end[element]; ++j) {
        if (verticesCEM[i].count(col_index[j]) == 1) {
          // std::cout << " element :" << col_index[j];
          if (globalTolocalCEM[i][col_index[j]] <=
              globalTolocalCEM[i][element]) {
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][element]);
            Ai_values[index3++] = val[j];
            // std::cout << " " << val[j];
          } else {
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][element]);
            Ai_values[index3++] = val[j];
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][col_index[j]]);
            Ai_values[index3++] = -val[j];
            // std::cout << " " << val[j];
          }
        } else {
          Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
          Ai_col_index[index2++] = (globalTolocalCEM[i][element]);
          Ai_values[index3++] = val[j];
        }
      }
      // std::cout << std::endl;
    }

    // Forming the matrix S_i
    for (const auto &element : overlapping[i]) {
      for (const auto &element1 : vertices[element]) {
        for (const auto &element2 : vertices[element]) {
          if (globalTolocalCEM[i][element1] <= globalTolocalCEM[i][element2]) {
            Ai_row_index[index1++] = globalTolocalCEM[i][element1];
            Ai_col_index[index2++] = globalTolocalCEM[i][element2];
            Ai_values[index3++] =
                sMatrix[element]
                       [globalTolocal[element1] * vertices[element].size() +
                        globalTolocal[element2]];
          }
        }
      }
    }

    Ai_row_index.resize(index1);
    Ai_col_index.resize(index2);
    Ai_values.resize(index3);

    std::vector<double> rhs(verticesCEM[i].size() * k0, 0.0);
    for (j = 0; j < k0; ++j) {
      for (const auto &element : vertices[i]) {
        rhs[j * verticesCEM[i].size() + globalTolocalCEM[i][element]] =
            sData[globalTolocalCEM[i][element]] *
            eigenvector[i][j * vertices[i].size() + globalTolocal[element]];
      }
    }

    sparse_matrix_t AiCOO;
    sparse_matrix_t Ai;
    // std::cout << "vertices size: " << verticesCEM[i].size() << std::endl;
    mkl_sparse_d_create_coo(&AiCOO, indexing, verticesCEM[i].size(),
                            verticesCEM[i].size(), Ai_values.size(),
                            Ai_row_index.data(), Ai_col_index.data(),
                            Ai_values.data());
    mkl_sparse_convert_csr(AiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Ai);

    MKL_INT *rows_start_new, *rows_end_new, *col_index_new;
    MKL_INT rows_new, cols_new;
    double *val_new;
    mkl_sparse_d_export_csr(Ai, &indexing, &rows_new, &cols_new,
                            &rows_start_new, &rows_end_new, &col_index_new,
                            &val_new);
    // for (j = 0; j < 80; ++j) {
    //   std::cout << rows_start_new[j] << " ";
    // }
    // std::cout << std::endl;
    // for (j = 0; j < 212; ++j) {
    //   std::cout << col_index_new[j] << " ";
    // }

    // std::cout << std::endl;
    // for (j = 0; j < 212; ++j) {
    //   std::cout << val_new[j] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Finish forming the matrix A_i and S_i in part " << i
    //           << std::endl;
    MKL_INT error;

    MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
    MKL_INT msglv1 = 0;

    MKL_INT idum;
    MKL_INT perm[64], iparm[64];
    void *pt[64];
    for (j = 0; j < 64; j++) {
      pt[j] = 0;
    }
    for (j = 0; j < 64; j++) {
      iparm[j] = 0;
    }
    iparm[34] = 1;
    iparm[0] = 1;
    cemBasis.resize(nparts);
    cemBasis[i].resize(verticesCEM[i].size() * k0);
    MKL_INT n = verticesCEM[i].size();

    // std::cout << "rhs size: " << rhs.size() << std::endl;
    // std::cout << "Matrix size: " << rows_new << " " << cols_new << std::endl;

    // std::cout << "Start solving the system in part " << i << std::endl;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val_new, rows_start_new,
            col_index_new, perm, &k0, iparm, &msglv1, rhs.data(),
            cemBasis[i].data(), &error);
    if (error != 0) {
      std::cout << "error in pardiso: " << error << std::endl;
    }
    phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val_new, rows_start_new,
            col_index_new, perm, &k0, iparm, &msglv1, rhs.data(),
            cemBasis[i].data(), &error);

    mkl_sparse_destroy(Ai);
    mkl_sparse_destroy(AiCOO);
  }
  std::cout << std::endl;
}

void System::formCEM2() {}

void System::formMatR() {
  auto start = std::chrono::high_resolution_clock::now();
  int index1 = 0, index2 = 0, index3 = 0;
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;

  row_indx.resize(k0 * nparts * nvtxs);
  col_indx.resize(k0 * nparts * nvtxs);
  values.resize(k0 * nparts * nvtxs);

  int i = 0, j = 0, k = 0;

  for (i = 0; i < nparts; ++i) {
    for (j = 0; j < k0; ++j) {
      for (k = 0; k < verticesCEM[i].size(); ++k) {
        row_indx[index1++] = i * k0 + j;
        col_indx[index2++] = localtoGlobalCEM[i][k];
        values[index3++] = cemBasis[i][j * verticesCEM[i].size() + k];
      }
    }
  }
  row_indx.resize(index1);
  col_indx.resize(index2);
  values.resize(index3);

  sparse_matrix_t matRCOO;
  mkl_sparse_d_create_coo(&matRCOO, indexing, k0 * nparts, nvtxs, values.size(),
                          row_indx.data(), col_indx.data(), values.data());
  mkl_sparse_convert_csr(matRCOO, SPARSE_OPERATION_NON_TRANSPOSE, &matR);
  mkl_sparse_destroy(matRCOO);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finish forming the matrix R in " << duration.count()
            << " ms======" << std::endl;
}

void System::solveCEM() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase IV: Solve the CEM problem======" << std::endl;
  int i = 0, j = 0, k = 0, l = 0;
  sparse_matrix_t ACEM;
  sparse_matrix_t A;
  sparse_matrix_t Acoo;

  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  std::vector<MKL_INT> A_row_index(nvtxs * 12, 0);
  std::vector<MKL_INT> A_col_index(nvtxs * 12, 0);
  std::vector<double> A_values(nvtxs * 12, 0);
  int index1 = 0, index2 = 0, index3 = 0;
  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      A_row_index[index1++] = i;
      A_col_index[index2++] = i;
      A_values[index3++] = val[j];
      if (col_index[j] != i) {
        A_row_index[index1++] = i;
        A_col_index[index2++] = col_index[j];
        A_values[index3++] = -val[j];
      }
    }
  }
  A_row_index.resize(index1);
  A_col_index.resize(index2);
  A_values.resize(index3);

  mkl_sparse_d_create_coo(&Acoo, indexing, nvtxs, nvtxs, A_values.size(),
                          A_row_index.data(), A_col_index.data(),
                          A_values.data());
  mkl_sparse_convert_csr(Acoo, SPARSE_OPERATION_NON_TRANSPOSE, &A);
  mkl_sparse_destroy(Acoo);

  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end1 - start;
  std::cout << "======Finish forming the matrix A in " << duration.count()
            << " ms======" << std::endl;

  sparse_matrix_t RA;
  mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, matR, A, &RA);
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, RA,
                  SPARSE_OPERATION_TRANSPOSE, descr, matR,
                  SPARSE_STAGE_FULL_MULT, &ACEM);
  mkl_sparse_order(ACEM); // Important

  std::vector<double> cemRHS(nparts * k0, 0.0);
  mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matR, descr,
                  vecRHS.data(), 0.0, cemRHS.data());

  auto end2 = std::chrono::high_resolution_clock::now();
  duration = end2 - end1;
  std::cout << "======Finish forming the matrix A(CEM) and rhs(CEM) in "
            << duration.count() << " ms======" << std::endl;

  mkl_sparse_destroy(A);
  mkl_sparse_destroy(RA);

  MKL_INT *rows_start_new, *rows_end_new, *col_index_new;
  double *val_new;
  mkl_sparse_d_export_csr(ACEM, &indexing, &rows, &cols, &rows_start_new,
                          &rows_end_new, &col_index_new, &val_new);

  std::cout << "======In CEM we solve a system with " << rows << " rows and "
            << cols << " columns" << std::endl;
  MKL_INT perm[64], iparm[64];
  void *pt[64];
  MKL_INT error;
  MKL_INT maxfct = 1, mnum = 1, mtype = 11, phase = 13;
  MKL_INT nrhs = 1, msglv1 = 1;
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;
  iparm[1] = 3;
  iparm[26] = 1;
  cemSOL.resize(nparts * k0);
  int n = nparts * k0;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val_new, rows_start_new,
          col_index_new, perm, &nrhs, iparm, &msglv1, cemRHS.data(),
          cemSOL.data(), &error);
  if (error != 0) {
    std::cout << "error in pardiso: " << error << std::endl;
  }
  phase = -1;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val, rows_start, col_index,
          perm, &nrhs, iparm, &msglv1, cemRHS.data(), cemSOL.data(), &error);

  mkl_sparse_destroy(ACEM);

  int incx = 1;
  double normDirect = cblas_dnrm2(nvtxs, vecSOL.data(), incx);
  mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, matR, descr, cemSOL.data(),
                  -1.0, vecSOL.data());
  double normResidual = cblas_dnrm2(nvtxs, vecSOL.data(), incx);
  std::cout << "The relative residual is: " << normResidual / normDirect
            << std::endl;

  // index1 = 0;
  // index2 = 0;
  // index3 = 0;

  // int rows_start_y[1] = {0};
  // int rows_end_y[1] = {0};
  // int rows_start_x[1] = {0};
  // int rows_end_x[1] = {0};
  // for (int i : tq::trange(nparts)) {
  //   // std::cout << "Processing part " << i << std::endl;
  //   for (j = i; j < nparts; ++j) {
  //     std::set<idx_t> intersection;
  //     std::set_intersection(overlapping[i].begin(), overlapping[i].end(),
  //                           overlapping[j].begin(), overlapping[j].end(),
  //                           std::inserter(intersection,
  //                           intersection.begin()));
  //     if (!intersection.empty()) {
  //       // std::cout << "part " << i << " and part " << j
  //       //           << " have overlapping area" << std::endl;
  //       for (k = 0; k < k0; ++k) {
  //         sparse_matrix_t y;
  //         rows_end_y[0] = verticesCEM[i].size();
  //         mkl_sparse_d_create_csr(&y, indexing, 1, nvtxs, rows_start_y,
  //                                 rows_end_y, localtoGlobalCEM[i].data(),
  //                                 &cemBasis[i][k * verticesCEM[i].size()]);
  //         sparse_matrix_t Ay;
  //         mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, y, A, &Ay);
  //         for (l = k; l < k0; ++l) {
  //           A_row_index[index1++] = i * k0 + k;
  //           A_col_index[index2++] = j * k0 + l;
  //           sparse_matrix_t x;
  //           rows_end_x[0] = verticesCEM[j].size();
  //           mkl_sparse_d_create_csr(&x, indexing, 1, nvtxs, rows_start_x,
  //                                   rows_end_x, localtoGlobalCEM[j].data(),
  //                                   &cemBasis[j][l * verticesCEM[j].size()]);
  //           sparse_matrix_t xAy;
  //           mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, x,
  //                           SPARSE_OPERATION_TRANSPOSE, descr, Ay,
  //                           SPARSE_STAGE_FULL_MULT, &xAy);
  //           MKL_INT *rows_start, *rows_end, *col_index;
  //           MKL_INT rows, cols;
  //           sparse_index_base_t indexing;
  //           mkl_sparse_d_export_csr(xAy, &indexing, &rows, &cols,
  //           &rows_start,
  //                                   &rows_end, &col_index, &val);
  //           A_values[index3++] = val[0];
  //           // std::cout << "val: " << val[0] << std::endl;
  //           mkl_sparse_destroy(x);
  //           mkl_sparse_destroy(xAy);
  //         }
  //         mkl_sparse_destroy(y);
  //         mkl_sparse_destroy(Ay);
  //       }
  //     }
  //   }
  // }

  // A_row_index.resize(index1);
  // A_col_index.resize(index2);
  // A_values.resize(index3);
  // mkl_sparse_d_create_coo(&ACEMcoo, indexing, nparts * k0, nparts * k0,
  //                         A_values.size(), A_row_index.data(),
  //                         A_col_index.data(), A_values.data());
  // mkl_sparse_convert_csr(ACEMcoo, SPARSE_OPERATION_NON_TRANSPOSE, &ACEM);
  // mkl_sparse_destroy(ACEMcoo);

  // mkl_sparse_destroy(ACEM);
}

System::System() {
  size = 512;
  indexing = SPARSE_INDEX_BASE_ZERO;
  nvtxs = size * size;
  nparts = 10;
  part = new int[nvtxs];
  overlap = 2;
  cStar = 1.0 * size * size;
  k0 = 3;
}

System::System(MKL_INT size) {
  this->size = size;
  indexing = SPARSE_INDEX_BASE_ZERO;
  nvtxs = size * size;
  nparts = 10;
  part = new int[nvtxs];
  overlap = 2;
  cStar = 1.0 * size * size;
  k0 = 3;
}

System::System(MKL_INT size, idx_t nparts) {
  this->size = size;
  this->nparts = nparts;
  indexing = SPARSE_INDEX_BASE_ZERO;
  nvtxs = size * size;
  part = new int[nvtxs];
  overlap = 2;
  cStar = 1.0 * size * size;
  k0 = 3;
}

System::System(MKL_INT size, idx_t nparts, int overlap) {
  this->overlap = overlap;
  this->size = size;
  this->nparts = nparts;
  indexing = SPARSE_INDEX_BASE_ZERO;
  nvtxs = size * size;
  part = new int[nvtxs];
  cStar = 1.0 * size * size;
  k0 = 3;
}

System::System(MKL_INT size, idx_t nparts, int overlap, int k0) {
  this->k0 = k0;
  this->overlap = overlap;
  this->size = size;
  this->nparts = nparts;
  indexing = SPARSE_INDEX_BASE_ZERO;
  nvtxs = size * size;
  part = new int[nvtxs];
  cStar = 1.0 * size * size;
}

System::~System() {
  delete[] part;
  mkl_sparse_destroy(matA);
  mkl_sparse_destroy(matL);
  mkl_sparse_destroy(matR);
}
