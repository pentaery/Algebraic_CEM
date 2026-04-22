#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include <algorithm>
#include <bits/types/clock_t.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <metis.h>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class System {
public:
  sparse_matrix_t matA;
  sparse_matrix_t matL;
  sparse_matrix_t matR; // A flat matrix
  std::vector<double> vecRHS;
  std::vector<double> vecSOL;
  std::vector<double> cemSOL;
  std::vector<double> matM;
  sparse_index_base_t indexing;
  MKL_INT rows, cols;

  double *val;
  MKL_INT size;
  MKL_INT nvtxs;
  idx_t nparts;
  idx_t *part;

  std::vector<int> count; // Record the size of each coarse element
  std::vector<std::set<idx_t>>
      vertices; // Global index of all vertices in a coarse element
  std::vector<std::unordered_set<idx_t>>
      verticesCEM; // To identify if two vertices are in the same overlapping
                   // region
  std::vector<idx_t> globalTolocal; // A map(vector) from global index of all
                                    // vertices in a coarse element to its local
                                    // index (in an ascending order)
  std::vector<std::unordered_map<idx_t, idx_t>>
      globalTolocalCEM; // A map from global index of all vertices in a
                        // overlapping region to its local index (in an
                        // ascending order)
  std::vector<std::vector<idx_t>> localtoGlobalCEM;
  std::vector<std::set<idx_t>> neighbours;
  std::vector<std::set<idx_t>> overlapping;

  std::vector<std::vector<double>>
      eigenvector; // Eigenvector of each coarse elemet, size of k0*count[i]
  std::vector<std::vector<double>> eigenvalue;
  std::vector<std::vector<double>> cemBasis;

  int overlap;
  double cStar;
  int k0;

public:
  void getDataPoisson2d();
  void getData();
  void getDatafromMFEM(const char *mesh_file);
  void formRHSPoisson2d();
  void formRHS();
  void testPoisson();
  void formA();
  void solve();
  void graphPartition();
  void graphPartitionPoisson();
  void findNeighbours();
  void formAUX();
  void formCEM();
  void formCEM2();
  void formMatR();
  void solveCEM();
  System();
  System(MKL_INT size);
  System(MKL_INT size, idx_t nparts);
  System(MKL_INT size, idx_t nparts, int overlap);
  System(MKL_INT size, idx_t nparts, int overlap, int k0);
  System(MKL_INT size, MKL_INT nvtxs, idx_t nparts, int overlap, int k0);
  ~System();
};
