#include "mkl_pardiso.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include "system.hh"

#include <cstdio>
#include <metis.h>
#include <vector>

int main() {
  auto start = std::chrono::high_resolution_clock::now();

  System sys(100, 100, 4, 4);
  sys.getDataPoisson2d();
  // get csr matrix matL(全矩阵)
  sys.formRHSPoisson2d();
  // get rhs vector vecRHS
  sys.formA();
  // form matrix matA from matL, and convert it to CSR format for
  // pardiso(上三角矩阵)
  sys.solve();
  // solve the system using pardiso and get the solution in vecSOL
  sys.graphPartition();
  // partition the graph to nparts parts using metis and get the partition
  // result in part sys.testPoisson();
  sys.findNeighbours();
  sys.formAUX();
  sys.formCEM();
  sys.formMatR();
  sys.solveCEM();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("Time elapsed: %f ms\n", duration.count());

  return 0;
}