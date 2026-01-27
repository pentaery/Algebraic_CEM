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
  sys.formRHSPoisson2d();
  sys.formA();
  sys.solve();
  sys.graphPartition();
  // sys.testPoisson();
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