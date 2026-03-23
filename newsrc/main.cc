#include "cem.hh"
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
  const char *mesh_file = "../../mesh/triangle.msh";
  int order = 1;

  if (argc > 1) {
    mesh_file = argv[1];
  }
  if (argc > 2) {
    order = std::atoi(argv[2]);
  }

  CEM cem(50, 2, 3);
  cem.getDatafromMFEM(mesh_file, order);
  cem.graphPartition();
  cem.findNeighbours();
  cem.formAUX();
  // cem.exportNeighboursData();
  const double rel_res = cem.solveFromLAndReleaseA();
  if (rel_res >= 0.0) {
    std::cout << "Relative residual ||Ax-b||/||b|| = " << rel_res << std::endl;
  }

  return 0;
}
