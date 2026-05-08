#include "cem.hh"
#include <cstdlib>

int main(int argc, char *argv[]) {
  const char *mesh_file = "../../mesh/rect01.msh";
  int order = 1;

  if (argc > 1) {
    mesh_file = argv[1];
  }
  if (argc > 2) {
    order = std::atoi(argv[2]);
  }

  CEM cem(100, 1, 4);
  cem.getDatafromMFEM(mesh_file, order);
  cem.solveFromLAndReleaseA(1e-2, 100);
  // cem.graphPartition();
  // cem.findNeighbours();
  // cem.formAUX();
  // cem.exportNeighboursData();
  // cem.formCEM();
  // cem.formMatR();
  // cem.solveCEM();

  return 0;
}
