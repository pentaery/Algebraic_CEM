#include "mfem.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <metis.h>
#include <vector>

// Right-hand side function f
double f_rhs(const mfem::Vector &x) {
  // Example: f = 2*pi^2 * sin(pi*x) * sin(pi*y)
  // This corresponds to exact solution u = sin(pi*x) * sin(pi*y)
  double pi = M_PI;
  return 2.0 * pi * pi * std::sin(pi * x(0)) * std::sin(pi * x(1));
}

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file = "../../data/rect.msh";
  int order = 1; // Polynomial order of finite element space

  if (argc > 1) {
    mesh_file = argv[1];
  }
  if (argc > 2) {
    order = std::atoi(argv[2]);
  }

    // 2. Open the mesh file.
  std::ifstream imesh(mesh_file);
  if (!imesh) {
    std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
    return 1;
  }

  // 3. Read the mesh.
  mfem::Mesh mesh(imesh, 1, 1);
  imesh.close();

  int dim = mesh.Dimension();

  // 4. Print the mesh details.
  std::cout << "Mesh read successfully!" << std::endl;
  std::cout << "Dimension: " << dim << std::endl;
  std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
  std::cout << "Number of vertices: " << mesh.GetNV() << std::endl;

  // 5. Define a finite element space on the mesh.
  //    H1 (continuous Lagrange) finite elements of the specified order.
  mfem::H1_FECollection fec(order, dim);
  mfem::FiniteElementSpace fespace(&mesh, &fec);

  std::cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
            << std::endl;

  // 6. Determine the list of true (i.e., conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet).
  mfem::Array<int> ess_tdof_list;
  if (mesh.bdr_attributes.Size()) {
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1; // Mark all boundary attributes as essential
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  // 7. Set up the linear form b(.) which corresponds to the right-hand side
  //    of the FEM linear system: b(v) = (f, v)
  mfem::LinearForm b(&fespace);
  mfem::FunctionCoefficient f_coef(f_rhs);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(f_coef));
  b.Assemble();

  // 8. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  mfem::GridFunction x(&fespace);
  x = 0.0;

  // 9. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the
  //    Diffusion domain integrator.
  mfem::BilinearForm a(&fespace);
  mfem::ConstantCoefficient one(1.0);
  a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
  a.Assemble();

  // 10. Form the linear system A X = B, applying any necessary transformations
  //     such as: eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, etc.
  mfem::SparseMatrix A;
  mfem::Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  std::cout << "Size of linear system: " << A.Height() << std::endl;

  // 11. Convert A to CSR format (MFEM SparseMatrix stores CSR internally).
  //     The CSR arrays are: I (row offsets), J (column indices), Data (values).
  int *I = A.GetI();
  int *J = A.GetJ();
  double *Data = A.GetData();

  std::cout << "CSR nnz: " << A.NumNonZeroElems() << std::endl;
  std::cout << "CSR arrays ready: I (size " << (A.Height() + 1)
            << "), J/Data (size " << A.NumNonZeroElems() << ")" << std::endl;
  // Check that I[0] == 0
  if (I[0] != 0) {
    std::cerr << "Error: CSR row offsets I do not start at 0." << std::endl;
    return 1;
  }

  // 11.5. Remove diagonal entries from A (set to zero) before partitioning.
  for (int row = 0; row < A.Height(); ++row) {
    for (int idx = I[row]; idx < I[row + 1]; ++idx) {
      if (J[idx] == row) {
        Data[idx] = 0.0;
      }
    }
  }

  // 12. Graph partitioning based on A (CSR). Use only off-diagonal entries
  //     to build the adjacency structure for METIS.
  const int n = A.Height();
  int num_parts = 100;

  // Use CSR I/J directly for METIS (idx_t is compatible with int in METIS).

  idx_t nvtxs = static_cast<idx_t>(n);
  idx_t ncon = 1;
  idx_t objval = 0;
  std::vector<idx_t> part(nvtxs, 0);

  METIS_PartGraphKway(&nvtxs, &ncon, reinterpret_cast<idx_t *>(I),
                      reinterpret_cast<idx_t *>(J), NULL, NULL, NULL,
                      &num_parts, NULL, NULL, NULL, &objval, part.data());

  std::ofstream part_ofs("../../script/partition.txt");
  for (int i = 0; i < n; ++i) {
    part_ofs << part[i] << ' ';
  }
  part_ofs.close();

  std::cout << "METIS objective: " << objval << std::endl;
  std::cout << "Partition saved to 'partition.txt'" << std::endl;

  return 0;
}
