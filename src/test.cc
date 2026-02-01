#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Right-hand side function f
double f_rhs(const Vector &x) {
  // Example: f = 2*pi^2 * sin(pi*x) * sin(pi*y)
  // This corresponds to exact solution u = sin(pi*x) * sin(pi*y)
  double pi = M_PI;
  return 2.0 * pi * pi * sin(pi * x(0)) * sin(pi * x(1));
}

// Exact solution (for error computation)
double u_exact(const Vector &x) {
  double pi = M_PI;
  return sin(pi * x(0)) * sin(pi * x(1));
}

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file = "../../data/rect.msh";
  int order = 1; // Polynomial order of finite element space

  if (argc > 1) {
    mesh_file = argv[1];
  }
  if (argc > 2) {
    order = atoi(argv[2]);
  }

  // 2. Open the mesh file.
  ifstream imesh(mesh_file);
  if (!imesh) {
    cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
    return 1;
  }

  // 3. Read the mesh.
  Mesh mesh(imesh, 1, 1);
  imesh.close();

  int dim = mesh.Dimension();

  // 4. Print the mesh details.
  cout << "Mesh read successfully!" << endl;
  cout << "Dimension: " << dim << endl;
  cout << "Number of elements: " << mesh.GetNE() << endl;
  cout << "Number of vertices: " << mesh.GetNV() << endl;

  // 5. Define a finite element space on the mesh.
  //    H1 (continuous Lagrange) finite elements of the specified order.
  H1_FECollection fec(order, dim);
  FiniteElementSpace fespace(&mesh, &fec);

  cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
       << endl;

  // 6. Determine the list of true (i.e., conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet).
  Array<int> ess_tdof_list;
  if (mesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1; // Mark all boundary attributes as essential
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  // 7. Set up the linear form b(.) which corresponds to the right-hand side
  //    of the FEM linear system: b(v) = (f, v)
  LinearForm b(&fespace);
  FunctionCoefficient f_coef(f_rhs);
  b.AddDomainIntegrator(new DomainLFIntegrator(f_coef));
  b.Assemble();

  // 8. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  GridFunction x(&fespace);
  x = 0.0;

  // 9. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the
  //    Diffusion domain integrator.
  BilinearForm a(&fespace);
  ConstantCoefficient one(1.0);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  // 10. Form the linear system A X = B, applying any necessary transformations
  //     such as: eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, etc.
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  cout << "Size of linear system: " << A.Height() << endl;

  // 11. Convert A to CSR format (MFEM SparseMatrix stores CSR internally).
  //     The CSR arrays are: I (row offsets), J (column indices), Data (values).
  const int *I = A.GetI();
  const int *J = A.GetJ();
  const double *Data = A.GetData();

  cout << "CSR nnz: " << A.NumNonZeroElems() << endl;
  cout << "CSR arrays ready: I (size " << (A.Height() + 1) << "), J/Data (size "
       << A.NumNonZeroElems() << ")" << endl;

  return 0;
}
