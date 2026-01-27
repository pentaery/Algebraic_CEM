# Algebraic CEM Solver - AI Coding Agent Instructions

## Project Overview
This is a C++ implementation of a Constraint Energy Minimization (CEM) solver that uses algebraic graph partitioning (METIS) instead of geometric partitioning. The solver is OpenMP-parallelized for auxiliary and CEM basis construction.

## Critical Dependencies
- **Intel MKL**: Math Kernel Library for sparse matrix operations and PARDISO direct solver
- **METIS**: Graph partitioning library for domain decomposition
- **GKLib**: METIS dependency (must be built first with GCC, not ICC)
- **OpenMP**: Parallel execution in `formAUX()` and `formCEM()` phases

## Architecture & Data Flow

### Main Pipeline (see [src/poisson2d.cc](src/poisson2d.cc))
```
System initialization → getDataPoisson2d() → formA() → solve() 
  → graphPartition() → findNeighbours() → formAUX() 
  → formCEM() → formMatR() → solveCEM()
```

### Core Components
- **System class** ([src/system.hh](src/system.hh)): Main solver state with sparse matrices (`matA`, `matL`, `matR`), vectors (`vecRHS`, `vecSOL`, `cemSOL`), and partition metadata
- **Graph partitioning**: METIS partitions the graph formed by `matL + matM`, stores results in `part[]` array and writes to `partition.txt`
- **Local-global indexing**: `globalTolocal` maps global vertex indices to local partition indices; `globalTolocalCEM` and `localtoGlobalCEM` handle overlapping regions
- **Overlapping regions**: Configurable overlap parameter extends partitions into neighboring domains

## Key Implementation Patterns

### Matrix Format Convention
- **COO → CSR conversion**: Always create matrices in COO format with `mkl_sparse_d_create_coo()`, then convert to CSR with `mkl_sparse_convert_csr()` for solver operations
- **CSR export**: Use `mkl_sparse_d_export_csr()` to access raw CSR arrays (`rows_start`, `col_index`, `val`)
- **Matrix symmetry**: Auxiliary space eigen problems use symmetric matrices (see `formAUX()` descriptor setup)

### Index Translation Pattern
```cpp
// Global to local (per partition):
localIdx = globalTolocal[globalIdx]

// Local to global (overlapping):
globalIdx = localtoGlobalCEM[partition][localIdx]

// Partition membership:
partition = part[globalIdx]
```

### Parallel Eigen Solver Loop
In `formAUX()`, eigen problems are solved per partition with `mkl_sparse_d_gv()`. The commented `#pragma omp parallel for` shows intended parallelization point (currently disabled).

### PARDISO Usage Pattern
1. Initialize `pt[64]`, `iparm[64]` to zero
2. Set `iparm[34] = 1` (0-based indexing), `iparm[0] = 1` (use defaults)
3. Call with `phase = 13` (analysis + factorization + solve)
4. Cleanup with `phase = -1`
5. Example in `solve()`, `formCEM()`, `solveCEM()`

## Build & Run Workflow

### Build Steps
```bash
mkdir -p build && cd build
cmake ..
make
```
Executable: `build/src/test`

### CMake Configuration Notes
- Hardcoded METIS path: `/home/ET/yjzhou/HPCSoft/METIS` in [src/CMakeLists.txt](src/CMakeLists.txt#L1)
- Links: `libmetis.so`, `libGKlib.a`, MKL dynamic libraries via pkg-config
- Debug build by default (`CMAKE_BUILD_TYPE Debug`)

### Runtime Behavior
- Reads matrix data from `data/{i,j,v,b}.txt` (or generates 2D Poisson problem)
- Writes partition to `partition.txt` in project root
- Outputs timing for each phase and error metrics (energy residual, L2 residual)

## Data Structures to Understand

### Partition Metadata
- `vertices[i]`: Global indices of all vertices in partition `i`
- `verticesCEM[i]`: Global indices including overlapping regions
- `neighbours[i]`: Adjacent partitions to partition `i`
- `overlapping[i]`: All partitions within overlap distance

### Basis Storage
- `eigenvector[i]`: k0 eigenvectors per partition (flattened: k0 × count[i])
- `cemBasis[i]`: CEM basis vectors in overlapping region (flattened: k0 × verticesCEM[i].size())
- `eigenvalue[i]`: k0 eigenvalues per partition

## Project-Specific Conventions
- **Matrix construction**: Build stiffness matrix `A` from Laplacian `L` by: diagonal += L_ij, off-diagonal -= L_ij
- **cStar parameter**: Scaling constant for auxiliary space stiffness (see `formAUX()` Si matrix construction)
- **k0**: Number of eigenvectors per partition (configurable via constructor)
- **File paths**: Many use relative paths `../../data/` assuming execution from `build/src/`

## Testing & Validation
- `testPoisson()`: Computes analytical solution error for 2D Poisson problem
- Error metrics: Energy norm and L2 norm residuals printed by `solveCEM()`
- No formal test suite; validation via console output

## Common Pitfalls
- **Path dependencies**: Hardcoded paths in CMakeLists.txt and data loading code
- **Index base**: MKL uses 0-based indexing with `iparm[34] = 1`; METIS uses 0-based by default
- **Memory management**: MKL sparse matrices must be explicitly destroyed with `mkl_sparse_destroy()`
- **Overlap calculation**: `findNeighbours()` iteratively expands overlapping set; loop count must match overlap parameter
