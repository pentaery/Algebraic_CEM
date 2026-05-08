# Algebraic CEM Solver - AI Coding Agent Instructions

## Project Scope
This repository currently centers on a single MFEM-driven CEM implementation in [src/cem.hh](../src/cem.hh), [src/cem.cc](../src/cem.cc), and [src/main.cc](../src/main.cc).

Do not assume legacy files or APIs from older versions (for example old `System`/`poisson2d` pipelines). Treat `CEM` as the authoritative solver class.

## Current End-to-End Pipeline
The executable entry in [src/main.cc](../src/main.cc) follows this order:

1. `CEM cem(2000, 3, 4)`
2. `getDatafromMFEM(mesh_file, order)`
3. `solveFromLAndReleaseA()` (direct reference solve)
4. `graphPartition()`
5. `findNeighbours()`
6. `formAUX()`
7. `exportNeighboursData()`
8. `formCEM()`
9. `formMatR()`
10. `solveCEM()`

When changing call order, preserve required preconditions checked in each phase.

## Dependencies and Build Facts
- Intel MKL (sparse BLAS + PARDISO)
- METIS + GKlib
- MFEM (currently resolved via `MFEM_DIR` in [src/CMakeLists.txt](../src/CMakeLists.txt))
- OpenMP

Build target is `cem_mfem` (not the old `test` target).

Typical build flow:

```bash
mkdir -p build
cd build
cmake ..
make -j
```

## Solver Data Model (CEM Class)
Primary sparse handles:
- `matL`: Laplacian-like operator assembled from MFEM reduced system and stored as CSR.
- `matA`: optional fine-scale stiffness handle (destroyed/rebuilt when needed).
- `matR`: coarse restriction/prolongation operator assembled from CEM basis.

Primary vectors:
- `vecRHS`: reduced right-hand side from MFEM.
- `vecSOL`: direct fine-scale reference solution (from `solveFromLAndReleaseA`).
- `cemSOL`: coarse CEM solution.

Partition and overlap metadata:
- `part`
- `vertices`, `neighbours`, `overlapping`
- `globalTolocal`, `count`
- `verticesCEM`, `globalTolocalCEM`, `localtoGlobalCEM`
- `eigenvector`, `eigenvalue`, `cemBasis`

## Matrix Semantics and Conventions
`getDatafromMFEM()` assembles MFEM diffusion and stores only `L` in CSR. Fine-grid `A` is reconstructed later when needed.

The code uses this mapping:
- `A(i,i) = sum_j L(i,j)`
- `A(i,j) = -L(i,j)` for `i != j`

Important MKL conventions used throughout:
- 0-based indexing with `iparm[34] = 1`
- PARDISO solve via `phase = 13`, release via `phase = -1`
- `mkl_sparse_d_export_csr` is used repeatedly to access CSR arrays

## Phase Responsibilities
1. `graphPartition()`:
- Exports CSR from `matL`
- Runs `METIS_PartGraphKway`
- Writes partition ids to `../../partition.txt`

2. `findNeighbours()`:
- Builds core partition vertex sets
- Builds neighboring partitions and overlap expansion by `overlap`
- Builds all global/local index maps for both core and overlap regions

3. `formAUX()`:
- Builds local `Ai` and `Si` per partition
- Solves generalized eigenproblems with `mkl_sparse_d_gv`
- Stores first `k0` eigenpairs per partition

4. `formCEM()`:
- Precomputes per-node `sData`
- Builds local overlap systems and right-hand sides
- Solves local CEM basis systems via PARDISO

5. `formMatR()`:
- Assembles global coarse operator `R` from `cemBasis` and overlap mappings

6. `solveCEM()`:
- Reconstructs fine-grid `A` from `L`
- Forms coarse matrix `A_CEM = R A R^T`
- Solves coarse system with PARDISO
- Recovers fine-scale CEM solution and reports residuals

7. `solveFromLAndReleaseA()`:
- Direct reference solve on fine grid using upper-triangular CSR construction
- Reports absolute and relative residuals

## MFEM Problem Setup Details
`getDatafromMFEM()` currently uses:
- `f_rhs`: sinusoidal forcing
- diffusion function pointer default: `diffusion_coeff_inclusions`

Alternative diffusion fields are implemented and selectable in code:
- `diffusion_coeff_channel`
- `diffusion_coeff_multichannel_complex`

If you add new coefficient fields, keep them deterministic and cheap to evaluate.

## Parallelism Notes
OpenMP is active in:
- `formAUX()` partition eigen-solves
- `formCEM()` partition-local basis solves

The code currently uses `num_threads(80)` explicitly in pragmas. If adapting to different hardware, prefer a configurable thread policy.

## Runtime File Outputs
- `../../partition.txt` from `graphPartition()`
- `../../data/neighbours_*.txt` files from `exportNeighboursData()`

Paths are relative to runtime working directory (typically under build tree). Keep this in mind when moving executables or changing launch configs.

## Safety and Editing Guidance for Agents
- Preserve function precondition checks and detailed runtime error messages.
- Maintain explicit MKL resource ownership (`mkl_sparse_destroy` and `destroyIfAllocated`).
- Do not silently change index bases or PARDISO `iparm` defaults.
- Keep partition mapping structures consistent in size and ordering; many downstream phases assume exact alignment.
- Validate any phase-order changes against `main.cc` and per-function guards.

## Common Failure Modes
- Hardcoded external paths in [src/CMakeLists.txt](../src/CMakeLists.txt) (`MFEM_DIR`, METIS, GKlib).
- Running from unexpected working directory breaks `../../data` and `../../partition.txt` outputs.
- Inconsistent overlap/local maps can trigger bounds checks in `formCEM()` and `formMatR()`.
- Large `nparts` with fixed `k0` and high thread count can stress memory and solver robustness.
