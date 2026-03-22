set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MKL REQUIRED IMPORTED_TARGET mkl-dynamic-lp64-iomp)

set(MFEM_DIR "/home/ET/yjzhou/HPCSoft/mfem-4.9/build")
find_package(MFEM REQUIRED)

add_executable(cem_mfem main.cc cem.cc)

target_include_directories(cem_mfem PRIVATE ${MKL_INCLUDE_DIRS})
target_include_directories(cem_mfem PRIVATE ${MFEM_INCLUDE_DIRS})

target_link_libraries(cem_mfem PRIVATE ${MKL_LIBRARIES})
target_link_libraries(cem_mfem PRIVATE ${MFEM_LIBRARIES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(cem_mfem PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(cem_mfem PRIVATE MKL_ENABLE_OPENMP)
