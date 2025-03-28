add_executable(test_edt_mpi test_edt_mpi.cpp)
target_link_libraries(test_edt_mpi PRIVATE MPI::MPI_CXX)
add_executable(test_quantizer_mpi test_quantizer_mpi.cpp)
target_link_libraries(test_quantizer_mpi PRIVATE MPI::MPI_CXX)
add_executable(test_boundary_mpi test_boundary_mpi.cpp)
target_link_libraries(test_boundary_mpi PRIVATE MPI::MPI_CXX)

add_executable(test_edt_mpi_function test_edt_mpi_function.cpp)
target_link_libraries(test_edt_mpi_function PRIVATE MPI::MPI_CXX)

add_executable(test_ssim_mpi test_ssim_mpi.cpp)
target_link_libraries(test_ssim_mpi PRIVATE MPI::MPI_CXX)

add_executable(test_ssim_segmentation test_ssim_segmentation.cpp)
target_link_libraries(test_ssim_segmentation PRIVATE MPI::MPI_CXX)