#include <mpi.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include "utils/file_utils.hpp"


// Map 3D index to 1D
inline int index3D(int x, int y, int z, int ny, int nz) {
    return x * (ny * nz) + y * nz + z;
}
template <typename Type>
void readfile1(const char *file, const size_t num, Type *data) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cout << " Error, Couldn't find the file: " << file << "\n";
        exit(0);
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    assert(num_elements == num && "File size is not equals to the input setting");
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(data), num_elements * sizeof(Type));
    fin.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Define 3D grid dimensions (adjust for your case)
    int mpi_dims[3] = {0, 0, 0};
    mpi_dims[0] = atoi(argv[1]);
    mpi_dims[1] = atoi(argv[2]);
    mpi_dims[2] = atoi(argv[3]);

    std::string dir_prefix = argv[4];  // Directory prefix for the blocks
    std::string orig_prefix = argv[5];  // Name prefix for the blocks 
    std::string decomp_suffix = argv[6];  // Name suffix for the blocks

    int orig_dims[3] = {0, 0, 0};
    orig_dims[0] = atoi(argv[7]);
    orig_dims[1] = atoi(argv[8]);
    orig_dims[2] = atoi(argv[9]);

    int local_dims[3] = {0, 0, 0};
    local_dims[0] = orig_dims[0] / mpi_dims[0];
    local_dims[1] = orig_dims[1] / mpi_dims[1];
    local_dims[2] = orig_dims[2] / mpi_dims[2];
    

    // Create 3D Cartesian topology
    int periods[3] = {0, 0, 0};  // Non-periodic grid
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_dims, periods, 1, &cart_comm);

    // Get Cartesian coordinates of the current rank
    int mpi_coords[3];
    MPI_Cart_coords(cart_comm, mpi_rank, 3, mpi_coords);

    // Allocate window memory (block of the 3D grid)
    size_t local_size = local_dims[0] * local_dims[1] * local_dims[2];
    float* orig_data;
    float* decomp_data; 

    MPI_Win win_orig, win_decomp;

    MPI_Win_allocate(local_size * sizeof(float), sizeof(float), MPI_INFO_NULL, cart_comm, &orig_data, &win_orig);
    MPI_Win_allocate(local_size * sizeof(float), sizeof(float), MPI_INFO_NULL, cart_comm, &decomp_data, &win_decomp);

    // read data for each rank
    char filename[1024];
    sprintf(filename, "%s/%s_%d_%d_%d.f32", dir_prefix.c_str(), orig_prefix.c_str(), mpi_coords[0], mpi_coords[1], mpi_coords[2]);
    size_t num_elements = 0; 
    auto data = readfile<float>(filename, num_elements);
    assert(local_size == num_elements);
    std::copy(data.get(), data.get() + local_size, orig_data);

    sprintf(filename, "%s/%s_%d_%d_%d.decomp.f32", dir_prefix.c_str(), orig_prefix.c_str(), mpi_coords[0], mpi_coords[1], mpi_coords[2]);
    data = readfile<float>(filename, num_elements); 
    assert(local_size == num_elements);
    std::copy(data.get(), data.get() + local_size, decomp_data);
    

    MPI_Barrier(cart_comm);
    MPI_Win_fence(0, win_decomp);
    MPI_Win_fence(0, win_orig);



    MPI_Win_free(&win_orig);
    MPI_Win_free(&win_decomp);
    MPI_Finalize();
    return 0;
}
