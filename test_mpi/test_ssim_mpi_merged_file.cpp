#include <mpi.h>
#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "mpi/boundary.hpp"
#include "mpi/compensation.hpp"
#include "mpi/data_exchange.hpp"
#include "mpi/edt.hpp"
#include "mpi/mpi_datatype.hpp"
#include "mpi/stats.hpp"
#include "utils/file_utils.hpp"
#include "utils/qcat_ssim.hpp"

namespace SZ = SZ3;

int main(int argc, char** argv) {
    int mpi_rank, size;

    int dims[3] = {2, 2, 2};  // Let MPI decide the best grid dimensions
    dims[0] = atoi(argv[1]);
    dims[1] = atoi(argv[2]);
    dims[2] = atoi(argv[3]);
    std::string orig_file = argv[4];     // Directory prefix for the blocks
    std::string decomp_filename = argv[5];    // Name prefix for the blocks

    int orig_dims[3] = {256, 384, 384};
    orig_dims[0] = atoi(argv[6]);
    orig_dims[1] = atoi(argv[7]);
    orig_dims[2] = atoi(argv[8]);

    int block_dims[3] = {0, 0, 0};
    block_dims[0] = orig_dims[0] / dims[0];
    block_dims[1] = orig_dims[1] / dims[1];
    block_dims[2] = orig_dims[2] / dims[2];
    size_t block_size = block_dims[0] * block_dims[1] * block_dims[2];

    size_t global_strides[3] = {orig_dims[2] * orig_dims[1], orig_dims[2], 1};
    size_t local_strides[3] = {block_dims[2] * block_dims[1], block_dims[2], 1};

    int periods[3] = {0, 0, 0};  // No periodicity in any dimension
    int coords[3] = {0, 0, 0};   // Coords of this process in the grid
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Dims_create(size, 3, dims);
    if (mpi_rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
        printf("Block dimensions: (%d, %d, %d)\n", block_dims[0], block_dims[1], block_dims[2]);
    }
    // Create a 3D Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    // Get coordinates of this process
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);

    // sprintf(filename, "%s/write_%d_%d_%d.f33", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // use MPI_File_write_at to write the data to the file
    MPI_File fh;
    MPI_File fh_decomp;
    size_t global_offset_start = coords[0] * block_dims[0] * global_strides[0] +
                                 coords[1] * block_dims[1] * global_strides[1] +
                                 coords[2] * block_dims[2] * global_strides[2];

    // prepare the a local buffer to read a larger size of the data,
    int ssim_win_size = 7;
    int ssim_win_shift = 2;
    int w_block_dims[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        if (coords[i] != dims[0] - 1) {
            w_block_dims[i] = block_dims[i] + ssim_win_size - ssim_win_shift;
        } else {
            w_block_dims[i] = block_dims[i];
        }
    }
    size_t w_block_size = w_block_dims[0] * w_block_dims[1] * w_block_dims[2];
    size_t w_local_strides[3] = {w_block_dims[2] * w_block_dims[1], w_block_dims[2], 1}; 
    std::vector<float> w_orig_data(w_block_size, 0);
    std::vector<float> w_decomp_data(w_block_size, 0); 
    // read the data from the original merged and decompressed files 
    MPI_File_open(MPI_COMM_WORLD, orig_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_open(MPI_COMM_WORLD, decomp_filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_decomp);
    for (int bz = 0; bz < w_block_dims[0]; bz++) {
        for (int by = 0; by < w_block_dims[1]; by++) {
            size_t local_offset = bz * w_local_strides[0] + by * w_local_strides[1];
            size_t cur_global_offset = global_offset_start + bz * global_strides[0] + by * global_strides[1];
            size_t row_offset = cur_global_offset * sizeof(float);
            MPI_File_read_at(fh, row_offset, &w_orig_data[local_offset], w_block_dims[2], MPI_FLOAT,
                             MPI_STATUS_IGNORE);
            MPI_File_read_at(fh_decomp, row_offset, &w_decomp_data[local_offset], w_block_dims[2], MPI_FLOAT,
                             MPI_STATUS_IGNORE);
        }
    }
    double local_ssim_sum =0;
    double local_nw = 0; 
    size_t max_offset2 = w_block_dims[0] - ssim_win_size;
    size_t max_offset1 = w_block_dims[1] - ssim_win_size;
    size_t max_offset0 = w_block_dims[2] - ssim_win_size;

    double time = MPI_Wtime(); 
    for (size_t offset2 = 0; offset2 <= max_offset2; offset2 += ssim_win_shift) {
        for (size_t offset1 = 0; offset1 <= max_offset1; offset1 += ssim_win_shift) {
            for (size_t offset0 = 0; offset0 <= max_offset0; offset0 += ssim_win_shift) {
                local_nw++;
                local_ssim_sum += PM::SSIM_3d_calcWindow(w_orig_data.data(), w_decomp_data.data(), 
                                                w_block_dims[1], w_block_dims[2],
                                                offset0, offset1, offset2, 
                                                ssim_win_size, ssim_win_size,
                                                        ssim_win_size);
            }
        }
    }
    double global_ssim_sum = 0;
    double global_nw = 0;
    MPI_Reduce(&local_ssim_sum, &global_ssim_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_nw, &global_nw, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    if (mpi_rank == 0) {
        printf("SSIM: %f\n", global_ssim_sum / global_nw);
    }
    double time_end = MPI_Wtime();
    if (mpi_rank == 0) {
        printf("Time: %f\n", time_end - time);
    }
    // free the data
    MPI_File_close(&fh);
    MPI_File_close(&fh_decomp);


    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
    return 0;
}