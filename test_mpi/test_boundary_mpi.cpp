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
#include "mpi/stats.hpp"
#include "utils/file_utils.hpp"
#include "mpi/mpi_datatype.hpp" 

namespace SZ = SZ3;

int main(int argc, char** argv) {
    int mpi_rank, size;

    int dims[3] = {2, 2, 2};  // Let MPI decide the best grid dimensions
    dims[0] = atoi(argv[1]);
    dims[1] = atoi(argv[2]);
    dims[2] = atoi(argv[3]);
    double rel_eb = atof(argv[4]);     // Relative error bound
    std::string dir_prefix = argv[5];  // Directory prefix for the blocks
    std::string name_prefix = argv[6];  // Name prefix for the blocks 

    int orig_dims[3] = {256, 384, 384};
    orig_dims[0] = atoi(argv[7]);
    orig_dims[1] = atoi(argv[8]);
    orig_dims[2] = atoi(argv[9]);
    

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
    }
    // Create a 3D Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    // Get coordinates of this process
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);

    // read data for each rank
    char filename[100];
    sprintf(filename, "%s/%s_%d_%d_%d.f32", dir_prefix.c_str(), name_prefix.c_str(), coords[0], coords[1], coords[2]);
    size_t num_elements = 0;
    auto data = readfile<float>(filename, num_elements);

    std::vector<float> orig_copy(data.get(), data.get() + num_elements);

    if (data == nullptr) {
        fprintf(stderr, "Error reading file %s\n", filename);
        printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);
        MPI_Finalize();
        return 1;
    }
    int block_dims[3] = {orig_dims[0] / dims[0], orig_dims[1] / dims[1], orig_dims[2] / dims[2]};
    int block_size = block_dims[0] * block_dims[1] * block_dims[2];
    size_t block_strides[3] = {block_dims[1] * block_dims[2], block_dims[2], 1};
    // assert(num_elements == block_size);
    // printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);

    // get the global max and min to get the gloibal value range
    float local_max = *std::max_element(data.get(), data.get() + num_elements);
    float local_min = *std::min_element(data.get(), data.get() + num_elements);
    float global_max, global_min;
    // barrier
    MPI_Barrier(cart_comm);
    double time = MPI_Wtime();  // Start the timer
    if(1)
    {
        MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, cart_comm);
        MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, 0, cart_comm);
        MPI_Bcast(&global_max, 1, MPI_FLOAT, 0, cart_comm);
        MPI_Bcast(&global_min, 1, MPI_FLOAT, 0, cart_comm);
    }
    double abs_eb = rel_eb * (global_max - global_min);
    double compensation_magnitude = abs_eb * 0.9;
    // printf("Rank %d, block_size: %d\n", mpi_rank, block_size);
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(abs_eb);
    std::vector<int> quant_inds(block_size, 0);
    size_t local_zero_count = 0; 
    for (int i = 0; i < block_size; i++) {
        quant_inds[i] = quantizer.quantize_and_overwrite(data[i],0) -32768;
        if(quant_inds[i] == 0) local_zero_count++;
    }
    
    MPI_Barrier(cart_comm);

    time = MPI_Wtime() - time;
    size_t global_zero_count = 0;
    
    MPI_Reduce(&local_zero_count, &global_zero_count, 1, mpi_get_type<size_t>(), MPI_SUM, 0, cart_comm);
    if(mpi_rank == 0)
    {
        printf("quantization time = %f \n", time); 
        printf("zero count = %ld \n", local_zero_count); 
    }
    double orig_psnr = -1; 
    if(1) orig_psnr = get_psnr_mpi(orig_copy.data(), data.get(), num_elements, cart_comm);
    if (mpi_rank == 0) {
        printf("Original PSNR: %f\n", orig_psnr);
    }


    int w_block_dims[3] = {0, 0, 0};
    size_t w_block_size = 1;
    for (int i = 0; i < 3; i++) {
        if (coords[i] == dims[i] - 1 || coords[i] == 0) {
            w_block_dims[i] = block_dims[i] + 1;
        } else {
            w_block_dims[i] = block_dims[i] + 2;
        }
        w_block_size *= w_block_dims[i];
        if (mpi_rank == 21) {
            printf("w_block_dims[%d]: %d\n", i, w_block_dims[i]);
        }
    }
    size_t w_block_strides[3] = {w_block_dims[1] * w_block_dims[2], w_block_dims[2], 1};
    std::vector<int> w_quant_inds(w_block_size, 0);

    // quantization index exchange
    MPI_Barrier(cart_comm);
    time = MPI_Wtime(); 
    if(1) data_exhange3d(quant_inds.data(), block_dims, block_strides, w_quant_inds.data(), w_block_dims, w_block_strides,
                   coords, dims, cart_comm);
    // barrier
    MPI_Barrier(cart_comm);
    time = MPI_Wtime() - time;
    if(mpi_rank ==0)
    {
        printf("data exchange time = %f \n", time); 
    }
    // boundary detection and sign map generation
    std::vector<char> boundary(block_size, 0);
    std::vector<char> sign_map(block_size, 0);
    std::vector<float> compensation_map(block_size, 0.0);
    if (1)
        get_boundary_and_sign_map3d<int, char>(w_quant_inds.data(), boundary.data(), sign_map.data(), w_block_dims,
                                               w_block_strides, block_dims, block_strides, coords, dims, cart_comm);

    MPI_Barrier(cart_comm);
    if(mpi_rank ==0)
    {
        printf("boundary and sign map done \n"); 
    } 

    // edt to get the distance map and the indexes

    int depth_dim = orig_dims[0] / dims[0];
    int height_dim = orig_dims[1] / dims[0];
    int width_dim = orig_dims[2] / dims[0];
    std::array<int, 3> data_block_dims = {depth_dim, height_dim, width_dim};
    std::vector<int> index(num_elements, 0);
    std::vector<float> distance(num_elements, 0.0);

    if (1) {
        edt_3d_and_sign_map(boundary.data(), distance.data(), index.data(), sign_map.data(), data_block_dims.data(), dims, coords,
                            mpi_rank, size, cart_comm);
        MPI_Barrier(cart_comm);
    }
    if(mpi_rank ==0)
    {
        printf("first edt done  \n"); 
    } 

    // complete the sign map for non-edge voxels
    // fill the compensation map for the edge voxels
    char b_tag = 1;
    // if (1) {
    //     fill_sign_map3d<char, float>(sign_map.data(), index.data(), compensation_map.data(), boundary.data(), b_tag,
    //                                  block_size, compensation_magnitude);
    // }
    std::vector<char> w_sign_map(w_block_size, 0);
    if (1) {
        data_exhange3d(sign_map.data(), block_dims, block_strides, w_sign_map.data(), w_block_dims, w_block_strides,
                       coords, dims, cart_comm);
    }
    MPI_Barrier(cart_comm);
    if(mpi_rank ==0)
    {
        printf("second data exchange done  \n"); 
    } 
    // get the second boundary map
    std::vector<char> boundary_neutral(block_size, 0);
    if (1) {
        get_boundary3d<char, char>(w_sign_map.data(), boundary_neutral.data(), w_block_dims, w_block_strides,
                                   block_dims, block_strides, coords, dims, cart_comm);
    }
    if (1) filter_neutral_boundary3d(boundary.data(), boundary_neutral.data(), b_tag, block_size);

    MPI_Barrier(cart_comm);

    if(mpi_rank ==0)
    {
        printf("new boundary completed  \n"); 
    }  
    std::vector<int> index_neutral(num_elements, 0);
    std::vector<float> distance_neutral(num_elements, 0.0);
    if (1) {
        edt_3d(boundary_neutral.data(), distance_neutral.data(), index_neutral.data(), data_block_dims.data(), dims,
               coords, mpi_rank, size, cart_comm);


        // edt_3d_and_sign_map(boundary_neutral.data(), distance_neutral.data(), sign_map.data(), data_block_dims.data(),
        //                     dims, coords, mpi_rank, size, cart_comm);
        MPI_Barrier(cart_comm);
    }

    if(mpi_rank ==0)
    {
        printf("sedond edt done  \n"); 
    } 

    // compensation
    if (0)
    {
        compensation_idw(compensation_map.data(), data.get(), distance.data(), distance_neutral.data(), sign_map.data(),
                         block_size, compensation_magnitude);
    }

    if (1)
    {
        if(mpi_rank ==0)
        {
            printf("orig dims = %d %d %d \n", orig_dims[0], orig_dims[1], orig_dims[2]); 
        }
        compensation_rbf(compensation_map.data(), data.get(), distance.data(), index.data(), 
                        distance_neutral.data(), index_neutral.data(), orig_dims,
                        sign_map.data(),  block_size, compensation_magnitude);
    }

    MPI_Barrier(cart_comm);
    if(mpi_rank ==0)
    {
        printf("idw donw  \n"); 
    } 
    // calculate the psnr
    double psnr = -1; 
    if(1) psnr = get_psnr_mpi(orig_copy.data(), data.get(), num_elements, cart_comm);
    if (mpi_rank == 0) {
        printf("PSNR: %f\n", psnr);
    }

    // write the quantized data to a file
    // char out_filename[100];
    // sprintf(out_filename, "%s/vx_%d_%d_%d.quant.i32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<int>(out_filename, quant_inds.data(), block_size);
    // // printf("Rank %d, wrote quantized data to %s\n", mpi_rank, out_filename);
    // sprintf(out_filename, "%s/vx_%d_%d_%d.decomp.f32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<float>(out_filename, data.get(), block_size);
    // sprintf(out_filename, "%s/vx_%d_%d_%d.wquant.i32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<int>(out_filename, w_quant_inds.data(), w_block_size);

    // sprintf(out_filename, "%s/vx_%d_%d_%d.boundary.i8", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<char>(out_filename, boundary.data(), block_size);

    // sprintf(out_filename, "%s/vx_%d_%d_%d.sign.i8", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<char>(out_filename, sign_map.data(), block_size);

    // char distance_filename[100];
    // sprintf(distance_filename, "%s/distance_%d_%d_%d.f32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<float>(distance_filename, distance.data(), num_elements);

    char distance_filename[100];
    sprintf(distance_filename, "%s/vx_%d_%d_%d.d2.f32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    writefile<float>(distance_filename, distance_neutral.data(), num_elements);

    // printf("Rank %d, wrote decomp data to %s\n", mpi_rank, out_filename);
    if (mpi_rank == 0) {
        printf("Global max: %f, Global min: %f, abs_eb = %f \n", global_max, global_min, abs_eb);
        printf("Rank %d, time: %f\n", mpi_rank, time);
    }

    // terminate MPI
    MPI_Finalize();
    return 0;
}

// if(0){
//     // second boundary
//     char out_filename[100];
//     sprintf(out_filename, "%s/vx_%d_%d_%d.b2.i8", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
//     writefile(out_filename, boundary_neutral.data(), boundary_neutral.size());
//     MPI_Barrier(cart_comm);
// }
// if(0){
//     // use ghost elements for the boundary points
//     // need to pass the boundary points to the neighboring blocks
//     // check if the block is at the boundary of the global domain to decide which direction to expand the dimension
//     // copy the data to the new array
//     size_t w_offset;
//     size_t orig_idx;
//     for (int i = 0; i < block_dims[0]; i++) {
//         int w_i = coords[0] == 0 ? i : i + 1;
//         for (int j = 0; j < block_dims[1]; j++) {
//             int w_j = coords[1] == 0 ? j : j + 1;
//             for (int k = 0; k < block_dims[2]; k++) {
//                 int w_k = coords[2] == 0 ? k : k + 1;
//                 w_offset = w_i * w_block_strides[0] + w_j * w_block_strides[1] + w_k * w_block_strides[2];
//                 orig_idx = i * block_strides[0] + j * block_strides[1] + k * block_strides[2];
//                 w_quant_inds[w_offset] = quant_inds[orig_idx];
//             }
//         }
//     }
//     if (1) {
//         // pass data to the neighboring blocks
//         // usually a block has 8 neighbors
//         // 1. deal with the 6 faces
//         for (int i = 0; i < 3; i++) {
//             int face_idx1 = (i + 1) % 3;
//             int face_idx2 = (i + 2) % 3;
//             int send_buffer_size = block_dims[face_idx1] * block_dims[face_idx2];
//             std::vector<int> send_buffer = std::vector<int>(send_buffer_size, 0);  // send buffer
//             std::vector<int> recv_buffer = std::vector<int>(send_buffer_size, 0);  // receive buffer
//             int recv_coords[3] = {coords[0], coords[1], coords[2]};
//             int receiver_rank;
//             int sender_rank;
//             MPI_Request req;
//             MPI_Request request_send, request_recv;
//             {
//                 MPI_Status status;
//                 if (coords[i] != dims[i] - 1) {
//                     // pass the block_dim[i] - 1 face to coords[i] + 1
//                     int* quant_ints_start = quant_inds.data() + (block_dims[i] - 1) * block_strides[i];
//                     for (int j = 0; j < block_dims[face_idx1]; j++) {
//                         for (int k = 0; k < block_dims[face_idx2]; k++) {
//                             send_buffer[j * block_dims[face_idx2] + k] =
//                                 quant_ints_start[j * block_strides[face_idx1] + k * block_strides[face_idx2]];
//                             // send_buffer[j * block_dims[face_idx2] + k] = -1;
//                         }
//                     }
//                     recv_coords[i] = coords[i] + 1;
//                     MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
//                     MPI_Send(send_buffer.data(), send_buffer.size(), MPI_INT, receiver_rank, 0, cart_comm);
//                     // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
//                 }
//                 if (coords[i] != 0) {
//                     // pass the 0 face to coords[i] - 1
//                     for (int j = 0; j < block_dims[face_idx1]; j++) {
//                         for (int k = 0; k < block_dims[face_idx2]; k++) {
//                             send_buffer[j * block_dims[face_idx2] + k] =
//                                 quant_inds[j * block_strides[face_idx1] + k * block_strides[face_idx2]];
//                             // send_buffer[j * block_dims[face_idx2] + k] = -1;
//                         }
//                     }
//                     recv_coords[i] = coords[i] - 1;
//                     MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
//                     MPI_Send(send_buffer.data(), send_buffer.size(), MPI_INT, receiver_rank, 0, cart_comm);
//                     // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
//                 }
//             }
//             // receive
//             {
//                 int source_rank;
//                 int source_coords[3] = {coords[0], coords[1], coords[2]};
//                 MPI_Status status;
//                 if (coords[i] != dims[i] - 1) {
//                     // receive one face and update the working block
//                     source_coords[i] = coords[i] + 1;
//                     MPI_Cart_rank(cart_comm, source_coords, &source_rank);
//                     MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm, &status);
//                     // printf("Rank %d, receive from %d, status %d \n", mpi_rank, receiver_rank,
//                     // (int)status.MPI_ERROR==MPI_SUCCESS);
//                     int i_start = coords[i] == 0 ? block_dims[i] : block_dims[i] + 1;
//                     int j_start = coords[face_idx1] == 0 ? 0 : 1;
//                     int k_start = coords[face_idx2] == 0 ? 0 : 1;
//                     int* w_quant_inds_start = w_quant_inds.data() + (i_start)*w_block_strides[i] +
//                                               (j_start)*w_block_strides[face_idx1] +
//                                               (k_start)*w_block_strides[face_idx2];
//                     if (mpi_rank == 21) {
//                         printf("i_start: %d, j_start: %d, k_start: %d\n", i_start, j_start, k_start);
//                     }
//                     int count = 0;
//                     for (int j = 0; j < block_dims[face_idx1]; j++) {
//                         for (int k = 0; k < block_dims[face_idx2]; k++) {
//                             w_quant_inds_start[j * w_block_strides[face_idx1] + k * w_block_strides[face_idx2]] =
//                                 recv_buffer[j * block_dims[face_idx2] + k];
//                         }
//                     }
//                 }
//                 if (coords[i] != 0) {
//                     source_coords[i] = coords[i] - 1;
//                     MPI_Cart_rank(cart_comm, source_coords, &source_rank);
//                     // MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm,
//                     // &status);
//                     MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm, &status);
//                     // update the working block
//                     int i_start = 0;
//                     int j_start = coords[face_idx1] == 0 ? 0 : 1;
//                     int k_start = coords[face_idx2] == 0 ? 0 : 1;
//                     int* w_quant_inds_start = w_quant_inds.data() + (i_start)*w_block_strides[i] +
//                                               (j_start)*w_block_strides[face_idx1] +
//                                               (k_start)*w_block_strides[face_idx2];
//                     for (int j = 0; j < block_dims[face_idx1]; j++) {
//                         for (int k = 0; k < block_dims[face_idx2]; k++) {
//                             w_quant_inds_start[j * w_block_strides[face_idx1] + k * w_block_strides[face_idx2]] =
//                                 recv_buffer[j * block_dims[face_idx2] + k];
//                         }
//                     }
//                 }
//             }
//         }
//         // 2. deal with the 12 edges
//         // edges are send from mpi_block {i,j,k} to {i-1, j-1,k}
//         if (1) {
//             // this edge is aling dim_idx1
//             // thie edge is normal to plane dim_idx2 and dim_idx3
//             for (int dim_idx1 = 0; dim_idx1 < 3; dim_idx1++) {
//                 // send
//                 int dim_idx2 = (dim_idx1 + 1) % 3;
//                 int dim_idx3 = (dim_idx1 + 2) % 3;
//                 int buffer_size = block_dims[dim_idx1];  // the size of the buffer to be sent
//                 std::vector<int> buffer(buffer_size, 0);
//                 std::vector<int> recv_buffer(buffer_size, 0);
//                 int recv_coords[3] = {coords[0], coords[1], coords[2]};
//                 int receiver_rank;
//                 {
//                     // depends on the current block's position, we need to send the edge to the corresponding block
//                     // top left
//                     if (coords[dim_idx2] != 0 && coords[dim_idx3] != 0) {
//                         // send the edge to the block with coords[dim_idx2] - 1, coords[dim_idx3] - 1
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             buffer[i] = quant_inds[i * block_strides[dim_idx1]];
//                         }
//                         recv_coords[dim_idx2] = coords[dim_idx2] - 1;
//                         recv_coords[dim_idx3] = coords[dim_idx3] - 1;
//                         MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
//                         MPI_Send(buffer.data(), buffer.size(), MPI_INT, receiver_rank, 0, cart_comm);
//                     }
//                     // top right
//                     if (coords[dim_idx2] != 0 && coords[dim_idx3] != dims[dim_idx3] - 1) {
//                         // send the edge to the block with coords[dim_idx2] - 1, coords[dim_idx3] + 1
//                         int* quant_ints_start =
//                             quant_inds.data() + (block_dims[dim_idx3] - 1) * block_strides[dim_idx3];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             buffer[i] = quant_ints_start[i * block_strides[dim_idx1]];
//                         }
//                         recv_coords[dim_idx2] = coords[dim_idx2] - 1;
//                         recv_coords[dim_idx3] = coords[dim_idx3] + 1;
//                         MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
//                         MPI_Send(buffer.data(), buffer.size(), MPI_INT, receiver_rank, 0, cart_comm);
//                     }
//                     // bottom left
//                     if (coords[dim_idx2] != dims[dim_idx2] - 1 && coords[dim_idx3] != 0) {
//                         // send the edge to the block with coords[dim_idx2] + 1, coords[dim_idx3] - 1
//                         int* quant_ints_start =
//                             quant_inds.data() + (block_dims[dim_idx2] - 1) * block_strides[dim_idx2];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             buffer[i] = quant_ints_start[i * block_strides[dim_idx1]];
//                         }
//                         recv_coords[dim_idx2] = coords[dim_idx2] + 1;
//                         recv_coords[dim_idx3] = coords[dim_idx3] - 1;
//                         MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
//                         MPI_Send(buffer.data(), buffer.size(), MPI_INT, receiver_rank, 0, cart_comm);
//                     }
//                     // bottom right
//                     if (coords[dim_idx2] != dims[dim_idx2] - 1 && coords[dim_idx3] != dims[dim_idx3] - 1) {
//                         // send the edge to the block with coords[dim_idx2] + 1, coords[dim_idx3] + 1
//                         int* quant_ints_start = quant_inds.data() +
//                                                 (block_dims[dim_idx2] - 1) * block_strides[dim_idx2] +
//                                                 (block_dims[dim_idx3] - 1) * block_strides[dim_idx3];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             buffer[i] = quant_ints_start[i * block_strides[dim_idx1]];
//                         }
//                         recv_coords[dim_idx2] = coords[dim_idx2] + 1;
//                         recv_coords[dim_idx3] = coords[dim_idx3] + 1;
//                         MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
//                         MPI_Send(buffer.data(), buffer.size(), MPI_INT, receiver_rank, 0, cart_comm);
//                     }
//                 }
//                 // receive
//                 {
//                     int source_rank;
//                     int source_coords[3] = {coords[0], coords[1], coords[2]};
//                     MPI_Status status;
//                     // top left
//                     if (coords[dim_idx2] != 0 && coords[dim_idx3] != 0) {
//                         // receive one edge and update the working block
//                         source_coords[dim_idx2] = coords[dim_idx2] - 1;
//                         source_coords[dim_idx3] = coords[dim_idx3] - 1;
//                         MPI_Cart_rank(cart_comm, source_coords, &source_rank);
//                         MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm,
//                                  &status);
//                         // update the working block
//                         int i_start = coords[dim_idx1] == 0 ? 0 : 1;
//                         int j_start = 0;
//                         int k_start = 0;
//                         int* w_quant_inds_start = w_quant_inds.data() + (i_start)*w_block_strides[dim_idx1] +
//                                                   (j_start)*w_block_strides[dim_idx2] +
//                                                   (k_start)*w_block_strides[dim_idx3];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             w_quant_inds_start[i * w_block_strides[dim_idx1]] = recv_buffer[i];
//                         }
//                     }
//                     // top right
//                     if (coords[dim_idx2] != 0 && coords[dim_idx3] != dims[dim_idx3] - 1) {
//                         // receive the edge to the block with coords[dim_idx2] - 1, coords[dim_idx3] + 1
//                         source_coords[dim_idx2] = coords[dim_idx2] - 1;
//                         source_coords[dim_idx3] = coords[dim_idx3] + 1;
//                         MPI_Cart_rank(cart_comm, source_coords, &source_rank);
//                         MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm,
//                                  &status);
//                         // update the working block
//                         int i_start = coords[dim_idx1] == 0 ? 0 : 1;
//                         int j_start = 0;
//                         int k_start = w_block_dims[dim_idx3] - 1;
//                         int* w_quant_inds_start = w_quant_inds.data() + (i_start)*w_block_strides[dim_idx1] +
//                                                   (j_start)*w_block_strides[dim_idx2] +
//                                                   (k_start)*w_block_strides[dim_idx3];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             w_quant_inds_start[i * w_block_strides[dim_idx1]] = recv_buffer[i];
//                         }
//                     }
//                     // bottom left
//                     if (coords[dim_idx2] != dims[dim_idx2] - 1 && coords[dim_idx3] != 0) {
//                         // receive the edge to the block with coords[dim_idx2] + 1, coords[dim_idx3] - 1
//                         source_coords[dim_idx2] = coords[dim_idx2] + 1;
//                         source_coords[dim_idx3] = coords[dim_idx3] - 1;
//                         MPI_Cart_rank(cart_comm, source_coords, &source_rank);
//                         MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm,
//                                  &status);
//                         // update the working block
//                         int i_start = coords[dim_idx1] == 0 ? 0 : 1;
//                         int j_start = w_block_dims[dim_idx2] - 1;
//                         int k_start = 0;
//                         int* w_quant_inds_start = w_quant_inds.data() + (i_start)*w_block_strides[dim_idx1] +
//                                                   (j_start)*w_block_strides[dim_idx2] +
//                                                   (k_start)*w_block_strides[dim_idx3];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             w_quant_inds_start[i * w_block_strides[dim_idx1]] = recv_buffer[i];
//                         }
//                     }
//                     // bottom right
//                     if (coords[dim_idx2] != dims[dim_idx2] - 1 && coords[dim_idx3] != dims[dim_idx3] - 1) {
//                         // receive the edge to the block with coords[dim_idx2] + 1, coords[dim_idx3] + 1
//                         source_coords[dim_idx2] = coords[dim_idx2] + 1;
//                         source_coords[dim_idx3] = coords[dim_idx3] + 1;
//                         MPI_Cart_rank(cart_comm, source_coords, &source_rank);
//                         MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT, source_rank, 0, cart_comm,
//                                  &status);
//                         // update the working block
//                         int i_start = coords[dim_idx1] == 0 ? 0 : 1;
//                         int j_start = w_block_dims[dim_idx2] - 1;
//                         int k_start = w_block_dims[dim_idx3] - 1;
//                         int* w_quant_inds_start = w_quant_inds.data() + (i_start)*w_block_strides[dim_idx1] +
//                                                   (j_start)*w_block_strides[dim_idx2] +
//                                                   (k_start)*w_block_strides[dim_idx3];
//                         for (int i = 0; i < block_dims[dim_idx1]; i++) {
//                             w_quant_inds_start[i * w_block_strides[dim_idx1]] = recv_buffer[i];
//                         }
//                     }
//                 }
//             }
//         }
//         // 3. deal with the 8 corners
//         if (1) {
//             // 8 corners
//             // top left fron
//             // send
//             {
//                 int target_coords[8][3] = {
//                     {coords[0] - 1, coords[1] - 1, coords[2] - 1}, {coords[0] - 1, coords[1] - 1, coords[2] + 1},
//                     {coords[0] - 1, coords[1] + 1, coords[2] - 1}, {coords[0] - 1, coords[1] + 1, coords[2] + 1},
//                     {coords[0] + 1, coords[1] - 1, coords[2] - 1}, {coords[0] + 1, coords[1] - 1, coords[2] + 1},
//                     {coords[0] + 1, coords[1] + 1, coords[2] - 1}, {coords[0] + 1, coords[1] + 1, coords[2] + 1}};
//                 int send_index[8][3] = {{0, 0, 0},
//                                         {0, 0, block_dims[2] - 1},
//                                         {0, block_dims[1] - 1, 0},
//                                         {0, block_dims[1] - 1, block_dims[2] - 1},
//                                         {block_dims[0] - 1, 0, 0},
//                                         {block_dims[0] - 1, 0, block_dims[2] - 1},
//                                         {block_dims[0] - 1, block_dims[1] - 1, 0},
//                                         {block_dims[0] - 1, block_dims[1] - 1, block_dims[2] - 1}};
//                 for (int i = 0; i < 8; i++) {
//                     int target_rank;
//                     int cur_coords[3] = {target_coords[i][0], target_coords[i][1], target_coords[i][2]};
//                     bool valid_coords = true;
//                     for (int j = 0; j < 3; j++) {
//                         if (cur_coords[j] < 0 || cur_coords[j] >= dims[j]) {
//                             valid_coords = false;
//                             break;
//                         }
//                     }
//                     if (!valid_coords) {
//                         continue;
//                     }
//                     int res = MPI_Cart_rank(cart_comm, cur_coords, &target_rank);
//                     if (res == MPI_SUCCESS) {
//                         size_t idx = send_index[i][0] * block_strides[0] + send_index[i][1] * block_strides[1] +
//                                      send_index[i][2];
//                         MPI_Send(&quant_inds[idx], 1, MPI_INT, target_rank, 0, cart_comm);
//                         // MPI_Send(&i, 1, MPI_INT, target_rank, 0, cart_comm);
//                     }
//                 }
//             }
//             // receive
//             if (1) {
//                 MPI_Status status;
//                 int source_coords[8][3] = {
//                     {coords[0] - 1, coords[1] - 1, coords[2] - 1}, {coords[0] - 1, coords[1] - 1, coords[2] + 1},
//                     {coords[0] - 1, coords[1] + 1, coords[2] - 1}, {coords[0] - 1, coords[1] + 1, coords[2] + 1},
//                     {coords[0] + 1, coords[1] - 1, coords[2] - 1}, {coords[0] + 1, coords[1] - 1, coords[2] + 1},
//                     {coords[0] + 1, coords[1] + 1, coords[2] - 1}, {coords[0] + 1, coords[1] + 1, coords[2] + 1}};
//                 int recv_idx[8][3] = {{0, 0, 0},
//                                       {0, 0, w_block_dims[2] - 1},
//                                       {0, w_block_dims[1] - 1, 0},
//                                       {0, w_block_dims[1] - 1, w_block_dims[2] - 1},
//                                       {w_block_dims[0] - 1, 0, 0},
//                                       {w_block_dims[0] - 1, 0, w_block_dims[2] - 1},
//                                       {w_block_dims[0] - 1, w_block_dims[1] - 1, 0},
//                                       {w_block_dims[0] - 1, w_block_dims[1] - 1, w_block_dims[2] - 1}};

//                 for (int i = 0; i < 8; i++) {
//                     bool valid_coords = true;
//                     int* cur_coords = source_coords[i];
//                     for (int j = 0; j < 3; j++) {
//                         if (cur_coords[j] < 0 || cur_coords[j] >= dims[j]) {
//                             valid_coords = false;
//                             break;
//                         }
//                     }
//                     if (!valid_coords) {
//                         continue;
//                     }
//                     int source_ranks;
//                     int res = MPI_Cart_rank(cart_comm, source_coords[i], &source_ranks);
//                     if (res == MPI_SUCCESS) {
//                         size_t idx = recv_idx[i][0] * w_block_strides[0] + recv_idx[i][1] * w_block_strides[1] +
//                                      recv_idx[i][2];
//                         MPI_Recv(&w_quant_inds[idx], 1, MPI_INT, source_ranks, 0, cart_comm, &status);
//                     }
//                 }
//             }
//         }
//     }
// }