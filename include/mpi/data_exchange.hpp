#ifndef MPI_DATA_EXCHANGE_HPP
#define MPI_DATA_EXCHANGE_HPP
#include <algorithm>
#include <vector>
#include "mpi.h"
#include "mpi/mpi_datatype.hpp"


template <typename T>
void data_exhange3d(T* src, int* src_dims, size_t* src_strides, T* dest, int* dest_dims, size_t* dest_strides,
                    int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    // use mpi_win to createa global memory to get the boundayer points

    // use ghost elements for the boundary points
    // need to pass the boundary points to the neighboring blocks
    // check if the block is at the boundary of the global domain to decide which direction to expand the dimension
    // copy the data to the new array
    size_t w_offset;
    size_t orig_idx;
    int mpi_rank;
    MPI_Datatype mpi_type = mpi_get_type<T>();
    // MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(T), &mpi_type);
    MPI_Comm_rank(cart_comm, &mpi_rank);
    // if(mpi_rank == 21){
    //     printf("MPI type size: %d, mpitye %d , mpi_int %d \n", sizeof(T), mpi_type, MPI_INT);
    // }

    for (int i = 0; i < src_dims[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + 1;
        for (int j = 0; j < src_dims[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + 1;
            for (int k = 0; k < src_dims[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + 1;
                w_offset = w_i * dest_strides[0] + w_j * dest_strides[1] + w_k * dest_strides[2];
                orig_idx = i * src_strides[0] + j * src_strides[1] + k * src_strides[2];
                dest[w_offset] = src[orig_idx];
            }
        }
    }
    if (1) {
        // pass data to the neighboring blocks
        // usually a block has 8 neighbors
        // 1. deal with the 6 faces
        for (int i = 0; i < 3; i++) {
            int face_idx1 = (i + 1) % 3;
            int face_idx2 = (i + 2) % 3;
            int send_buffer_size = src_dims[face_idx1] * src_dims[face_idx2];
            std::vector<T> send_buffer = std::vector<T>(send_buffer_size, 0);  // send buffer
            std::vector<T> recv_buffer = std::vector<T>(send_buffer_size, 0);  // receive buffer
            int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            int receiver_rank;
            int sender_rank;
            MPI_Request req;
            MPI_Request request_send, request_recv;

            {
                MPI_Status status;
                if (mpi_coords[i] != mpi_dims[i] - 1) {
                    // pass the block_dim[i] - 1 face to mpi_coords[i] + 1
                    int* quant_ints_start = src + (src_dims[i] - 1) * src_strides[i];
                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            send_buffer[j * src_dims[face_idx2] + k] =
                                quant_ints_start[j * src_strides[face_idx1] + k * src_strides[face_idx2]];
                            // send_buffer[j * src_dims[face_idx2] + k] = -1;
                        }
                    }
                    recv_coords[i] = mpi_coords[i] + 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    MPI_Send(send_buffer.data(), send_buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                }
                if (mpi_coords[i] != 0) {
                    // pass the 0 face to mpi_coords[i] - 1
                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            send_buffer[j * src_dims[face_idx2] + k] =
                                src[j * src_strides[face_idx1] + k * src_strides[face_idx2]];
                            // send_buffer[j * src_dims[face_idx2] + k] = -1;
                        }
                    }
                    recv_coords[i] = mpi_coords[i] - 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    MPI_Send(send_buffer.data(), send_buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                }
            }
            // receive
            {
                int source_rank;
                int source_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
                MPI_Status status;
                if (mpi_coords[i] != mpi_dims[i] - 1) {
                    // receive one face and update the working block
                    source_coords[i] = mpi_coords[i] + 1;
                    MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                    MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm, &status);
                    // printf("Rank %d, receive from %d, status %d \n", mpi_rank, receiver_rank,
                    // (int)status.MPI_ERROR==MPI_SUCCESS);
                    int i_start = mpi_coords[i] == 0 ? src_dims[i] : src_dims[i] + 1;
                    int j_start = mpi_coords[face_idx1] == 0 ? 0 : 1;
                    int k_start = mpi_coords[face_idx2] == 0 ? 0 : 1;
                    T* w_quant_inds_start = dest + (i_start)*dest_strides[i] + (j_start)*dest_strides[face_idx1] +
                                              (k_start)*dest_strides[face_idx2];
                    if (mpi_rank == 21) {
                        printf("i_start: %d, j_start: %d, k_start: %d\n", i_start, j_start, k_start);
                    }
                    int count = 0;
                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            w_quant_inds_start[j * dest_strides[face_idx1] + k * dest_strides[face_idx2]] =
                                recv_buffer[j * src_dims[face_idx2] + k];
                        }
                    }
                }
                if (mpi_coords[i] != 0) {
                    source_coords[i] = mpi_coords[i] - 1;
                    MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                    MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm, &status);
                    // update the working block
                    int i_start = 0;
                    int j_start = mpi_coords[face_idx1] == 0 ? 0 : 1;
                    int k_start = mpi_coords[face_idx2] == 0 ? 0 : 1;
                    T* w_quant_inds_start = dest + (i_start)*dest_strides[i] + (j_start)*dest_strides[face_idx1] +
                                              (k_start)*dest_strides[face_idx2];
                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            w_quant_inds_start[j * dest_strides[face_idx1] + k * dest_strides[face_idx2]] =
                                recv_buffer[j * src_dims[face_idx2] + k];
                        }
                    }
                }
            }
        }
        // 2. deal with the 12 edges
        // edges are send from mpi_block {i,j,k} to {i-1, j-1,k}
        if (1) {
            // this edge is aling dim_idx1
            // thie edge is normal to plane dim_idx2 and dim_idx3
            for (int dim_idx1 = 0; dim_idx1 < 3; dim_idx1++) {
                // send
                int dim_idx2 = (dim_idx1 + 1) % 3;
                int dim_idx3 = (dim_idx1 + 2) % 3;
                int buffer_size = src_dims[dim_idx1];  // the size of the buffer to be sent
                std::vector<T> buffer(buffer_size, 0);
                std::vector<T> recv_buffer(buffer_size, 0);
                int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
                int receiver_rank;
                {
                    // depends on the current block's position, we need to send the edge to the corresponding block
                    // top left
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] - 1
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = src[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                    }
                    // top right
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] + 1
                        T* quant_ints_start = src + (src_dims[dim_idx3] - 1) * src_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = quant_ints_start[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                    }
                    // bottom left
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] - 1
                        T* quant_ints_start = src + (src_dims[dim_idx2] - 1) * src_strides[dim_idx2];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = quant_ints_start[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                    }
                    // bottom right
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 &&
                        mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] + 1
                        T* quant_ints_start = src + (src_dims[dim_idx2] - 1) * src_strides[dim_idx2] +
                                                (src_dims[dim_idx3] - 1) * src_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = quant_ints_start[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                    }
                }
                // receive
                {
                    int source_rank;
                    int source_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
                    MPI_Status status;
                    // top left
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != 0) {
                        // receive one edge and update the working block
                        source_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        source_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                        MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm, &status);
                        // update the working block
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                        int j_start = 0;
                        int k_start = 0;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                  (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                        }
                    }
                    // top right
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // receive the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] + 1
                        source_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        source_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                        MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm, &status);
                        // update the working block
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                        int j_start = 0;
                        int k_start = dest_dims[dim_idx3] - 1;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                  (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                        }
                    }
                    // bottom left
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 && mpi_coords[dim_idx3] != 0) {
                        // receive the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] - 1
                        source_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        source_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                        MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm, &status);
                        // update the working block
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                        int j_start = dest_dims[dim_idx2] - 1;
                        int k_start = 0;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                  (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                        }
                    }
                    // bottom right
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 &&
                        mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // receive the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] + 1
                        source_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        source_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                        MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm, &status);
                        // update the working block
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                        int j_start = dest_dims[dim_idx2] - 1;
                        int k_start = dest_dims[dim_idx3] - 1;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                  (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                        }
                    }
                }
            }
        }
        // 3. deal with the 8 corners
        if (1) {
            // 8 corners
            // top left fron
            // send
            {
                int target_mpi_coords[8][3] = {{mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] + 1}};
                int send_index[8][3] = {{0, 0, 0},
                                        {0, 0, src_dims[2] - 1},
                                        {0, src_dims[1] - 1, 0},
                                        {0, src_dims[1] - 1, src_dims[2] - 1},
                                        {src_dims[0] - 1, 0, 0},
                                        {src_dims[0] - 1, 0, src_dims[2] - 1},
                                        {src_dims[0] - 1, src_dims[1] - 1, 0},
                                        {src_dims[0] - 1, src_dims[1] - 1, src_dims[2] - 1}};
                for (int i = 0; i < 8; i++) {
                    int target_rank;
                    int cur_coords[3] = {target_mpi_coords[i][0], target_mpi_coords[i][1], target_mpi_coords[i][2]};
                    bool valid_coords = true;
                    for (int j = 0; j < 3; j++) {
                        if (cur_coords[j] < 0 || cur_coords[j] >= mpi_dims[j]) {
                            valid_coords = false;
                            break;
                        }
                    }
                    if (!valid_coords) {
                        continue;
                    }
                    int res = MPI_Cart_rank(cart_comm, cur_coords, &target_rank);
                    size_t idx =
                        send_index[i][0] * src_strides[0] + send_index[i][1] * src_strides[1] + send_index[i][2];
                    MPI_Send(&src[idx], 1, mpi_type, target_rank, 0, cart_comm);
                }
            }
            // receive
            if (1) {
                MPI_Status status;
                int source_coords[8][3] = {{mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                           {mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                           {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                           {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] + 1},
                                           {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                           {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                           {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                           {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] + 1}};
                int recv_idx[8][3] = {{0, 0, 0},
                                      {0, 0, dest_dims[2] - 1},
                                      {0, dest_dims[1] - 1, 0},
                                      {0, dest_dims[1] - 1, dest_dims[2] - 1},
                                      {dest_dims[0] - 1, 0, 0},
                                      {dest_dims[0] - 1, 0, dest_dims[2] - 1},
                                      {dest_dims[0] - 1, dest_dims[1] - 1, 0},
                                      {dest_dims[0] - 1, dest_dims[1] - 1, dest_dims[2] - 1}};

                for (int i = 0; i < 8; i++) {
                    bool valid_coords = true;
                    int* cur_coords = source_coords[i];
                    for (int j = 0; j < 3; j++) {
                        if (cur_coords[j] < 0 || cur_coords[j] >= mpi_dims[j]) {
                            valid_coords = false;
                            break;
                        }
                    }
                    if (!valid_coords) {
                        continue;
                    }
                    int source_ranks;
                    int res = MPI_Cart_rank(cart_comm, source_coords[i], &source_ranks);
                    size_t idx = recv_idx[i][0] * dest_strides[0] + recv_idx[i][1] * dest_strides[1] + recv_idx[i][2];
                    MPI_Recv(&dest[idx], 1, mpi_type, source_ranks, 0, cart_comm, &status);
                }
            }
        }
    }
}

#endif  // MPI_DATA_EXCHANGE_HPP
