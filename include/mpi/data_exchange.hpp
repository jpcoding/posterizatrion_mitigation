#ifndef MPI_DATA_EXCHANGE_HPP
#define MPI_DATA_EXCHANGE_HPP
#include <cstddef>
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
        // 1. deal with the 6 faces
        for (int i = 0; i < 3; i++) {
            int face_idx1 = (i + 1) % 3;
            int face_idx2 = (i + 2) % 3;
            int send_buffer_size = src_dims[face_idx1] * src_dims[face_idx2];
            std::vector<T> send_buffer_vector = std::vector<T>(send_buffer_size, 0);  // send buffer
            std::vector<T> recv_buffer_vector = std::vector<T>(send_buffer_size, 0);  // receive buffer
            T* send_buffer = send_buffer_vector.data();
            T* recv_buffer = recv_buffer_vector.data();
            int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            int receiver_rank;
            int sender_rank;
            MPI_Request req;
            MPI_Request request_send, request_recv;
            MPI_Status status;
            // printf("send buffer \n");
            //  if(mpi_rank ==0 ) printf("coords %d, %d %d \n ",mpi_coords[0], mpi_coords[1], mpi_coords[2]);
            if (1) {
                if (mpi_coords[i] != mpi_dims[i] - 1) {
                    // pass the block_dim[i] - 1 face to mpi_coords[i] + 1
                    T* quant_ints_start = src + (src_dims[i] - 1) * src_strides[i];
                    // printf("start  copy to buffer\n");

                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            send_buffer[j * src_dims[face_idx2] + k] =
                                quant_ints_start[j * src_strides[face_idx1] + k * src_strides[face_idx2]];
                            // send_buffer[j * src_dims[face_idx2] + k] = -1;
                        }
                    }
                    // printf("complete copy to buffer\n");
                    recv_coords[i] = mpi_coords[i] + 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    // MPI_Send(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, cart_comm);
                    MPI_Sendrecv(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, recv_buffer,
                                 send_buffer_size, mpi_type, receiver_rank, 0, cart_comm, &status);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    {
                        int i_start = mpi_coords[i] == 0 ? src_dims[i] : src_dims[i] + 1;
                        int j_start = mpi_coords[face_idx1] == 0 ? 0 : 1;
                        int k_start = mpi_coords[face_idx2] == 0 ? 0 : 1;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[i] + (j_start)*dest_strides[face_idx1] +
                                                (k_start)*dest_strides[face_idx2];
                        // if (mpi_rank == 21) {
                        //     printf("i_start: %d, j_start: %d, k_start: %d\n", i_start, j_start, k_start);
                        // }
                        for (int j = 0; j < src_dims[face_idx1]; j++) {
                            for (int k = 0; k < src_dims[face_idx2]; k++) {
                                w_quant_inds_start[j * dest_strides[face_idx1] + k * dest_strides[face_idx2]] =
                                    recv_buffer[j * src_dims[face_idx2] + k];
                            }
                        }
                    }
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
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    // MPI_Send(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, cart_comm);
                    MPI_Sendrecv(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, recv_buffer,
                                 send_buffer_size, mpi_type, receiver_rank, 0, cart_comm, &status);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    {
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
                MPI_Status status;
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
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);

                        // MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm,
                        // &status);

                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        // update the working block
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = 0;
                            int k_start = 0;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
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
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = 0;
                            int k_start = dest_dims[dim_idx3] - 1;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
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
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = dest_dims[dim_idx2] - 1;
                            int k_start = 0;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
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
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        {
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
        }
        // 3. deal with the 8 corners
        if (1) {
            // 8 corners
            // top left fron
            // send
            {
                MPI_Status status;
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

                int receive_index[8][3] = {{0, 0, 0},
                                           {0, 0, dest_dims[2] - 1},
                                           {0, dest_dims[1] - 1, 0},
                                           {0, dest_dims[1] - 1, dest_dims[2] - 1},
                                           {dest_dims[0] - 1, 0, 0},
                                           {dest_dims[0] - 1, 0, dest_dims[2] - 1},
                                           {dest_dims[0] - 1, dest_dims[1] - 1, 0},
                                           {dest_dims[0] - 1, dest_dims[1] - 1, dest_dims[2] - 1}};
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
                    if (valid_coords) {
                        int res = MPI_Cart_rank(cart_comm, cur_coords, &target_rank);
                        size_t idx =
                            send_index[i][0] * src_strides[0] + send_index[i][1] * src_strides[1] + send_index[i][2];
                        // MPI_Send(&src[idx], 1, mpi_type, target_rank, 0, cart_comm);
                        size_t idx2 = receive_index[i][0] * dest_strides[0] + receive_index[i][1] * dest_strides[1] +
                                      receive_index[i][2];
                        MPI_Sendrecv(&src[idx], 1, mpi_type, target_rank, 0, &dest[idx2], 1, mpi_type, target_rank, 0,
                                     cart_comm, &status);
                    }
                }
            }
        }
    }
}

template <typename T>
void data_exhange3d_extended(T* src, int* src_dims, size_t* src_strides, T* dest, int* dest_dims, size_t* dest_strides,
                             int extend_size, int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
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
    int ndims = 3;

    for (int i = 0; i < src_dims[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + extend_size;
        for (int j = 0; j < src_dims[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + extend_size;
            for (int k = 0; k < src_dims[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + extend_size;
                w_offset = w_i * dest_strides[0] + w_j * dest_strides[1] + w_k * dest_strides[2];
                orig_idx = i * src_strides[0] + j * src_strides[1] + k * src_strides[2];
                dest[w_offset] = src[orig_idx];
            }
        }
    }
    if (1) {
        // 1. deal with the 6 faces i
        for (int i = 0; i < 3; i++) {  // i is the direction normal to the face
            int face_idx1 = (i + 1) % 3;
            int face_idx2 = (i + 2) % 3;
            int send_buffer_size = src_dims[face_idx1] * src_dims[face_idx2] * extend_size;
            int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
            subsizes[i] = extend_size;
            size_t subarray_size = (size_t)src_dims[face_idx1] * src_dims[face_idx2] * extend_size;
            int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            int receiver_rank;
            int sender_rank;
            MPI_Status status;
            MPI_Datatype subarray_send;
            MPI_Datatype subarray_recv;
            if (1) {
                if (mpi_coords[i] != mpi_dims[i] - 1) {
                    // create a view on the src array
                    recv_coords[i] = mpi_coords[i] + 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    int src_starts[3] = {0, 0, 0};
                    src_starts[i] = src_dims[i] - extend_size;
                    MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_send);
                    MPI_Type_commit(&subarray_send);
                    int i_start = mpi_coords[i] == 0 ? src_dims[i] : src_dims[i] + extend_size;
                    int j_start = mpi_coords[face_idx1] == 0 ? 0 : extend_size;
                    int k_start = mpi_coords[face_idx2] == 0 ? 0 : extend_size;
                    int dest_starts[3] = {0, 0, 0};
                    dest_starts[i] = i_start;
                    dest_starts[face_idx1] = j_start;
                    dest_starts[face_idx2] = k_start;
                    MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_recv);
                    MPI_Type_commit(&subarray_recv);
                    MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, 
                                        dest, 1, subarray_recv, receiver_rank, 0,
                                 cart_comm, &status);
                    MPI_Type_free(&subarray_send);
                    MPI_Type_free(&subarray_recv);
                }
                if (mpi_coords[i] != 0) {
                    // pass the 0 face to mpi_coords[i] - 1
                    recv_coords[i] = mpi_coords[i] - 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    int src_starts[3] = {0, 0, 0};
                    src_starts[i] = 0;
                    MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_send);
                    MPI_Type_commit(&subarray_send);
                    int i_start = 0;
                    int j_start = mpi_coords[face_idx1] == 0 ? 0 : extend_size;
                    int k_start = mpi_coords[face_idx2] == 0 ? 0 : extend_size;
                    int dest_starts[3] = {0, 0, 0};
                    dest_starts[i] = i_start;
                    dest_starts[face_idx1] = j_start;
                    dest_starts[face_idx2] = k_start;
                    MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_recv);
                    MPI_Type_commit(&subarray_recv);
                    MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                 cart_comm, &status);
                    MPI_Type_free(&subarray_send);
                    MPI_Type_free(&subarray_recv);
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
                int buffer_size = src_dims[dim_idx1] * extend_size * extend_size;  // the size of the buffer to be sent
                int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
                subsizes[dim_idx2] = extend_size;
                subsizes[dim_idx3] = extend_size;
                int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
                int receiver_rank;
                MPI_Status status;
                MPI_Datatype subarray_send;
                MPI_Datatype subarray_recv;
                {
                    // depends on the current block's position, we need to send the edge to the corresponding block
                    // top left
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] - 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx1] = 0;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = 0;
                        int k_start = 0;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                    // top right
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] + 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx3] = src_dims[dim_idx3] - extend_size;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = 0;
                        int k_start = dest_dims[dim_idx3] - extend_size;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                    // bottom left
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] - 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx2] = src_dims[dim_idx2] - extend_size;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = dest_dims[dim_idx2] - extend_size;
                        int k_start = 0;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                    // bottom right
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 &&
                        mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] + 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx2] = src_dims[dim_idx2] - extend_size;
                        src_starts[dim_idx3] = src_dims[dim_idx3] - extend_size;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);

                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = dest_dims[dim_idx2] - extend_size;
                        int k_start = dest_dims[dim_idx3] - extend_size;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
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
                int send_index[8][3] = {
                    {0, 0, 0},
                    {0, 0, src_dims[2] - extend_size},
                    {0, src_dims[1] - extend_size, 0},
                    {0, src_dims[1] - extend_size, src_dims[2] - extend_size},
                    {src_dims[0] - extend_size, 0, 0},
                    {src_dims[0] - extend_size, 0, src_dims[2] - extend_size},
                    {src_dims[0] - extend_size, src_dims[1] - extend_size, 0},
                    {src_dims[0] - extend_size, src_dims[1] - extend_size, src_dims[2] - extend_size}};
                int receive_index[8][3] = {
                    {0, 0, 0},
                    {0, 0, dest_dims[2] - extend_size},
                    {0, dest_dims[1] - extend_size, 0},
                    {0, dest_dims[1] - extend_size, dest_dims[2] - extend_size},
                    {dest_dims[0] - extend_size, 0, 0},
                    {dest_dims[0] - extend_size, 0, dest_dims[2] - extend_size},
                    {dest_dims[0] - extend_size, dest_dims[1] - extend_size, 0},
                    {dest_dims[0] - extend_size, dest_dims[1] - extend_size, dest_dims[2] - extend_size}};
                int subsizes[3] = {extend_size, extend_size, extend_size};
                MPI_Datatype subarray_send;
                MPI_Datatype subarray_recv;
                MPI_Status status;
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
                    if (valid_coords) {
                        int res = MPI_Cart_rank(cart_comm, cur_coords, &target_rank);
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, send_index[i], MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, receive_index[i], MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, target_rank, 0, dest, 1, subarray_recv, target_rank,
                                     0, cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                }
            }
        }
    }
}

#endif  // MPI_DATA_EXCHANGE_HPP
