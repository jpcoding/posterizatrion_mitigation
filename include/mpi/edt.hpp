#ifndef EDT_MPI
#define EDT_MPI

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <vector>

// this function does not require communication between processes
void edt_init_mpi(char *input, int *output, char b_tag, uint width, uint height, uint depth, int *mpi_coords) {
    uint block_size = width * height * depth;
    int dim = 3;
    int global_z = mpi_coords[0] * depth;
    int global_y = mpi_coords[1] * height;
    int global_x = mpi_coords[2] * width;
    for (uint i = 0; i < block_size; i++) {
        int z = i / (width * height);
        int y = (i % (width * height)) / width;
        int x = (i % (width * height)) % width;
        if (input[i] == b_tag) {
            output[i * dim] = z + global_z;
            output[i * dim + 1] = y + global_y;
            output[i * dim + 2] = x + global_x;
        } else {
            output[i * dim] = -1;
            //   // output[i*dim + 1] = -1;
            //   // output[i*dim + 2] = -1;
            //   output[i*dim] = z + global_z;
            //   output[i*dim + 1] = y + global_y;
            //   output[i*dim + 2] = x + global_x;
        }
    }
}

template <typename T>
void calculate_distance(int *output, T *distance, uint width, uint height, uint depth, int *mpi_coords) {
    uint block_size = width * height * depth;
    int dim = 3;
    int global_z = mpi_coords[0] * depth;
    int global_y = mpi_coords[1] * height;
    int global_x = mpi_coords[2] * width;
    for (uint i = 0; i < block_size; i++) {
        int z = i / (width * height);
        int y = (i % (width * height)) / width;
        int x = (i % (width * height)) % width;
        int dz = output[i * dim] - z - global_z;
        int dy = output[i * dim + 1] - y - global_y;
        int dx = output[i * dim + 2] - x - global_x;
        distance[i] = dz * dz + dy * dy + dx * dx;
        distance[i] = std::sqrt(distance[i]);
    }
}

void edt_core_mpi(int *d_output, size_t stride, uint rank, uint d, uint len, uint width, uint height,
                  size_t width_stride, size_t height_stride, int local_x, int local_y, int **f, int *g, int mpi_rank,
                  int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width, int mpi_direction,
                  int *mpi_dims, MPI_Comm &cart_comm) {
    int l = -1, ii, maxl, idx1, idx2, jj;

    int dim = 3;
    int global_z = mpi_coords[0] * mpi_depth;
    int global_y = mpi_coords[1] * mpi_height;
    int global_x = mpi_coords[2] * mpi_width;

    int global_coord[3];
    global_coord[0] = global_z;
    global_coord[1] = global_y;
    global_coord[2] = global_x;

    int coor[3];
    coor[d] = 0;
    coor[(d + 1) % 3] = local_x + global_coord[(d + 1) % 3];
    coor[(d + 2) % 3] = local_y + global_coord[(d + 2) % 3];

    int global_line_size = len * mpi_dims[d];
    //   global_line_size = 384;

    //   MPI_Comm_rank(cart_comm, &mpi_rank);

    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;

    int chunck_start = global_coord[d];

    for (ii = 0; ii < len; ii++) {
        for (jj = 0; jj < rank; jj++) {
            f[ii + chunck_start][jj] = d_output_start[ii * stride + jj];
        }
    }

    MPI_Status MPI_status;

    int *fptr = f[0];
    int f_buffer_length = len * 3 * mpi_coords[d];
    // changing size of each time how much to pass

    if (mpi_coords[d] != 0) {
        int sender_coords[3] = {0, 0, 0};
        int sender_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, sender_coords);
        sender_coords[d] = mpi_coords[d] - 1;
        MPI_Cart_rank(cart_comm, sender_coords, &sender_rank);
        MPI_Recv(&l, 1, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        MPI_Recv(g, global_line_size, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        MPI_Recv(fptr, len * rank * (mpi_coords[d]), MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        // printf("Rank %d received l from %d, l = %d, status =%d \n", mpi_rank,
        //    neighbor_up, l, MPI_status.MPI_ERROR);
        // wait for the top block to finish
    }
    int local_l;

    for (ii = chunck_start; ii < len + chunck_start; ii++) {
        if (f[ii][0] >= 0) {
            int fd = f[ii][d];
            int wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    int tw = (f[ii][jj] - coor[jj]);
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                int a, b, c, uR = 0.0, vR = 0.0, f1;
                idx1 = g[l];
                idx2 = g[l - 1];
                f1 = f[idx1][d];
                a = f1 - f[idx2][d];
                b = fd - f1;
                c = a + b;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        int cc = coor[jj];
                        int tu = f[idx2][jj] - cc;
                        int tv = f[idx1][jj] - cc;
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0.0) {
                    break;
                    --l;
                }
                ++l;
                g[l] = ii;
            }
        }

        if (mpi_coords[d] < mpi_dims[d] - 1) {
            // copy the g to the global g
            int recr_coords[3] = {0, 0, 0};
            int receiver_rank;
            MPI_Cart_coords(cart_comm, mpi_rank, 3, recr_coords);
            recr_coords[d] = mpi_coords[d] + 1;
            MPI_Cart_rank(cart_comm, recr_coords, &receiver_rank);
            MPI_Send(&l, 1, MPI_INT, receiver_rank, 0, cart_comm);
            MPI_Send(g, global_line_size, MPI_INT, receiver_rank, 0, cart_comm);
            MPI_Send(fptr, len * rank * (mpi_coords[d] + 1), MPI_INT, receiver_rank, 0, cart_comm);
        }
        // barrier

        MPI_Barrier(cart_comm);
        // now we have to update the g and f for all the previous blocks

        fptr = f[0];

        // TODO
        //  1. optimize the communication by only pass necessary length of f;

        if (mpi_coords[d] == mpi_dims[d] - 1) {
            int aux_coord[3] = {0, 0, 0};
            int cur_rank;
            MPI_Comm_rank(cart_comm, &cur_rank);
            MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
            for (int i = 0; i < mpi_dims[d] - 1; i++) {
                aux_coord[d] = i;
                int receiver_id;
                MPI_Cart_rank(cart_comm, aux_coord, &receiver_id);
                MPI_Send(&l, 1, MPI_INT, receiver_id, 0, cart_comm);
                MPI_Send(g, global_line_size, MPI_INT, receiver_id, 0, cart_comm);
                MPI_Send(fptr, global_line_size * 3, MPI_INT, receiver_id, 0, cart_comm);  // send the whole array
            }
        } else {
            // receive informaton from the last block
            int aux_coord[3] = {0, 0, 0};
            int cur_rank;
            MPI_Comm_rank(cart_comm, &cur_rank);
            MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
            aux_coord[d] = mpi_dims[d] - 1;
            int sender_id;
            MPI_Cart_rank(cart_comm, aux_coord, &sender_id);
            MPI_Recv(&l, 1, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
            MPI_Recv(g, global_line_size, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
            MPI_Recv(fptr, global_line_size * 3, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        }

        maxl = l;
        int local_maxl = local_l;

        // no need for communication since we have the g and f for all the blocks
        if (local_maxl >= 0) {
            l = 0;
            for (ii = chunck_start; ii < len + chunck_start; ii++) {
                int delta1 = 0.0, t;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];
                    delta1 += t * t;
                }
                while (l < maxl) {
                    int delta2 = 0.0;
                    for (jj = 0; jj < rank; jj++) {
                        t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];
                        delta2 += t * t;
                    }
                    if (delta1 <= delta2) break;
                    delta1 = delta2;
                    ++l;
                }
                idx1 = g[l];
                for (jj = 0; jj < rank; jj++) {
                    d_output_start[(ii - chunck_start) * stride + jj] = f[idx1][jj];
                }
            }
        }
    }
}

#endif  // EDT_MPI