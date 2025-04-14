#ifndef EDT_LOCAL_MPI
#define EDT_LOCAL_MPI

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "mpi.h"
#include "mpi/edt.hpp"

inline void edt_core_mpi_local(int *d_output, const size_t stride, const uint rank, const uint d, const uint len,
                               const uint width, const uint height, const size_t width_stride,
                               const size_t height_stride, const int local_x, const int local_y, int **f, int *g,
                               int mpi_rank, int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height,
                               int mpi_width, int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
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
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    int chunck_start = 0;
    for (ii = 0; ii < len; ii++) {
        for (jj = 0; jj < rank; jj++) {
            f[ii + chunck_start][jj] = d_output_start[ii * stride + jj];
        }
    }

    for (ii = 0; ii < len; ii++) {
        if (f[ii][0] >= 0) {
            double fd = f[ii][d];
            double wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    double tw = (f[ii][jj] - coor[jj]);
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                double a, b, c, uR = 0, vR = 0, f1;
                idx1 = g[l];
                f1 = f[idx1][d];
                idx2 = g[l - 1];
                a = f1 - f[idx2][d];
                b = fd - f1;

                c = a + b;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f[idx2][jj] - cc;
                        double tv = f[idx1][jj] - cc;
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0) break;
                --l;
            }
            ++l;
            g[l] = ii;
        }
    }
    maxl = l;
    chunck_start = global_coord[d];
    // no need for communication since we have the g and f for all the blocks
    if (maxl >= 0) {
        l = 0;
        for (ii = chunck_start; ii < len + chunck_start; ii++) {
            double delta1 = 0, t;
            for (jj = 0; jj < rank; jj++) {
                t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];

                delta1 += t * t;
            }
            while (l < maxl) {
                double delta2 = 0.0;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];

                    delta2 += t * t;
                }
                if (delta1 <= delta2) break;
                delta1 = delta2;
                ++l;
            }
            idx1 = g[l];
            for (jj = 0; jj < rank; jj++) d_output_start[(ii - chunck_start) * stride + jj] = f[idx1][jj];
        }
    }
}

inline void edt_and_sign_core_mpi_local(int *d_output, char *sign_map, const size_t stride, const uint rank,
                                        const uint d, const uint len, const uint width, const uint height,
                                        const size_t width_stride, const size_t height_stride, const int local_x,
                                        const int local_y, int **f, char *local_sign_buffer, int *g, int mpi_rank,
                                        int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
                                        int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
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
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    char *sign_start = sign_map + local_x * width_stride / 3 + local_y * height_stride / 3;
    int chunck_start = 0;
    for (ii = 0; ii < len; ii++) {
        for (jj = 0; jj < rank; jj++) {
            f[ii + chunck_start][jj] = d_output_start[ii * stride + jj];
        }
        local_sign_buffer[ii + chunck_start] = sign_start[ii * stride / 3];
    }

    for (ii = 0; ii < len; ii++) {
        if (f[ii][0] >= 0) {
            double fd = f[ii][d];
            double wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    double tw = (f[ii][jj] - coor[jj]);
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                double a, b, c, uR = 0, vR = 0, f1;
                idx1 = g[l];
                f1 = f[idx1][d];
                idx2 = g[l - 1];
                a = f1 - f[idx2][d];
                b = fd - f1;
                c = a + b;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f[idx2][jj] - cc;
                        double tv = f[idx1][jj] - cc;
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0) break;
                --l;
            }
            ++l;
            g[l] = ii;
        }
    }
    maxl = l;
    chunck_start = global_coord[d];
    // no need for communication since we have the g and f for all the blocks
    if (maxl >= 0) {
        l = 0;
        for (ii = chunck_start; ii < len + chunck_start; ii++) {
            double delta1 = 0, t;
            for (jj = 0; jj < rank; jj++) {
                t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];

                delta1 += t * t;
            }
            while (l < maxl) {
                double delta2 = 0.0;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];

                    delta2 += t * t;
                }
                if (delta1 <= delta2) break;
                delta1 = delta2;
                ++l;
            }
            idx1 = g[l];
            for (jj = 0; jj < rank; jj++) d_output_start[(ii - chunck_start) * stride + jj] = f[idx1][jj];
            sign_start[(ii - chunck_start) * stride / 3] = local_sign_buffer[idx1];
        }
    }
}

inline void edt_local_update_forward(int *d_output, size_t stride, uint rank, uint d, uint len, uint width, uint height,
                                     size_t width_stride, size_t height_stride, int local_x, int local_y, int **f,
                                     char *local_sign_buffer, int *received_coords, int mpi_rank, int mpi_size,
                                     int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width, int mpi_direction,
                                     int *mpi_dims, MPI_Comm &cart_comm) {
    if (received_coords[0] < 0) return;
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
    int chunck_start = global_coord[d];
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    for (int ii = 0; ii < len; ii++) {
        for (int jj = 0; jj < rank; jj++) {
            f[ii][jj] = d_output_start[ii * stride + jj];
        }
    }
    if (1) {
        for (int ii = chunck_start; ii < len + chunck_start; ii++) {
            double delta1 = 0, t, delta2 = 0;
            for (int jj = 0; jj < rank; jj++) {
                t = jj == d ? received_coords[jj] - ii : received_coords[jj] - coor[jj];
                delta1 += t * t;
                t = jj == d ? f[ii - chunck_start][jj] - ii : f[ii - chunck_start][jj] - coor[jj];
                delta2 += t * t;
            }
            if (delta1 >= delta2) {
                break;
            } else {  // update
                for (int jj = 0; jj < rank; jj++) {
                    d_output_start[(ii - chunck_start) * stride + jj] = received_coords[jj];
                }
            }
        }
    }
}

inline void edt_and_sign_local_update_forward(int *d_output, char *sign_map, size_t stride, uint rank, uint d, uint len,
                                              uint width, uint height, size_t width_stride, size_t height_stride,
                                              int local_x, int local_y, int **f, char *local_sign_buffer,
                                              int *received_coords, char received_sign, int mpi_rank, int mpi_size,
                                              int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
                                              int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
    if (received_coords[0] < 0) return;
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
    int chunck_start = global_coord[d];
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    char *sign_start = sign_map + local_x * width_stride / 3 + local_y * height_stride / 3;
    for (int ii = 0; ii < len; ii++) {
        for (int jj = 0; jj < rank; jj++) {
            f[ii][jj] = d_output_start[ii * stride + jj];
        }
        local_sign_buffer[ii] = sign_start[ii * stride / 3];
    }
    if (1) {
        for (int ii = chunck_start; ii < len + chunck_start; ii++) {
            double delta1 = 0, t, delta2 = 0;
            for (int jj = 0; jj < rank; jj++) {
                t = jj == d ? received_coords[jj] - ii : received_coords[jj] - coor[jj];
                delta1 += t * t;
                t = jj == d ? f[ii - chunck_start][jj] - ii : f[ii - chunck_start][jj] - coor[jj];
                delta2 += t * t;
            }
            if (delta1 >= delta2) {
                break;
            } else {  // update
                for (int jj = 0; jj < rank; jj++) {
                    d_output_start[(ii - chunck_start) * stride + jj] = received_coords[jj];
                }
                sign_start[(ii - chunck_start) * stride / 3] = received_sign;
            }
        }
    }
}

inline void edt_local_update_backward(int *d_output, size_t stride, uint rank, uint d, uint len, uint width,
                                      uint height, size_t width_stride, size_t height_stride, int local_x, int local_y,
                                      int **f, char *local_sign_buffer, int *received_coords, int mpi_rank,
                                      int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
                                      int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
    if (received_coords[0] < 0) return;
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
    int chunck_start = global_coord[d];
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    for (int ii = 0; ii < len; ii++) {
        for (int jj = 0; jj < rank; jj++) {
            f[ii][jj] = d_output_start[ii * stride + jj];
        }
    }

    if (1) {
        for (int ii = chunck_start + len - 1; ii >= chunck_start; ii--) {
            double delta1 = 0.0, t, delta2 = 0.0;
            for (int jj = 0; jj < rank; jj++) {
                t = jj == d ? received_coords[jj] - ii : received_coords[jj] - coor[jj];
                delta1 += t * t;
                t = jj == d ? f[ii - chunck_start][jj] - ii : f[ii - chunck_start][jj] - coor[jj];
                delta2 += t * t;
            }
            if (delta1 >= delta2) {
                break;
            } else {  // update
                for (int jj = 0; jj < rank; jj++) {
                    d_output_start[(ii - chunck_start) * stride + jj] = received_coords[jj];
                }
            }
        }
    }
}

inline void edt_and_sign_local_update_backward(int *d_output, char *sign_map, size_t stride, uint rank, uint d,
                                               uint len, uint width, uint height, size_t width_stride,
                                               size_t height_stride, int local_x, int local_y, int **f,
                                               char *local_sign_buffer, int *received_coords, char received_sign,
                                               int mpi_rank, int mpi_size, int *mpi_coords, int mpi_depth,
                                               int mpi_height, int mpi_width, int mpi_direction, int *mpi_dims,
                                               MPI_Comm &cart_comm) {
    if (received_coords[0] < 0) return;
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
    int chunck_start = global_coord[d];
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    char *sign_start = sign_map + local_x * width_stride / 3 + local_y * height_stride / 3;
    for (int ii = 0; ii < len; ii++) {
        for (int jj = 0; jj < rank; jj++) {
            f[ii][jj] = d_output_start[ii * stride + jj];
        }
        local_sign_buffer[ii] = sign_start[ii * stride / 3];
    }

    if (1) {
        for (int ii = chunck_start + len - 1; ii >= chunck_start; ii--) {
            double delta1 = 0.0, t, delta2 = 0.0;
            for (int jj = 0; jj < rank; jj++) {
                t = jj == d ? received_coords[jj] - ii : received_coords[jj] - coor[jj];
                delta1 += t * t;
                t = jj == d ? f[ii - chunck_start][jj] - ii : f[ii - chunck_start][jj] - coor[jj];
                delta2 += t * t;
            }
            if (delta1 >= delta2) {
                break;
            } else {  // update
                for (int jj = 0; jj < rank; jj++) {
                    d_output_start[(ii - chunck_start) * stride + jj] = received_coords[jj];
                }
                sign_start[(ii - chunck_start) * stride / 3] = received_sign;
            }
        }
    }
}

inline void exchange_and_update(int *output, char *sign_map, int *data_block_dims, int direction, size_t *input_stride,
                                size_t *output_stride, int *face_buffer_up_send, int *face_buffer_up_recv,
                                int *face_buffer_down_send, int *face_buffer_down_recv, char *face_sign_buffer_up_send,
                                char *face_sign_buffer_up_recv, char *face_sign_buffer_down_send,
                                char *face_sign_buffer_down_recv, int **ff, char *local_sign_buffer, int mpi_rank,
                                int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
                                int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
    int x_dir = (direction + 1) % 3;
    int y_dir = (direction + 2) % 3;
    size_t send_face_size = data_block_dims[x_dir] * data_block_dims[y_dir];
    if (mpi_coords[direction] != 0) {
        int target_coord[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
        target_coord[direction] = target_coord[direction] - 1;
        int target_rank = 0;
        MPI_Cart_rank(cart_comm, target_coord, &target_rank);
        // copy data to send buffer
        int *top_face_start = output;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                for (int k = 0; k < 3; k++) {
                    face_buffer_up_send[i * data_block_dims[y_dir] * 3 + j * 3 + k] =
                        top_face_start[i * output_stride[x_dir] + j * output_stride[y_dir] + k];
                }
                face_sign_buffer_up_send[i * data_block_dims[y_dir] + j] =
                    sign_map[i * input_stride[x_dir] + j * input_stride[y_dir]];
            }
        }
        // exchange data with upside face
        MPI_Sendrecv(face_buffer_up_send, send_face_size * 3, MPI_INT, target_rank, 0, face_buffer_up_recv,
                     send_face_size * 3, MPI_INT, target_rank, 0, cart_comm, MPI_STATUS_IGNORE);
        // exchange sign map with upside face
        MPI_Sendrecv(face_sign_buffer_up_send, send_face_size, MPI_CHAR, target_rank, 0, face_sign_buffer_up_recv,
                     send_face_size, MPI_CHAR, target_rank, 0, cart_comm, MPI_STATUS_IGNORE);
        // now use the received data to update the output data
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                size_t face_index = i * data_block_dims[y_dir] + j;
                edt_and_sign_local_update_forward(
                    output, sign_map, output_stride[direction], 3, direction, data_block_dims[direction],
                    data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir], i, j,
                    ff, local_sign_buffer, &face_buffer_up_recv[face_index * 3], face_sign_buffer_up_recv[face_index],
                    mpi_rank, mpi_size, mpi_coords, data_block_dims[0], data_block_dims[1], data_block_dims[2],
                    direction, mpi_dims, cart_comm);
            }
        }
    }
    // exchane with downside
    if (mpi_coords[direction] != mpi_dims[direction] - 1) {
        int target_coord[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
        target_coord[direction] = target_coord[direction] + 1;
        int target_rank = 0;
        MPI_Cart_rank(cart_comm, target_coord, &target_rank);
        // copy data to send buffer
        int *bottom_face_start = output + (data_block_dims[direction] - 1) * output_stride[direction];
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                for (int k = 0; k < 3; k++) {
                    face_buffer_down_send[i * data_block_dims[y_dir] * 3 + j * 3 + k] =
                        bottom_face_start[i * output_stride[x_dir] + j * output_stride[y_dir] + k];
                }
                face_sign_buffer_down_send[i * data_block_dims[y_dir] + j] =
                    sign_map[i * input_stride[x_dir] + j * input_stride[y_dir]];
            }
        }
        // exchange data with downside face
        size_t send_size = data_block_dims[x_dir] * data_block_dims[y_dir] * 3;
        MPI_Sendrecv(face_buffer_down_send, send_size, MPI_INT, target_rank, 0, face_buffer_down_recv, send_size,
                     MPI_INT, target_rank, 0, cart_comm, MPI_STATUS_IGNORE);
        // exchange sign map with downside face
        MPI_Sendrecv(face_sign_buffer_down_send, send_face_size, MPI_CHAR, target_rank, 0, face_sign_buffer_down_recv,
                     send_face_size, MPI_CHAR, target_rank, 0, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                size_t face_index = i * data_block_dims[y_dir] + j;
                edt_and_sign_local_update_backward(
                    output, sign_map, output_stride[direction], 3, direction, data_block_dims[direction],
                    data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir], i, j,
                    ff, local_sign_buffer, &face_buffer_down_recv[face_index * 3],
                    face_sign_buffer_down_recv[face_index], mpi_rank, mpi_size, mpi_coords, data_block_dims[0],
                    data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
    }
}

inline void exchange_and_update(int *output, int *data_block_dims, int direction, size_t *input_stride,
                                size_t *output_stride, int *face_buffer_up_send, int *face_buffer_up_recv,
                                int *face_buffer_down_send, int *face_buffer_down_recv, int **ff,
                                char *local_sign_buffer, int mpi_rank, int mpi_size, int *mpi_coords, int mpi_depth,
                                int mpi_height, int mpi_width, int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
    int x_dir = (direction + 1) % 3;
    int y_dir = (direction + 2) % 3;
    size_t send_face_size = data_block_dims[x_dir] * data_block_dims[y_dir];
    if (mpi_coords[direction] != 0) {
        int target_coord[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
        target_coord[direction] = target_coord[direction] - 1;
        int target_rank = 0;
        MPI_Cart_rank(cart_comm, target_coord, &target_rank);
        // copy data to send buffer
        int *top_face_start = output;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                for (int k = 0; k < 3; k++) {
                    face_buffer_up_send[i * data_block_dims[y_dir] * 3 + j * 3 + k] =
                        top_face_start[i * output_stride[x_dir] + j * output_stride[y_dir] + k];
                }
            }
        }
        // exchange data with upside face
        MPI_Sendrecv(face_buffer_up_send, send_face_size * 3, MPI_INT, target_rank, 0, face_buffer_up_recv,
                     send_face_size * 3, MPI_INT, target_rank, 0, cart_comm, MPI_STATUS_IGNORE);

        // now use the received data to update the output data
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                size_t face_index = i * data_block_dims[y_dir] + j;
                edt_local_update_forward(
                    output, output_stride[direction], 3, direction, data_block_dims[direction], data_block_dims[x_dir],
                    data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer,
                    &face_buffer_up_recv[face_index * 3], mpi_rank, mpi_size, mpi_coords, data_block_dims[0],
                    data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
    }
    // exchane with downside
    if (mpi_coords[direction] != mpi_dims[direction] - 1) {
        int target_coord[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
        target_coord[direction] = target_coord[direction] + 1;
        int target_rank = 0;
        MPI_Cart_rank(cart_comm, target_coord, &target_rank);
        // copy data to send buffer
        int *bottom_face_start = output + (data_block_dims[direction] - 1) * output_stride[direction];
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                for (int k = 0; k < 3; k++) {
                    face_buffer_down_send[i * data_block_dims[y_dir] * 3 + j * 3 + k] =
                        bottom_face_start[i * output_stride[x_dir] + j * output_stride[y_dir] + k];
                }
            }
        }
        // exchange data with downside face
        size_t send_size = data_block_dims[x_dir] * data_block_dims[y_dir] * 3;
        MPI_Sendrecv(face_buffer_down_send, send_size, MPI_INT, target_rank, 0, face_buffer_down_recv, send_size,
                     MPI_INT, target_rank, 0, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                size_t face_index = i * data_block_dims[y_dir] + j;
                edt_local_update_backward(
                    output, output_stride[direction], 3, direction, data_block_dims[direction], data_block_dims[x_dir],
                    data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer,
                    &face_buffer_down_recv[face_index * 3], mpi_rank, mpi_size, mpi_coords, data_block_dims[0],
                    data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
    }
}

template <typename T_boundary, typename T_distance, typename T_index>
void edt_3d_and_sign_map_opt(T_boundary *boundary, T_distance *distance, T_index *indexes, char *sign_map,
                             int *data_block_dims, int *mpi_dims, int *mpi_coords, int mpi_rank, int size,
                             MPI_Comm cart_comm, bool local_edt = false) {
    size_t block_size = 1;
    for (int i = 0; i < 3; i++) {
        block_size *= data_block_dims[i];
    }
    std::vector<int> output_data = std::vector<int>(block_size * 3, 0);
    int *output = output_data.data();
    int max_dim, second_max_dim;
    {
        int block_dims_copy[3] = {data_block_dims[0], data_block_dims[1], data_block_dims[2]};
        std::sort(block_dims_copy, block_dims_copy + 3);
        max_dim = block_dims_copy[2];
        second_max_dim = block_dims_copy[1];
    }

    int max_mpi_dim = *std::max_element(mpi_dims, mpi_dims + 3);

    int global_dims[3] = {data_block_dims[0] * mpi_dims[0], data_block_dims[1] * mpi_dims[1],
                          data_block_dims[2] * mpi_dims[2]};

    // local data buffer to store the line
    int *g = (int *)malloc(sizeof(int) * max_dim);
    char *local_sign_buffer = (char *)malloc(sizeof(char) * max_dim);
    int *f = (int *)malloc(sizeof(int) * max_dim * 3);
    int **ff = (int **)malloc(sizeof(int *) * max_dim);
    for (int i = 0; i < max_dim; i++) {
        ff[i] = f + i * 3;
    }
    size_t face_size = max_dim * second_max_dim;
    int *face_buffer_up_send;
    int *face_buffer_up_recv;
    int *face_buffer_down_send;
    int *face_buffer_down_recv;
    char *face_sign_buffer_up_send;
    char *face_sign_buffer_up_recv;
    char *face_sign_buffer_down_send;
    char *face_sign_buffer_down_recv;
    if (local_edt == false) {
        face_buffer_up_send = (int *)malloc(sizeof(int) * face_size * 3);
        face_buffer_up_recv = (int *)malloc(sizeof(int) * face_size * 3);
        face_buffer_down_send = (int *)malloc(sizeof(int) * face_size * 3);
        face_buffer_down_recv = (int *)malloc(sizeof(int) * face_size * 3);
        face_sign_buffer_up_send = (char *)malloc(sizeof(char) * face_size);
        face_sign_buffer_up_recv = (char *)malloc(sizeof(char) * face_size);
        face_sign_buffer_down_send = (char *)malloc(sizeof(char) * face_size);
        face_sign_buffer_down_recv = (char *)malloc(sizeof(char) * face_size);
    }

    size_t input_stride[3] = {(size_t)data_block_dims[2] * data_block_dims[1], (size_t)data_block_dims[2], 1};
    size_t output_stride[3] = {(size_t)3 * input_stride[0], (size_t)3 * input_stride[1], (size_t)3 * input_stride[2]};
    int direction;
    direction = 0;
    int x_dir = (direction + 1) % 3;
    int y_dir = (direction + 2) % 3;

    edt_init_mpi(boundary, output, 1, data_block_dims[2], data_block_dims[1], data_block_dims[0], mpi_coords);

    if (1) {
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_and_sign_core_mpi_local(output, sign_map, output_stride[direction], 3, direction,
                                            data_block_dims[direction], data_block_dims[x_dir], data_block_dims[y_dir],
                                            output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer, g,
                                            mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
                                            data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        // if (local_edt == false) {
        //     MPI_Barrier(cart_comm);
        //     exchange_and_update(output, sign_map, data_block_dims, direction, input_stride, output_stride,
        //                         face_buffer_up_send, face_buffer_up_recv, face_buffer_down_send, face_buffer_down_recv,
        //                         face_sign_buffer_up_send, face_sign_buffer_up_recv, face_sign_buffer_down_send,
        //                         face_sign_buffer_down_recv, ff, local_sign_buffer, mpi_rank, size, mpi_coords,
        //                         data_block_dims[0], data_block_dims[1], data_block_dims[2], direction, mpi_dims,
        //                         cart_comm);
        //     MPI_Barrier(cart_comm);
        // }
    }

    // dim1
    direction = 1;
    if (1) {
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_and_sign_core_mpi_local(output, sign_map, output_stride[direction], 3, direction,
                                            data_block_dims[direction], data_block_dims[x_dir], data_block_dims[y_dir],
                                            output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer, g,
                                            mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
                                            data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }

        // if (local_edt == false) {
        //     MPI_Barrier(cart_comm);
        //     exchange_and_update(output, sign_map, data_block_dims, direction, input_stride, output_stride,
        //                         face_buffer_up_send, face_buffer_up_recv, face_buffer_down_send, face_buffer_down_recv,
        //                         face_sign_buffer_up_send, face_sign_buffer_up_recv, face_sign_buffer_down_send,
        //                         face_sign_buffer_down_recv, ff, local_sign_buffer, mpi_rank, size, mpi_coords,
        //                         data_block_dims[0], data_block_dims[1], data_block_dims[2], direction, mpi_dims,
        //                         cart_comm);
        //     MPI_Barrier(cart_comm);
        // }
    }

    // dim 2
    if (1) {
        direction = 2;
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_and_sign_core_mpi_local(output, sign_map, output_stride[direction], 3, direction,
                                            data_block_dims[direction], data_block_dims[x_dir], data_block_dims[y_dir],
                                            output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer, g,
                                            mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
                                            data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        // printf("rank %d finished dim 2\n", mpi_rank);
        // if (local_edt == false) {
        //     MPI_Barrier(cart_comm);
        //     exchange_and_update(output, sign_map, data_block_dims, direction, input_stride, output_stride,
        //                         face_buffer_up_send, face_buffer_up_recv, face_buffer_down_send, face_buffer_down_recv,
        //                         face_sign_buffer_up_send, face_sign_buffer_up_recv, face_sign_buffer_down_send,
        //                         face_sign_buffer_down_recv, ff, local_sign_buffer, mpi_rank, size, mpi_coords,
        //                         data_block_dims[0], data_block_dims[1], data_block_dims[2], direction, mpi_dims,
        //                         cart_comm);
        // }
    }

    // calculate distance
    calculate_distance_and_index(output, distance, indexes, data_block_dims[2], data_block_dims[1], data_block_dims[0],
                                 global_dims[2], global_dims[1], global_dims[0], mpi_coords);

    free(g);
    free(f);
    free(ff);
    free(local_sign_buffer);
    if (local_edt == false) {
        free(face_buffer_up_send);
        free(face_buffer_up_recv);
        free(face_buffer_down_send);
        free(face_buffer_down_recv);
        free(face_sign_buffer_up_send);
        free(face_sign_buffer_up_recv);
        free(face_sign_buffer_down_send);
        free(face_sign_buffer_down_recv);
    }
}

template <typename T_boundary, typename T_distance, typename T_index>
void edt_3d_opt(T_boundary *boundary, T_distance *distance, T_index *indexes, int *data_block_dims, int *mpi_dims,
                int *mpi_coords, int mpi_rank, int size, MPI_Comm cart_comm, bool local_edt = 1) {
    size_t block_size = 1;
    for (int i = 0; i < 3; i++) {
        block_size *= data_block_dims[i];
    }
    std::vector<int> output_data =
        std::vector<int>(block_size * 3, 0);
    int *output = output_data.data();
    int max_dim, second_max_dim;
    {
        int block_dims_copy[3] = {data_block_dims[0], data_block_dims[1], data_block_dims[2]};
        std::sort(block_dims_copy, block_dims_copy + 3);
        max_dim = block_dims_copy[2];
        second_max_dim = block_dims_copy[1];
    }

    int max_mpi_dim = *std::max_element(mpi_dims, mpi_dims + 3);

    int global_dims[3] = {data_block_dims[0] * mpi_dims[0], data_block_dims[1] * mpi_dims[1],
                          data_block_dims[2] * mpi_dims[2]};

    // local data buffer to store the line
    int *g = (int *)malloc(sizeof(int) * max_dim );
    char *local_sign_buffer = (char *)malloc(sizeof(char) * max_dim);
    int *f = (int *)malloc(sizeof(int) * max_dim * 3);
    int **ff = (int **)malloc(sizeof(int *) * max_dim);
    for (int i = 0; i < max_dim; i++) {
        ff[i] = f + i * 3;
    }
    size_t face_size = max_dim * second_max_dim;
    int *face_buffer_up_send;
    int *face_buffer_up_recv;
    int *face_buffer_down_send;
    int *face_buffer_down_recv;

    if (local_edt == false) {
        face_buffer_up_send = (int *)malloc(sizeof(int) * face_size * 3);
        face_buffer_up_recv = (int *)malloc(sizeof(int) * face_size * 3);
        face_buffer_down_send = (int *)malloc(sizeof(int) * face_size * 3);
        face_buffer_down_recv = (int *)malloc(sizeof(int) * face_size * 3);
    }

    size_t input_stride[3] = {(size_t)data_block_dims[2] * data_block_dims[1], (size_t)data_block_dims[2], 1};
    size_t output_stride[3] = {(size_t)3 * input_stride[0], (size_t)3 * input_stride[1], (size_t)3 * input_stride[2]};
    int direction;
    direction = 0;
    int x_dir = (direction + 1) % 3;
    int y_dir = (direction + 2) % 3;
    edt_init_mpi(boundary, output, 1, data_block_dims[2], data_block_dims[1], data_block_dims[0], mpi_coords);
    if (1) {
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_core_mpi_local(output, output_stride[direction], 3, direction, data_block_dims[direction],
                                   data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir],
                                   output_stride[y_dir], i, j, ff, g, mpi_rank, size, mpi_coords, data_block_dims[0],
                                   data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        // if (local_edt == false) {
        //     MPI_Barrier(cart_comm);
        //     exchange_and_update(output, data_block_dims, direction, input_stride, output_stride, face_buffer_up_send,
        //                         face_buffer_up_recv, face_buffer_down_send, face_buffer_down_recv, ff,
        //                         local_sign_buffer, mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
        //                         data_block_dims[2], direction, mpi_dims, cart_comm);

        //     MPI_Barrier(cart_comm);
        // }
    }

    // dim1
    direction = 1;
    if (1) {
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_core_mpi_local(output, output_stride[direction], 3, direction, data_block_dims[direction],
                                   data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir],
                                   output_stride[y_dir], i, j, ff, g, mpi_rank, size, mpi_coords, data_block_dims[0],
                                   data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        // if (local_edt == false) {
        //     MPI_Barrier(cart_comm);
        //     exchange_and_update(output, data_block_dims, direction, input_stride, output_stride, face_buffer_up_send,
        //                         face_buffer_up_recv, face_buffer_down_send, face_buffer_down_recv, ff,
        //                         local_sign_buffer, mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
        //                         data_block_dims[2], direction, mpi_dims, cart_comm);

        //     MPI_Barrier(cart_comm);
        // }
    }

    // dim 2
    if (1) {
        direction = 2;
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_core_mpi_local(output, output_stride[direction], 3, direction, data_block_dims[direction],
                                   data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir],
                                   output_stride[y_dir], i, j, ff, g, mpi_rank, size, mpi_coords, data_block_dims[0],
                                   data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        // printf("rank %d finished dim 2\n", mpi_rank);
        // if (local_edt == false) {
        //     MPI_Barrier(cart_comm);
        //     exchange_and_update(output, data_block_dims, direction, input_stride, output_stride, face_buffer_up_send,
        //                         face_buffer_up_recv, face_buffer_down_send, face_buffer_down_recv, ff,
        //                         local_sign_buffer, mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
        //                         data_block_dims[2], direction, mpi_dims, cart_comm);
        //     MPI_Barrier(cart_comm);
        // }
    }

    // calculate distance
    calculate_distance_and_index(output, distance, indexes, data_block_dims[2], data_block_dims[1], data_block_dims[0],
                                 global_dims[2], global_dims[1], global_dims[0], mpi_coords);

    free(g);
    free(f);
    free(ff);
    free(local_sign_buffer);

    if (local_edt == false) {
        free(face_buffer_up_send);
        free(face_buffer_up_recv);
        free(face_buffer_down_send);
        free(face_buffer_down_recv);
    }
}

#endif  // EDT_LOCAL_MPI