#ifndef BOUNDARY_MPI
#define BOUNDARY_MPI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "mpi.h"

// width is the fastest changing dimension
// thsi function works inside a global memory model MPI remote memory access
template <typename T_quant, typename T_boundary>
void get_boundary3d(T_quant* w_quant_inds, T_boundary* boundary, int* w_dims, size_t* w_strides, int* orig_dims,
                    size_t* orig_strides, int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    size_t w_offset;
    size_t orig_idx;
    int starts[3] = {0, 0, 0};
    int ends[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        if (mpi_coords[i] == 0) {
            starts[i] = 1;
        } else {
            starts[i] = 0;
        }
        if (mpi_coords[i] == mpi_dims[i] - 1) {
            ends[i] = orig_dims[i] - 1;
        } else {
            ends[i] = orig_dims[i] - 0;
        }
    }
    for (int i = starts[0]; i < ends[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + 1;
        for (int j = starts[1]; j < ends[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + 1;
            for (int k = starts[2]; k < ends[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + 1;
                w_offset = w_i * w_strides[0] + w_j * w_strides[1] + w_k * w_strides[2];
                orig_idx = i * orig_strides[0] + j * orig_strides[1] + k * orig_strides[2];
                T_quant cur_quant = w_quant_inds[w_offset];
                T_quant left = w_quant_inds[w_offset - 1];
                T_quant right = w_quant_inds[w_offset + 1];
                T_quant up = w_quant_inds[w_offset - w_strides[1]];
                T_quant down = w_quant_inds[w_offset + w_strides[1]];
                T_quant front = w_quant_inds[w_offset - w_strides[0]];
                T_quant back = w_quant_inds[w_offset + w_strides[0]];
                if (left != cur_quant || right != cur_quant || up != cur_quant || down != cur_quant ||
                    front != cur_quant || back != cur_quant) {
                    boundary[orig_idx] = 1;
                } else {
                    boundary[orig_idx] = 0;
                }
            }
        }
    }
}

template <typename T_data>
inline char get_sign(T_data a) {
    return a > 0 ? 1 : (a < 0 ? -1 : 0);
}

template <typename T_quant, typename T_boundary>
void get_boundary_and_sign_map3d(T_quant* w_quant_inds, T_boundary* boundary, T_boundary* sign_map, int* w_dims,
                                 size_t* w_strides, int* orig_dims, size_t* orig_strides, int* mpi_coords,
                                 int* mpi_dims, MPI_Comm& cart_comm) {
    size_t w_offset;
    size_t orig_idx;
    int starts[3] = {0, 0, 0};
    int ends[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        if (mpi_coords[i] == 0) {
            starts[i] = 1;
        } else {
            starts[i] = 0;
        }
        if (mpi_coords[i] == mpi_dims[i] - 1) {
            ends[i] = orig_dims[i] - 1;
        } else {
            ends[i] = orig_dims[i] - 0;
        }
    }
    T_quant neighbor_quant[6];
    double neighbor_grad[3];
    for (int i = starts[0]; i < ends[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + 1;
        for (int j = starts[1]; j < ends[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + 1;
            for (int k = starts[2]; k < ends[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + 1;
                w_offset = w_i * w_strides[0] + w_j * w_strides[1] + w_k * w_strides[2];
                orig_idx = i * orig_strides[0] + j * orig_strides[1] + k * orig_strides[2];
                T_quant cur_quant = w_quant_inds[w_offset];
                T_quant left = w_quant_inds[w_offset - 1];
                T_quant right = w_quant_inds[w_offset + 1];
                T_quant up = w_quant_inds[w_offset - w_strides[1]];
                T_quant down = w_quant_inds[w_offset + w_strides[1]];
                T_quant front = w_quant_inds[w_offset - w_strides[0]];
                T_quant back = w_quant_inds[w_offset + w_strides[0]];

                neighbor_quant[0] = up;
                neighbor_quant[1] = down;
                neighbor_quant[2] = left;
                neighbor_quant[3] = right;
                neighbor_quant[4] = front;
                neighbor_quant[5] = back;
                
                if(left != cur_quant || right != cur_quant || up != cur_quant || down != cur_quant || front != cur_quant || back != cur_quant) {
                    boundary[orig_idx] = 1;
                    char sign = 0; 
                    for (int i = 0; i < 6; i++) {
                        if(neighbor_quant[i] != cur_quant) {
                            sign = get_sign( neighbor_quant[i] -cur_quant); 
                            break;
                        }
                    }
                    sign_map[orig_idx] = sign;
                    double grad_x = std::abs((right - left) / 2.0);
                    double grad_y = std::abs((down - up) / 2.0);
                    double grad_z = std::abs((back - front) / 2.0);
                    double max_grad = std::max(std::max(grad_x, grad_y), grad_z);   
                    if(max_grad >=1.0) {
                        sign_map[orig_idx] = 0;
                    } 
                }
                else {
                    boundary[orig_idx] = 0;
                }
            }
        }
    }
}

template <typename T_boundary, typename T_data, typename T_index>
void fill_sign_map3d(T_boundary* sign_map, T_index* index, T_data* compensation_map, T_boundary* boundary,
                     T_boundary b_tag, size_t block_size, T_data compensation) {
    // for (size_t i = 0; i < block_size; i++) {
    //     if (sign_map[i] == b_tag) {
    //         compensation_map[i] = sign_map[i] * compensation;
    //     }
    // }
}

template <typename T_boundary>
void filter_neutral_boundary3d(T_boundary* orig_boundary, T_boundary* neutral_boundary, T_boundary b_tag,
                               size_t block_size) {
    for (size_t i = 0; i < block_size; i++) {
        if (orig_boundary[i] == b_tag && neutral_boundary[i] == b_tag) {
            neutral_boundary[i] = 0;
        }
    }
}

#endif
