#ifndef COMPENSATION_MPI
#define COMPENSATION_MPI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        }
        else {
            starts[i] = 0;
        }
        if (mpi_coords[i] == mpi_dims[i] - 1) {
            ends[i] = orig_dims[i] - 1;
        }
        else {
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

template<typename T_data> 
inline char get_sign(T_data &a) {
    return a > 0 ? 1 : (a < 0 ? -1 : 0);
}

template <typename T_quant, typename T_boundary>
void get_sign_map3d(T_quant* w_quant_inds, T_boundary* sign_map, int* w_dims, size_t* w_strides, int* orig_dims,
                    size_t* orig_strides, int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    size_t w_offset;
    size_t orig_idx;
    int starts[3] = {0, 0, 0};
    int ends[3] = {0, 0, 0}; 
    for (int i = 0; i < 3; i++) {
        if (mpi_coords[i] == 0) {
            starts[i] = 1;
        }
        else {
            starts[i] = 0;
        }
        if (mpi_coords[i] == mpi_dims[i] - 1) {
            ends[i] = orig_dims[i] - 1;
        }
        else {
            ends[i] = orig_dims[i] - 0;
        }
        
    }
    T_quant neighbor_quant[6];
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
                neighbor_quant[2] = front;
                neighbor_quant[3] = back;
                neighbor_quant[4] = left;
                neighbor_quant[5] = right;
                int sign = 0; 
                for(int l = 0; l < 6; l++) {
                    sign = l%2 == 0 ? -1 : 1; 
                    if (neighbor_quant[l] != cur_quant) {
                        sign *= get_sign(neighbor_quant[l] - cur_quant);
                        break;
                    }
                }
                sign_map[orig_idx] = sign;
            }
        }   
    }
}



#endif
