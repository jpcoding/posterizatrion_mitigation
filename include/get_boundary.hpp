#ifndef GET_BOUNDARY_HPP
#define GET_BOUNDARY_HPP
#include <cstddef>
#include <iostream>
#include <vector>

namespace PM {
std::vector<char> get_boundary_2d(int* quant_index, int N,
                                 int *dims) {
  std::vector<char> boundary;
  
  size_t stride[2];
  stride[0] = dims[1];
  stride[1] = 1;
  size_t n = dims[0] * dims[1];
  boundary.resize(n, 0);


  for (size_t i = 1; i < dims[0] - 1; i++) {
    for (size_t j = 1; j < dims[1] - 1; j++) {
      size_t idx = i * stride[0] + j * stride[1];
      if (quant_index[idx] != quant_index[idx - stride[0]] ||
          quant_index[idx] != quant_index[idx + stride[0]] ||
          quant_index[idx] != quant_index[idx - stride[1]] ||
          quant_index[idx] != quant_index[idx + stride[1]]) {
        boundary[idx] = 1;
      }
    }
  }
  // filter boundary
  for (size_t i = 1; i < dims[0] - 1; i++) {
    for (size_t j = 1; j < dims[1] - 1; j++) {
      size_t idx = i * stride[0] + j * stride[1];
      if (quant_index[idx] != 0) {
        int left = quant_index[idx - stride[0]];
        int right = quant_index[idx + stride[0]];
        int up = quant_index[idx - stride[1]];
        int down = quant_index[idx + stride[1]];
        if (left == 0 && right == 0 && up == 0 && down == 0) {
          boundary[idx] = 0;
          boundary[idx - stride[0]] = 0;
          boundary[idx + stride[0]] = 0;
          boundary[idx - stride[1]] = 0;
          boundary[idx + stride[1]] = 0;
        }
      }
    }
  }
  return boundary;
}

std::vector<char> get_boundary_3d(int* quant_index, int N,
                                 int *dims) {
  // define stride
  size_t stride[3];
  stride[0] = dims[1] * dims[2];
  stride[1] = dims[2];
  stride[2] = 1;

  std::vector<char> boundary;
  size_t n = dims[0] * dims[1] * dims[2];
  boundary.resize(n, 0);

  for (size_t i = 1; i < dims[0] - 1; i++) {
    for (size_t j = 1; j < dims[1] - 1; j++) {
      for (size_t k = 1; k < dims[2] - 1; k++) {
        size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
        if (quant_index[idx] != quant_index[idx - stride[0]] ||
            quant_index[idx] != quant_index[idx + stride[0]] ||
            quant_index[idx] != quant_index[idx - stride[1]] ||
            quant_index[idx] != quant_index[idx + stride[1]] ||
            quant_index[idx] != quant_index[idx - stride[2]] ||
            quant_index[idx] != quant_index[idx + stride[2]]) {
          boundary[idx] = 1;
        }
      }
    }
  }
  // filter boundary
  for (size_t i = 1; i < dims[0] - 1; i++) {
    for (size_t j = 1; j < dims[1] - 1; j++) {
      for (size_t k = 1; k < dims[2] - 1; k++) {
        size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
        if (quant_index[idx] != 0) {
          int left = quant_index[idx - stride[0]];
          int right = quant_index[idx + stride[0]];
          int up = quant_index[idx - stride[1]];
          int down = quant_index[idx + stride[1]];
          int front = quant_index[idx - stride[2]];
          int back = quant_index[idx + stride[2]];
          if (left == 0 && right == 0 && up == 0 && down == 0 && front == 0 &&
              back == 0) {
            boundary[idx] = 0;
            boundary[idx - stride[0]] = 0;
            boundary[idx + stride[0]] = 0;
            boundary[idx - stride[1]] = 0;
            boundary[idx + stride[1]] = 0;
            boundary[idx - stride[2]] = 0;
            boundary[idx + stride[2]] = 0;
          }
        }
      }
    }
  }
  return boundary;
}

std::vector<char> get_boundary(int* quant_index, int N, int *dims) {
  if (N == 2) {
    return get_boundary_2d(quant_index, N, dims);
  } else if (N == 3) {
    return get_boundary_3d(quant_index, N, dims);
  } else {
    std::cout << "Error: N should be 2 or 3" << std::endl;
    exit(1);
  }
}
} // namespace PM
#endif

// int main(int argc, char** argv)
// {

//     return 0;
// }
