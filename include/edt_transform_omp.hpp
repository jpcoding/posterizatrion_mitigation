#ifndef EDT_TRANSFORM_OMP_HPP
#define EDT_TRANSFORM_OMP_HPP

#include <omp.h>

#include <cmath>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>
#include <vector>

#include "utils/timer.hpp"

namespace PM2 {

using npy_intp = int;
using npy_uint32 = unsigned int;
using npy_int32 = int;
using npy_double = double;
using npy_int8 = char;
#define NPY_MAXDIMS 5

class EDT_OMP{



public:
    EDT_OMP() = default; 

void set_num_threads(int num_threads)
{
    this->num_threads = num_threads;
}

double  get_edt_time()
{
    return edt_time;  
}

double get_distance_time()
{
    return distance_time; 
} 

double get_total_time()
{
    return edt_time + distance_time; 
} 

void reset_timer ()
{
    this->edt_time = 0; 
    this->distance_time = 0; 
}


static void VoronoiFT(int *pf, npy_intp len, npy_intp *coor, int rank,
                     int d, npy_intp stride, npy_intp cstride,
                      npy_intp **f, npy_intp *g) {
    npy_intp l = -1, ii, maxl, idx1, idx2;
    npy_intp jj;

    for (jj = 0; jj < rank; jj++)
        for (ii = 0; ii < len; ii++) f[ii][jj] = *(pf + ii * stride + cstride * jj); 

    for (ii = 0; ii < len; ii++) {
        if (*(pf + ii * stride) >= 0) {
            double fd = f[ii][d];
            double wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    double tw = f[ii][jj] - coor[jj];
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                double a, b, c, uR = 0.0, vR = 0.0, f1;
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
                if (c * vR - b * uR - a * wR - a * b * c <= 0.0) break;
                --l;
            }
            ++l;
            g[l] = ii;
        }
    }
    maxl = l;
    if (maxl >= 0) {
        l = 0;
        for (ii = 0; ii < len; ii++) {
            double delta1 = 0.0, t;
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
            for (jj = 0; jj < rank; jj++) *(pf + ii * stride + jj * cstride) = f[idx1][jj];
        }
    }
}

void ComputeFT2D3D(char *pi, int *pf, npy_intp *ishape, const npy_intp *istrides, const npy_intp *fstrides,
                          int rank, npy_intp **f, npy_intp *g) {
    std::vector<npy_intp> coor(rank, 0);

    if (rank == 2) {
        omp_set_num_threads(num_threads);
        int max_dim = std::max(ishape[0], ishape[1]);
        std::vector<std::vector<npy_intp>> local_f(num_threads, std::vector<npy_intp>(max_dim * 2));
        std::vector<std::vector<npy_intp>> local_g(num_threads, std::vector<npy_intp>(max_dim));
        std::vector<std::vector<npy_intp *>> local_f_ptrs(num_threads, std::vector<npy_intp *>(max_dim));
        std::vector<std::array<int, 2>> coor_local(num_threads, {0, 0});
        for (int i = 0; i < num_threads; i++) {
            for (int j = 0; j < max_dim; j++) {
                local_f_ptrs[i][j] = local_f[i].data() + j * 2;
            }
        }
        // std::cout << "num_threads = " << num_threads << std::endl;  
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < ishape[1]; i++) {
            int thread_id = omp_get_thread_num(); 
            coor_local[thread_id][1] = i;
            for (int j = 0; j < ishape[0]; j++) {
                size_t idx = i * istrides[1] + j * istrides[0];
                if (pi[idx]) {
                    pf[idx] = -1;
                } else {
                    pf[idx] = j;
                    pf[fstrides[0] + idx] = i;
                }
            }
            VoronoiFT(pf + i * fstrides[2], 
            ishape[0], coor_local[thread_id].data(), rank, 0, 
            fstrides[1], fstrides[0], local_f_ptrs[thread_id].data(), 
                                                            local_g[thread_id].data());
        }
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < ishape[0]; i++) {
            int thread_id = omp_get_thread_num(); 
            coor_local[thread_id][0] = i;
            VoronoiFT(pf + i * fstrides[1], ishape[1], coor_local[thread_id].data(), rank, 
                        1, fstrides[2], fstrides[0], local_f_ptrs[thread_id].data(), 
                                                            local_g[thread_id].data());
        }

    } else if (rank == 3) {
        // The last dimension is the fastest changing dimension

        omp_set_num_threads(num_threads);
        // get the current number of threads
        // std::cout << "num_threads = " << omp_get_max_threads() << std::endl;

        // prepare shared memory for the threads
        int max_dim = std::max(ishape[0], std::max(ishape[1], ishape[2]));

        std::vector<std::vector<npy_intp>> local_f(num_threads, std::vector<npy_intp>(max_dim * 3));
        std::vector<std::vector<npy_intp>> local_g(num_threads, std::vector<npy_intp>(max_dim));
        std::vector<std::vector<npy_intp *>> local_f_ptrs(num_threads, std::vector<npy_intp *>(max_dim));
        std::vector<std::array<int, 3>> coor_local(num_threads, {0, 0, 0});
        for (int i = 0; i < num_threads; i++) {
            for (int j = 0; j < max_dim; j++) {
                local_f_ptrs[i][j] = local_f[i].data() + j * 3;
            }
        }

#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[2]; i++)  // 384
        {
            for (int j = 0; j < ishape[1]; j++)  // 384
            {
                for (int k = 0; k < ishape[0]; k++) {
                    size_t idx = i * istrides[2] + j * istrides[1] + k * istrides[0];
                    if (pi[idx]) {
                        pf[idx] = -1;
                    } else {
                        pf[idx] = k;
                        pf[fstrides[0] + idx] = j;
                        pf[fstrides[0] * 2 + idx] = i;
                    }
                }
                int cur_thread_id = omp_get_thread_num();
                coor_local[cur_thread_id][0] = 0;
                coor_local[cur_thread_id][1] = 0;
                coor_local[cur_thread_id][2] = 0;
                VoronoiFT(pf + i * fstrides[3] + j * fstrides[2], ishape[0], coor_local[cur_thread_id].data(), rank, 0,
                          fstrides[1], fstrides[0], local_f_ptrs[cur_thread_id].data(), local_g[cur_thread_id].data());
            }
        }

// the second dimension
#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[2]; i++) {
            for (int j = 0; j < ishape[0]; j++) {
                int cur_thread_id = omp_get_thread_num();
                coor_local[cur_thread_id][0] = j;
                coor_local[cur_thread_id][1] = 0;
                coor_local[cur_thread_id][2] = i;
                VoronoiFT(pf + i * fstrides[3] + j * fstrides[1], ishape[1], coor_local[cur_thread_id].data(), rank, 1,
                          fstrides[2], fstrides[0], local_f_ptrs[cur_thread_id].data(), local_g[cur_thread_id].data());
            }
        }

// the first dimension
#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[1]; i++) {
            for (int j = 0; j < ishape[0]; j++) {
                int cur_thread_id = omp_get_thread_num();
                coor_local[cur_thread_id][0] = j;
                coor_local[cur_thread_id][1] = i;
                coor_local[cur_thread_id][2] = 0;
                VoronoiFT(pf + i * fstrides[2] + j * fstrides[1], ishape[2], coor_local[cur_thread_id].data(), rank, 2,
                          fstrides[3], fstrides[0], local_f_ptrs[cur_thread_id].data(), local_g[cur_thread_id].data());
            }
        }
    }
}

/* Exact euclidean feature transform, as described in: C. R. Maurer,
     Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing
     exact euclidean distance transforms of binary images in arbitrary
     dimensions. IEEE Trans. PAMI 25, 265-270, 2003. */

template <typename T, typename T_int>
std::tuple<std::vector<T>, std::vector<size_t>> calculate_distance_and_index(T_int *festures, int *index_strides, int N,
                                                                             int *feature_dims) {
    size_t size = 1;
    for (int i = 0; i < N; i++) {
        size *= feature_dims[i];
    }

    std::vector<T> distance(size, 0);
    std::vector<size_t> indexes(size, 0);
    T *distance_pos = distance.data();
    size_t *index_pos = indexes.data();
    if (N == 2) {
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                T_int x = festures[i * index_strides[1] + j * index_strides[2]];
                T_int y = festures[i * index_strides[1] + j * index_strides[2] + index_strides[0]];
                double dist = (x - i) * (x - i) + (y - j) * (y - j);
                *distance_pos = sqrt(dist);
                size_t index = x * index_strides[1] + y * index_strides[2];
                *index_pos = index;
                index_pos++;
                distance_pos++;
            }
        }
    } else if (N == 3) {
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                for (int k = 0; k < feature_dims[2]; k++) {
                    T_int x = festures[i * index_strides[1] + j * index_strides[2] + k * index_strides[3]];
                    T_int y =
                        festures[i * index_strides[1] + j * index_strides[2] + k * index_strides[3] + index_strides[0]];
                    T_int z = festures[i * index_strides[1] + j * index_strides[2] + k * index_strides[3] +
                                       index_strides[0] * 2];
                    // std::cout << "i = " << i << " j = " << j << " k = " << k << std::endl;
                    // std::cout <<  "x = " << x << " y = " << y << " z = " << z << std::endl;
                    double dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                    size_t index = x * index_strides[1] + y * index_strides[2] + z * index_strides[3];
                    *index_pos = index;
                    *distance_pos = sqrt(dist);
                    distance_pos++;
                    index_pos++;
                }
            }
        }
    }
    // make a pair of the two vectors
    printf("distance address  = %ld \n", &distance[0]);
    printf("index address  = %ld \n", &indexes[0]); 
    std::tuple<std::vector<T>, std::vector<size_t>> result = std::make_tuple(std::move(distance), std::move(indexes));
    return result;
    // return distance;
}

template <typename T, typename T_int>
std::vector<T> calculate_distance(T_int *festures, int *index_strides, int N, int *feature_dims) {
    size_t size = 1;
    for (int i = 0; i < N; i++) {
        size *= feature_dims[i];
    }

    std::vector<T> distance(size, 0);
    Timer timer; 
    timer.start();
    if (N == 2) {
        #pragma omp parallel for collapse(2) num_threads(this->num_threads)
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                T_int x = festures[i * index_strides[1] + j * index_strides[2]];
                T_int y = festures[i * index_strides[1] + j * index_strides[2] + index_strides[0]];
                double dist = (x - i) * (x - i) + (y - j) * (y - j);
                size_t index = x * index_strides[1] + y * index_strides[2]; 
                distance[index] = sqrt(dist); 
            }
        }
    } else if (N == 3) {
        #pragma omp parallel for collapse(3) num_threads(this->num_threads)
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                for (int k = 0; k < feature_dims[2]; k++) {
                    size_t index = i * index_strides[1] + j * index_strides[2] + k * index_strides[3]; 
                    T_int x = festures[index];
                    T_int y = festures[index + index_strides[0]];
                    T_int z = festures[index+index_strides[0] * 2];
                    double dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                    distance[index] = sqrt(dist);
                }
            }
        }
    }
    distance_time = timer.stop(); 
    printf("distacne address  = %ld \n", &distance[0]); 
    return distance;
}

int NI_EuclideanFeatureTransform(char *input, int *features, int N, int *dims, int num_threads = 64) {
    int ii;
    npy_intp coor[NPY_MAXDIMS], mx = 0, jj;
    npy_intp *tmp = NULL, **f = NULL, *g = NULL;
    char *pi;
    int *pf;

    pi = (input);
    pf = (features);
    for (ii = 0; ii < N; ii++) {
        coor[ii] = 0;
        if (dims[ii] > mx) {
            mx = dims[ii];
        }
    }

    std::vector<int> strides(N);
    std::vector<int> index_strides(N + 1);
    strides[N - 1] = 1;
    index_strides[N] = 1;

    if (N == 2) {
        strides[0] = dims[1];
        index_strides[1] = dims[1];
        index_strides[0] = dims[0] * dims[1];
    } else if (N == 3) {
        strides[1] = dims[2];
        strides[0] = dims[1] * dims[2];
        index_strides[2] = dims[2];
        index_strides[1] = dims[1] * dims[2];
        index_strides[0] = dims[0] * dims[1] * dims[2];

    } else {
        exit(0);
    }

    /* Some temporaries */
    // f = (npy_intp **)malloc(mx * sizeof(npy_intp *));
    // g = (npy_intp *)malloc(mx * sizeof(npy_intp));
    // tmp = (npy_intp *)malloc(mx * N * sizeof(npy_intp));
    // for (jj = 0; jj < mx; jj++) {
    //     f[jj] = tmp + jj * N;
    // }

    /* First call of recursive feature transform */
    auto timer = Timer();
    timer.start();
    ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N, f, g);
    edt_time = timer.stop();

    // calculate_distance<int, int>(features, index_strides.data(), N, dims);
    //  auto distance = calculate_distance<double, int> (features, index_strides.data(), N, dims);
    //  for(int i = 0; i < distance.size(); i++)
    //  {
    //      std::cout << "distance[" << i << "] = " << distance[i] << std::endl;
    //  }

    // free(f);
    // free(g);
    // free(tmp);

    return 0;
}

template <typename T, typename T_int>
std::tuple<std::vector<T>, std::vector<size_t>> NI_EuclideanFeatureTransform(char *input, int N, int *dims,
                                                                             int num_threads = 64) {
    int ii;
    npy_intp coor[NPY_MAXDIMS], mx = 0, jj;
    npy_intp *tmp = NULL, **f = NULL, *g = NULL;
    char *pi;
    int *pf;
    size_t input_size = 1;

    pi = (input);

    for (ii = 0; ii < N; ii++) {
        coor[ii] = 0;
        if (dims[ii] > mx) {
            mx = dims[ii];
        }
        input_size *= dims[ii];
    }
    std::vector<int> features(input_size * N, 0);
    pf = features.data();

    std::vector<int> strides(N);
    std::vector<int> index_strides(N + 1);
    strides[N - 1] = 1;
    index_strides[N] = 1;

    if (N == 2) {
        strides[0] = dims[1];
        index_strides[1] = dims[1];
        index_strides[0] = dims[0] * dims[1];
    } else if (N == 3) {
        strides[1] = dims[2];
        strides[0] = dims[1] * dims[2];
        index_strides[2] = dims[2];
        index_strides[1] = dims[1] * dims[2];
        index_strides[0] = dims[0] * dims[1] * dims[2];

    } else {
        exit(0);
    }

    /* Some temporaries */
    // f = (npy_intp **)malloc(mx * sizeof(npy_intp *));
    // g = (npy_intp *)malloc(mx * sizeof(npy_intp));
    // tmp = (npy_intp *)malloc(mx * N * sizeof(npy_intp));

    // for (jj = 0; jj < mx; jj++) {
    //     f[jj] = tmp + jj * N;
    // }

    /* First call of recursive feature transform */
    auto timer = Timer();
    timer.start();
    ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N, f, g);
    edt_time = timer.stop();
    // std::cout << "_VoronoiFT_time = " << _VoronoiFT_time << std::endl;

    // free(f);
    // free(g);
    // free(tmp);
    return calculate_distance_and_index<T, T_int>(features.data(), index_strides.data(), N, dims);
}


private:
    /* data */
    Timer timer; 
    int num_threads = 1;
    double edt_time = 0;
    double distance_time = 0; 




};
} // namespace PM2

#endif  // EDT_TRANSFORM_HPP