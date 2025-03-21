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
#include "utils/file_utils.hpp"

namespace PM2 {

using npy_intp = int;
using npy_uint32 = unsigned int;
using npy_int32 = int;
using npy_double = double;
using npy_int8 = char;
#define NPY_MAXDIMS 5

template<typename T_distance, typename T_int>
class EDT_OMP{


struct Distance_and_Index {
    std::unique_ptr<T_distance[]> distance;
    std::unique_ptr<size_t[]> indexes;
};

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
    for (ii = 0; ii < len; ii++) 
        for (jj = 0; jj < rank; jj++)
            f[ii][jj] = *(pf + ii * stride + cstride * jj); 

    for (ii = 0; ii < len; ii++) {
        if (*(pf + ii * stride) >= 0) {
            int fd = f[ii][d];
            int wR = 0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    int tw = f[ii][jj] - coor[jj];
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                int a, b, c, uR = 0, vR = 0, f1;
                idx1 = g[l];
                f1 = f[idx1][d];
                idx2 = g[l - 1];
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
                if (c * vR - b * uR - a * wR - a * b * c <= 0 ) break;
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
            int delta1 = 0, t;
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
            for (jj = 0; jj < rank; jj++) *(pf + ii * stride + jj * cstride) = f[idx1][jj];
        }
    }
}


int fn(int x , int i, int* g )
{
    return (x - i)*(x-i) + g[i]*g[i]; 
}
int Sep(int u , int i, int* g )
{
    return u*u - i*i + g[u]*g[u] - g[i]*g[i]; 
}

// in this case pf will have the same shape as the input image
void VoronoiFT_distance(int *pf, npy_intp len, npy_intp *coor, int rank,
                     int d, npy_intp stride, npy_intp cstride,
                      npy_intp *f, npy_intp *g) {
    int q = 0;
    f[0] = 0;
    g[0] = 0; 
    int ii;
    for (int ii = 0; ii < len; ii++) 
            f[ii] = *(pf + ii * stride ); 

    for ( ii =1; ii< len; ii++)
    {
        while (q >= 0 && fn(f[q], g[q], g) > fn(f[q], ii, g))
        {
            q--;
        }
        if (q < 0)
        {
            q = 0;
            f[0] = ii;
        }
        else
        {
           int w = 1 + Sep(ii, f[q], g) ; 
           if (w < len)
           {
               q++;
               f[q] = ii;
               g[q] = w;
           }
        }
    }
    for (ii = len-1; ii >= 0; ii--)
    {
        int delta = (ii - f[q])*(ii - f[q]) + g[q]*g[q];
        while (q > 0 && (ii - f[q-1])*(ii - f[q-1]) + g[q-1]*g[q-1] < delta)
        {
            q--;
            delta = (ii - f[q])*(ii - f[q]) + g[q]*g[q];
        }
        *(pf + ii * stride) = delta;
    }

}



void ComputeFT2D3D(char *pi, int *pf, npy_intp *ishape, const npy_intp *istrides, const npy_intp *fstrides,
                          int rank) {
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

        
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < ishape[1]; i++) {
            int thread_id = omp_get_thread_num(); 
            coor_local[thread_id][1] = i;
            for (int j = 0; j < ishape[0]; j++) {
                size_t idx = i * istrides[1] + j * istrides[0];
                if (pi[idx]) { // non-boundary points 
                    pf[idx*2] = -1;
                } else {
                    pf[idx*2] = j;
                    pf[idx*2 +1] = i;
                }
            }
            VoronoiFT(pf + i * fstrides[1], 
            ishape[0], coor_local[thread_id].data(), rank, 0, 
            fstrides[0], fstrides[2], local_f_ptrs[thread_id].data(), 
                                                            local_g[thread_id].data());
        }
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < ishape[0]; i++) {
            int thread_id = omp_get_thread_num(); 
            coor_local[thread_id][0] = i;
            VoronoiFT(pf + i * fstrides[0], ishape[1], coor_local[thread_id].data(), rank, 
                        1, fstrides[1], fstrides[2], local_f_ptrs[thread_id].data(), 
                                                            local_g[thread_id].data());
        }

    } else if (rank == 3) {

        omp_set_num_threads(num_threads);
        int max_dim = std::max(ishape[0], std::max(ishape[1], ishape[2]));
        // use malloc to allocate the memory 
        npy_intp* local_f = (npy_intp*) malloc(num_threads * max_dim * 3 * sizeof(npy_intp)); 
        npy_intp* local_g = (npy_intp*) malloc(num_threads * max_dim * sizeof(npy_intp));
        npy_intp** local_f_ptrs = (npy_intp**) malloc(num_threads * max_dim * sizeof(npy_intp*));
        npy_intp* coor_locals = (npy_intp*) malloc(num_threads * 3 * sizeof(npy_intp));
        for (int i = 0; i < num_threads; i++) {
            for (int j = 0; j < max_dim; j++) {
                local_f_ptrs[i * max_dim + j] = local_f + i * max_dim * 3 + j * 3;
            }
        }
        printf("cache line allocation time = %f\n", global_timer.stop());


#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[2]; i++)  // 384
        {
            for (int j = 0; j < ishape[1]; j++)  // 384
            {
                for (int k = 0; k < ishape[0]; k++) {
                    size_t idx = i * istrides[2] + j * istrides[1] + k * istrides[0];
                    if (pi[idx]) {
                        pf[idx*3] = -1;
                    } else {
                        pf[idx*3] = k;
                        pf[fstrides[3] + idx*3] = j;
                        pf[fstrides[3] * 2 + idx*3] = i;
                    }
                }
                int cur_thread_id = omp_get_thread_num();
                npy_intp*  coor_local = coor_locals + cur_thread_id * 3;
                coor_local[0] = 0;
                coor_local[1] = j;
                coor_local[2] = i;
                VoronoiFT(pf + i * fstrides[2] + j * fstrides[1], ishape[0], 
                     coor_local, rank, 0,
                          fstrides[0], fstrides[3], 
                          local_f_ptrs+max_dim*cur_thread_id, local_g+cur_thread_id*max_dim);
            }
        }

// the second dimension
#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[2]; i++) {
            for (int j = 0; j < ishape[0]; j++) {
                int cur_thread_id = omp_get_thread_num();
                npy_intp*  coor_local = coor_locals + cur_thread_id * 3;
                coor_local[0] = j;
                coor_local[1] = 0;
                coor_local[2] = i;
                VoronoiFT(pf + i * fstrides[2] + j * fstrides[0], ishape[1],
                     coor_local, rank, 1,
                          fstrides[1], fstrides[3], 
                          local_f_ptrs+max_dim*cur_thread_id, local_g+cur_thread_id*max_dim);
            }
        }

// the first dimension
#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[1]; i++) {
            for (int j = 0; j < ishape[0]; j++) {
                int cur_thread_id = omp_get_thread_num();
                npy_intp*  coor_local = coor_locals + cur_thread_id * 3;
                coor_local[0] = j;
                coor_local[1] = i;
                coor_local[2] = 0;
                VoronoiFT(pf + i * fstrides[1] + j * fstrides[0], ishape[2], 
                        coor_local, rank, 2,
                          fstrides[2], fstrides[3], 
                          local_f_ptrs+max_dim*cur_thread_id, local_g+cur_thread_id*max_dim);
            }
        }
        free(local_f_ptrs);
        free(local_f);
        free(local_g);
    }
}

/* Exact euclidean feature transform, as described in: C. R. Maurer,
     Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing
     exact euclidean distance transforms of binary images in arbitrary
     dimensions. IEEE Trans. PAMI 25, 265-270, 2003. */


std::tuple<std::vector<T_distance>, std::vector<size_t>> calculate_distance_and_index(T_int *festures, int *index_strides, int N,
                                                                             int *feature_dims) {
    
    size_t size = 1;
    for (int i = 0; i < N; i++) {
        size *= feature_dims[i];
    }
    global_timer.start();
    std::vector<T_distance> distance;
    distance.resize(size);
    std::vector<size_t> indexes;
    indexes.resize(size);

    printf("allocate distance time = %f\n", global_timer.stop());
    double dist = 0;
    size_t global_idx = 0;
    T_int x, y, z;
    global_timer.start();
    if (N == 2) {
        #pragma omp parallel for collapse(1) num_threads(this->num_threads) private(dist, global_idx, x, y)
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                global_idx = i * feature_dims[1] + j;
                x = festures[global_idx*2];
                y = festures[global_idx*2 + index_strides[2]];
                dist = (x - i) * (x - i) + (y - j) * (y - j);
                distance[global_idx] = sqrt(dist);
                indexes[global_idx] = x * feature_dims[1] + y ;
            }
        }
    } else if (N == 3) {
        size_t d1xd2 = feature_dims[1] * feature_dims[2];
        #pragma omp parallel for collapse(2) num_threads(this->num_threads) private(dist, global_idx, x, y, z)
        for (int i = 0; i < feature_dims[0]; i++) { 
            for (int j = 0; j < feature_dims[1]; j++) {
                for (int k = 0; k < feature_dims[2]; k++) {
                    global_idx = i * d1xd2 + j * feature_dims[2] + k; 
                    x = festures[global_idx*3 ];
                    y = festures[global_idx*3 + index_strides[3]];
                    z = festures[global_idx*3 + index_strides[3] * 2];
                    dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                    distance[global_idx] = sqrt(dist);
                    indexes[global_idx] = x * d1xd2 + y * feature_dims[2] + z;
                }
            }
        }
    }
    distance_time = global_timer.stop();
    return {std::move(distance), std::move(indexes)};;
}


Distance_and_Index calculate_distance_and_index_(T_int *festures, int *index_strides, int N,
                                                                             int *feature_dims) {
    
    size_t size = 1;
    for (int i = 0; i < N; i++) {
        size *= feature_dims[i];
    }
    global_timer.start();
    std::unique_ptr<T_distance[]> distance(static_cast<T_distance*>(std::malloc(size * sizeof(T_distance))));
    std::unique_ptr<size_t[]> indexes(static_cast<size_t*>(std::malloc(size * sizeof(size_t))));
    double dist = 0;
    size_t global_idx = 0;
    T_int x, y, z;
    if (N == 2) {
        #pragma omp parallel for collapse(1) num_threads(this->num_threads) private(dist, global_idx, x, y)
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                global_idx = i * feature_dims[1] + j;
                x = festures[global_idx*2];
                y = festures[global_idx*2 + index_strides[2]];
                dist = (x - i) * (x - i) + (y - j) * (y - j);
                distance[global_idx] = sqrt(dist);
                indexes[global_idx] = x * feature_dims[1] + y ;
            }
        }
    } else if (N == 3) {
        size_t d1xd2 = feature_dims[1] * feature_dims[2];
        #pragma omp parallel for collapse(1) num_threads(this->num_threads) private(dist, global_idx, x, y, z)
        for (int i = 0; i < feature_dims[0]; i++) { 
            for (int j = 0; j < feature_dims[1]; j++) {
                for (int k = 0; k < feature_dims[2]; k++) {
                    global_idx = i * d1xd2 + j * feature_dims[2] + k; 
                    x = festures[global_idx*3 ];
                    y = festures[global_idx*3 + index_strides[3]];
                    z = festures[global_idx*3 + index_strides[3] * 2];
                    dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                    distance[global_idx] = sqrt(dist);
                    indexes[global_idx] = x * d1xd2 + y * feature_dims[2] + z;
                }
            }
        }
    }
    distance_time = global_timer.stop();
    return {std::move(distance), std::move(indexes)};;
}



std::vector<T_distance> calculate_distance(T_int *festures, int *index_strides, int N, int *feature_dims) {
    size_t size = 1;
    for (int i = 0; i < N; i++) {
        size *= feature_dims[i];
    }

    std::vector<T_distance> distance(size);
    Timer timer; 
    timer.start();
    if (N == 2) {
        #pragma omp parallel for collapse(2) num_threads(this->num_threads)
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                size_t global_idx = i * feature_dims[1] + j;
                T_int x = festures[global_idx*2];
                T_int y = festures[global_idx*2 + index_strides[2]];
                double dist = (x - i) * (x - i) + (y - j) * (y - j);
                distance[global_idx] = sqrt(dist);
            }
        }
    } else if (N == 3) {
        #pragma omp parallel for collapse(3) num_threads(this->num_threads)
        for (int i = 0; i < feature_dims[0]; i++) {
            for (int j = 0; j < feature_dims[1]; j++) {
                for (int k = 0; k < feature_dims[2]; k++) {
                    size_t global_idx = i * feature_dims[1]* feature_dims[2] + j * feature_dims[2] + k; 
                    T_int x = festures[global_idx*3 ];
                    T_int y = festures[global_idx*3 + index_strides[3]];
                    T_int z = festures[global_idx*3 + index_strides[3] * 2];
                    double dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                    distance[global_idx] = sqrt(dist);
                }
            }
        }
    }
    distance_time = timer.stop(); 
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
        index_strides[1] = 2;
        index_strides[0] = dims[1]*2;
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
    global_timer.start();
    ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
    edt_time = global_timer.stop();

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

// do not use. the performance is terrible becase of the vector memory allocation 
std::tuple<std::vector<T_distance>, std::vector<size_t>> NI_EuclideanFeatureTransform_(char *input, int N, int *dims,
                                                                             int num_threads = 64) {
    global_timer.start();
    char *pi;
    int *pf;
    size_t input_size = 1;
    pi = (input);
    for (int ii = 0; ii < N; ii++) {
        input_size *= dims[ii];
    }
    // std::vector<int> features(input_size * N);
    int *features = (int *)malloc(input_size * N * sizeof(int)); 
    // pf = features.data();
    pf = features;
    std::vector<int> strides(N);
    std::vector<int> index_strides(N + 1);
    strides[N - 1] = 1;
    index_strides[N] = 1;
    if (N == 2) {
        strides[0] = dims[1];
        index_strides[0] = dims[1]*2;
        index_strides[1] = 2;
    } else if (N == 3) {
        strides[1] = dims[2];
        strides[0] = dims[1] * dims[2];
        index_strides[0] = dims[1] * dims[2]*3;
        index_strides[1] = dims[2] * 3;
        index_strides[2] = 3;
    } else {
        exit(0);
    }
    printf("aux time = %f \n", global_timer.stop());
    global_timer.start();
    ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
    edt_time = global_timer.stop();
    auto result = calculate_distance_and_index(features,index_strides.data(), N, dims); 
    free(features);
    return result;
}

Distance_and_Index NI_EuclideanFeatureTransform(char *input, int N, int *dims,
                                                    int num_threads = 64) {
    global_timer.start();
    char *pi;
    int *pf;
    size_t input_size = 1;
    pi = (input);
    for (int ii = 0; ii < N; ii++) {
        input_size *= dims[ii];
    }
    // std::vector<int> features(input_size * N);
    int *features = (int *)malloc(input_size * N * sizeof(int)); 
    // pf = features.data();
    pf = features;
    std::vector<int> strides(N);
    std::vector<int> index_strides(N + 1);
    strides[N - 1] = 1;
    index_strides[N] = 1;
    if (N == 2) {
        strides[0] = dims[1];
        index_strides[0] = dims[1]*2;
        index_strides[1] = 2;
    } else if (N == 3) {
        strides[1] = dims[2];
        strides[0] = dims[1] * dims[2];
        index_strides[0] = dims[1] * dims[2]*3;
        index_strides[1] = dims[2] * 3;
        index_strides[2] = 3;
    } else {
        exit(0);
    }
    printf("aux time = %f \n", global_timer.stop());
    global_timer.start();
    ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
    edt_time = global_timer.stop();
    auto result = calculate_distance_and_index_(features,index_strides.data(), N, dims); 
    free(features);
    return result;
}


private:
    /* data */
    Timer timer; 
    int num_threads = 1;
    double edt_time = 0;
    double distance_time = 0; 
    Timer global_timer; 




};
} // namespace PM2

#endif  // EDT_TRANSFORM_HPP