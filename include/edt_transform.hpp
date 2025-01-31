#ifndef EDT_TRANSFORM_HPP
#define EDT_TRANSFORM_HPP

#include <cstddef>
#include<vector> 
#include<iostream>
#include<cmath>
#include <tuple>
#include "utils/timer.hpp"


namespace PM{


using npy_intp = int;
using npy_uint32 = unsigned int;
using npy_int32 = int;
using npy_double = double; 
using npy_int8 = char; 
#define NPY_MAXDIMS 5


void print_feature(int* vec)
{
    std::cout << "\n================" << std::endl;
    for(int i = 0; i < 3*3*3*3; i++)
    {
        if(i%3 == 0 && i != 0)
        {
            std::cout << std::endl;
        }
        std::cout << vec[i] << " ";

    }
    std::cout << "\n================" << std::endl;
}



typedef struct {
    int rank_m1;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp backstrides[NPY_MAXDIMS];
} NI_Iterator;

/* initialize iterations over single array elements: */
int NI_InitPointIterator(NI_Iterator *iterator, int N, int* dims, const int* strides )
{
    int ii;

    iterator->rank_m1 = N - 1;
    for(ii = 0; ii < N; ii++) {
        /* adapt dimensions for use in the macros: */
        iterator->dimensions[ii] = dims[ii] - 1;
        /* initialize coordinates: */
        iterator->coordinates[ii] = 0;
        /* initialize strides: */
        iterator->strides[ii] = strides[ii];
        /* calculate the strides to move back at the end of an axis: */
        iterator->backstrides[ii] =
                strides[ii] * iterator->dimensions[ii];
    }
    return 1;
}

/* go to the next point in a single array */
#define NI_ITERATOR_NEXT(iterator, pointer)                         \
{                                                                   \
    int _ii;                                                          \
    for(_ii = (iterator).rank_m1; _ii >= 0; _ii--)                    \
        if ((iterator).coordinates[_ii] < (iterator).dimensions[_ii]) { \
            (iterator).coordinates[_ii]++;                                \
            pointer += (iterator).strides[_ii];                           \
            break;                                                        \
        } else {                                                        \
            (iterator).coordinates[_ii] = 0;                              \
            pointer -= (iterator).backstrides[_ii];                       \
        }                                                               \
}


/* initialize iteration over a lower sub-space: */
int NI_SubspaceIterator(NI_Iterator *iterator, npy_uint32 axes)
{
    int ii, last = 0;
    for(ii = 0; ii <= iterator->rank_m1; ii++) {
        if (axes & (((npy_uint32)1) << ii)) {
            if (last != ii) {
                iterator->dimensions[last] = iterator->dimensions[ii];
                iterator->strides[last] = iterator->strides[ii];
                iterator->backstrides[last] = iterator->backstrides[ii];
            }
            ++last;
        }
    }
    iterator->rank_m1 = last - 1;
    return 1;
}



double _VoronoiFT_time =0; 

static void _VoronoiFT(int *pf, npy_intp len, npy_intp *coor, int rank,
                       int d, npy_intp stride, npy_intp cstride,
                       npy_intp **f, npy_intp *g, const npy_double *sampling)
{
    npy_intp l = -1, ii, maxl, idx1, idx2;
    npy_intp jj;

    // auto timer = Timer();
    // timer.start();

    // for(ii = 0; ii < len; ii++)
    //     for(jj = 0; jj < rank; jj++)
    //         f[ii][jj] = *(pf + ii * stride + cstride * jj); // ctrisde = data_size

    for(jj = 0; jj < rank; jj++)
        for(ii = 0; ii < len; ii++)
            f[ii][jj] = *(pf + ii * stride + cstride * jj); // ctrisde = data_size

    
    // _VoronoiFT_time += timer.stop(); 
    for(ii = 0; ii < len; ii++) {
        if (*(pf + ii * stride) >= 0) {
            double fd = f[ii][d];
            double wR = 0.0;
            for(jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    double tw = f[ii][jj] - coor[jj];

                    wR += tw * tw;
                }
            }
            while(l >= 1) {
                double a, b, c, uR = 0.0, vR = 0.0, f1;
                idx1 = g[l];
                f1 = f[idx1][d];
                idx2 = g[l - 1];
                a = f1 - f[idx2][d];
                b = fd - f1;

                c = a + b;
                for(jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f[idx2][jj] - cc;
                        double tv = f[idx1][jj] - cc;
                        // if (sampling) {
                        //     tu *= sampling[jj];
                        //     tv *= sampling[jj];
                        // }
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0.0)
                    break;
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
            for(jj = 0; jj < rank; jj++) {
                t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];

                delta1 += t * t;
            }
            while (l < maxl) {
                double delta2 = 0.0;
                for(jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];

                    delta2 += t * t;
                }
                if (delta1 <= delta2)
                    break;
                delta1 = delta2;
                ++l;
            }
            idx1 = g[l];
            for(jj = 0; jj < rank; jj++)
                *(pf + ii * stride + jj * cstride) = f[idx1][jj];
        }
    }
}


/* Recursive feature transform */
static void _ComputeFT(char *pi, int *pf, npy_intp *ishape,
                       const npy_intp *istrides, const npy_intp *fstrides,
                       int rank, int d, npy_intp *coor, npy_intp **f,
                       npy_intp *g, int* features,
                       const npy_double *sampling = NULL)
{
    npy_intp kk;
    npy_intp jj;

    if (d == 0) {
    int *tf1 = pf;
        for(jj = 0; jj < ishape[0]; jj++) {
            if (*pi) {
                *tf1 = -1;
            } else {
                int *tf2 = tf1;
                *tf2 = jj;
                for(kk = 1; kk < rank; kk++) {
                    tf2 += fstrides[0];
                    *tf2 = coor[kk];
                }
            }
            pi += istrides[0];
            tf1 += fstrides[1];
        }
        _VoronoiFT(pf, ishape[0], coor, rank, 0, fstrides[1], fstrides[0], f,
                             g, sampling);
    } else {
        
        npy_uint32 axes = 0;
        int *tf = pf;
        npy_intp size = 1;
        NI_Iterator iter;

        for(jj = 0; jj < ishape[d]; jj++) {
            coor[d] = jj;
            _ComputeFT(pi, tf, ishape, istrides, fstrides, rank, d - 1, coor, f,
                                 g, features, sampling);
            
            pi += istrides[d];
            tf += fstrides[d + 1];
        }

        for(jj = 0; jj < d; jj++) {
            axes |= (npy_uint32)1 << (jj + 1);
            size *= ishape[jj];
        }

        std::vector<npy_intp> feature_shapes(rank+1);
        feature_shapes[0] = rank;

        for (int i = 1; i < rank+1; i++)
        {
            feature_shapes[i] = ishape[i-1];
        }

        NI_InitPointIterator(&iter, rank+1, feature_shapes.data(), fstrides);
        NI_SubspaceIterator(&iter, axes);
        tf = pf;
        for(jj = 0; jj < size; jj++) {
            for(kk = 0; kk < d; kk++)
            {  
                coor[kk] = iter.coordinates[kk];
            }
            _VoronoiFT(tf, ishape[d], coor, rank, d, fstrides[d + 1],
                                 fstrides[0], f, g, sampling);
            
            NI_ITERATOR_NEXT(iter, tf);
        }
        for(kk = 0; kk < d; kk++)
        {
            coor[kk] = 0;
        }
    }
}

/* Exact euclidean feature transform, as described in: C. R. Maurer,
     Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing
     exact euclidean distance transforms of binary images in arbitrary
     dimensions. IEEE Trans. PAMI 25, 265-270, 2003. */


template <typename T, typename T_int>
std::tuple<std::vector<T>, std::vector<size_t>>  calculate_distance_and_index(T_int* festures,  int* index_strides, 
                                                                                    int N, int* feature_dims)
{  
    auto timer = Timer(); 
    timer.start();
    size_t size = 1;
    for(int i = 0; i < N; i++)
    {
        size *= feature_dims[i];
    }

    std::vector<T> distance(size, 0);
    std::vector<size_t> indexes(size, 0);
    T* distance_pos = distance.data();
    size_t* index_pos = indexes.data();  
    if(N ==2)
    {
        for(int i = 0; i < feature_dims[0]; i++)
        {
            for(int j = 0; j < feature_dims[1] ; j++)
            {
                T_int x = festures[i*index_strides[1] + j*index_strides[2] ];
                T_int y = festures[i*index_strides[1] + j*index_strides[2] + index_strides[0]];
                double dist = (x - i)*(x - i) + (y - j)*(y - j);
                *distance_pos = sqrt(dist);
                size_t index = x*index_strides[1] + y*index_strides[2];
                *index_pos = index;
                index_pos++;
                distance_pos++;
            }
        }
    }
    else if(N == 3)
    {
        for (int i =0; i < feature_dims[0]; i++)
        {
            for(int j =0; j < feature_dims[1]; j++)
            {
                for (int k = 0; k< feature_dims[2]; k++)
                {
                    T_int x = festures[i*index_strides[1] + j*index_strides[2] + k*index_strides[3]];
                    T_int y = festures[i*index_strides[1] + j*index_strides[2] + k*index_strides[3] + index_strides[0]];
                    T_int z = festures[i*index_strides[1] + j*index_strides[2] + k*index_strides[3] + index_strides[0]*2];
                    // std::cout << "i = " << i << " j = " << j << " k = " << k << std::endl;
                    // std::cout <<  "x = " << x << " y = " << y << " z = " << z << std::endl;
                    double dist = (x - i)*(x - i) + (y - j)*(y - j) + (z - k)*(z - k);
                    size_t index = x*index_strides[1] + y*index_strides[2] + z*index_strides[3];
                    *index_pos = index;
                    *distance_pos = sqrt(dist);
                    distance_pos++;   
                    index_pos++;             
                }
            }
        }
    }
    // make a pair of the two vectors
    std::tuple<std::vector<T>, std::vector<size_t>> result = std::make_tuple(distance, indexes); 
    // std::cout << "distance time = " << timer.stop() << std::endl;
    printf("distance time = %.10f \n", timer.stop()); 
    return    result; 
    // return distance;
}


template <typename T, typename T_int>
std::vector<T> calculate_distance(T_int* festures,  int* index_strides, int N, int* feature_dims)
{  
    size_t size = 1;
    for(int i = 0; i < N; i++)
    {
        size *= feature_dims[i];
    }

    std::vector<T> distance(size, 0);
    T* distance_pos = distance.data();
    if(N ==2)
    {
        for(int i = 0; i < feature_dims[0]; i++)
        {
            for(int j = 0; j < feature_dims[1] ; j++)
            {
                T_int x = festures[i*index_strides[1] + j*index_strides[2] ];
                T_int y = festures[i*index_strides[1] + j*index_strides[2] + index_strides[0]];
                double dist = (x - i)*(x - i) + (y - j)*(y - j);
                *distance_pos = sqrt(dist);
                distance_pos++;
            }
        }
    }
    else if(N == 3)
    {
        for (int i =0; i < feature_dims[0]; i++)
        {
            for(int j =0; j < feature_dims[1]; j++)
            {
                for (int k = 0; k< feature_dims[2]; k++)
                {
                    T_int x = festures[i*index_strides[1] + j*index_strides[2] + k*index_strides[3]];
                    T_int y = festures[i*index_strides[1] + j*index_strides[2] + k*index_strides[3] + index_strides[0]];
                    T_int z = festures[i*index_strides[1] + j*index_strides[2] + k*index_strides[3] + index_strides[0]*2];
                    // std::cout << "i = " << i << " j = " << j << " k = " << k << std::endl;
                    // std::cout <<  "x = " << x << " y = " << y << " z = " << z << std::endl;
                    double dist = (x - i)*(x - i) + (y - j)*(y - j) + (z - k)*(z - k);
                    *distance_pos = sqrt(dist);
                    distance_pos++;   
                }
            }
        }
    }
    return distance;
}


int NI_EuclideanFeatureTransform(char* input,  int* features, int N, int *dims)
{
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
    std::vector<int> index_strides(N+1);
    strides[N-1] = 1;
    index_strides[N] = 1;
    
    if(N==2)
    {
        strides[0] = dims[1];
        index_strides[1] = dims[1];
        index_strides[0] = dims[0]*dims[1];
    }
    else if (N==3)
    {
        strides[1] = dims[2];
        strides[0] = dims[1]*dims[2];
        index_strides[2] = dims[2];
        index_strides[1] = dims[1]*dims[2];
        index_strides[0] = dims[0]*dims[1]*dims[2];

    }
    else
    {
        exit(0);
    }


    /* Some temporaries */
    f = (npy_intp**)malloc(mx * sizeof(npy_intp*));
    g = (npy_intp*)malloc(mx * sizeof(npy_intp));
    tmp = (npy_intp*) malloc(mx * N * sizeof(npy_intp));

    for(jj = 0; jj < mx; jj++) {
        f[jj] = tmp + jj * N;
    }

    /* First call of recursive feature transform */
    auto timer = Timer(); 
    timer.start();
    _ComputeFT(pi, pf, dims, strides.data(),
               index_strides.data(), N,
               N - 1, coor, f, g, features, 0);
    std::cout << "edt time = "  << timer.stop() << std::endl;

    
    //calculate_distance<int, int>(features, index_strides.data(), N, dims);
    // auto distance = calculate_distance<double, int> (features, index_strides.data(), N, dims);
    // for(int i = 0; i < distance.size(); i++)
    // {
    //     std::cout << "distance[" << i << "] = " << distance[i] << std::endl;
    // }

    free(f);
    free(g);
    free(tmp);

    return 0;
}


template <typename T, typename T_int>
std::tuple<std::vector<T>, std::vector<size_t>> NI_EuclideanFeatureTransform(char* input, int N, int *dims)
{
    auto timer  = Timer(); 
    timer.start();
    int ii;
    npy_intp coor[NPY_MAXDIMS], mx = 0, jj;
    npy_intp *tmp = NULL, **f = NULL, *g = NULL;
    char *pi;
    int *pf;
    size_t input_size = 1 ;

    pi = (input);

    for (ii = 0; ii < N; ii++) {
        coor[ii] = 0;
        if (dims[ii] > mx) {
            mx = dims[ii];
        }
        input_size *= dims[ii]; 
    }
    std::vector<int> features(input_size*N, 0);
    pf = features.data();

    std::vector<int> strides(N);
    std::vector<int> index_strides(N+1);
    strides[N-1] = 1;
    index_strides[N] = 1;
    
    if(N==2)
    {
        strides[0] = dims[1];
        index_strides[1] = dims[1];
        index_strides[0] = dims[0]*dims[1];
    }
    else if (N==3)
    {
        strides[1] = dims[2];
        strides[0] = dims[1]*dims[2];
        index_strides[2] = dims[2];
        index_strides[1] = dims[1]*dims[2];
        index_strides[0] = dims[0]*dims[1]*dims[2];

    }
    else
    {
        exit(0);
    }


    /* Some temporaries */
    f = (npy_intp**)malloc(mx * sizeof(npy_intp*));
    g = (npy_intp*)malloc(mx * sizeof(npy_intp));
    tmp = (npy_intp*) malloc(mx * N * sizeof(npy_intp));

    for(jj = 0; jj < mx; jj++) {
        f[jj] = tmp + jj * N;
    }
    std::cout << "aux time = " << timer.stop() << std::endl;

    /* First call of recursive feature transform */
    
    timer.start();
    _ComputeFT(pi, pf, dims, strides.data(),
               index_strides.data(), N,
               N - 1, coor, f, g, features.data(), 0);
    std::cout << "edt time = "  << timer.stop() << std::endl;


    free(f);
    free(g);
    free(tmp);
    return calculate_distance_and_index<T, T_int>(features.data(), index_strides.data(), N, dims);
}







}

#endif // EDT_TRANSFORM_HPP