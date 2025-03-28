#ifndef QCAT_SSIM_MPI_HPP
#define QCAT_SSIM_MPI_HPP
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "utils/file_utils.hpp" 

#define K1 0.01
#define K2 0.03

template <class T>
int SSIM_3d_calcWindow_(T *data, T *other, size_t size1, size_t size0, int offset0, int offset1, int offset2,
                          int windowSize0, int windowSize1, int windowSize2, int cur_count, T* buffer) {
    int i0, i1, i2, index;
    int np = 0;  // Number of points
    for (i2 = offset2; i2 < offset2 + windowSize2; i2++) {
        for (i1 = offset1; i1 < offset1 + windowSize1; i1++) {
            for (i0 = offset0; i0 < offset0 + windowSize0; i0++) {
                index = i0 + size0 * (i1 + size1 * i2);
                buffer[np] = data[index]; 
                np++; 
            }
        }
    }
    return np;
}

template <class T>
int  SSIM_3d_windowed_(T *oriData, T *decData, size_t size2, size_t size1, size_t size0, int windowSize0,
                        int windowSize1, int windowSize2, int windowShift0, int windowShift1, int windowShift2, const char* prefix ) {
    std::vector<T> oriData_win(size2 * size1 * size0, 0);
    std::vector<T> decData_win(size2 * size1 * size0, 0);

    int offset0, offset1, offset2;
    int nw = 0;  // Number of windows
    double ssimSum = 0;
    int offsetInc0, offsetInc1, offsetInc2;

    if (windowSize0 > size0) {
        printf("ERROR: windowSize0 = %d > %zu\n", windowSize0, size0);
    }
    if (windowSize1 > size1) {
        printf("ERROR: windowSize1 = %d > %zu\n", windowSize1, size1);
    }
    if (windowSize2 > size2) {
        printf("ERROR: windowSize2 = %d > %zu\n", windowSize2, size2);
    }

    // offsetInc0=windowSize0/2;
    // offsetInc1=windowSize1/2;
    // offsetInc2=windowSize2/2;
    offsetInc0 = windowShift0;
    offsetInc1 = windowShift1;
    offsetInc2 = windowShift2;

    size_t max_offset2 = size2 - windowSize2;
    size_t max_offset1 = size1 - windowSize1;
    size_t max_offset0 = size0 - windowSize0;

    std::vector<int> nps; 
    std::vector<T> buffer(windowSize0 * windowSize1 * windowSize2, 0);
    char filename[1024]; 

    for (size_t offset2 = 0; offset2 <= max_offset2; offset2 += offsetInc2) {
        for (size_t offset1 = 0; offset1 <= max_offset1; offset1 += offsetInc1) {
            for (size_t offset0 = 0; offset0 <= max_offset0; offset0 += offsetInc0) {
                int x = SSIM_3d_calcWindow_(oriData, decData, size1, size0,
                                            offset0, offset1, offset2,
                                            windowSize0, windowSize1, windowSize2, nw, buffer.data());
                nps.push_back(x);
                sprintf(filename, "%s_%d_%d_%d.ssim.f32", prefix, offset0, offset1, offset2);
                printf("filename = %s\n", filename);
                writefile(filename, buffer.data(), x);
                nw++;
            }
        }
    }
    printf("nw = %d\n", nw);
    return (double) nw;
}

template <class T>
double calculateSSIM_(T *oriData, T *decData, int dim, size_t *dims, const char* prefix) {
    int windowSize0 = 7;
    int windowSize1 = 7;
    int windowSize2 = 7;
    int windowSize3 = 7;
    int windowShift0 = 2;
    int windowShift1 = 2;
    int windowShift2 = 2;
    int windowShift3 = 2;
    double result = -1;

    switch (dim) {
        case 1:
            result = 0;
            break;
        case 2:
            result = 0;
            break;
        case 3:
            result = SSIM_3d_windowed_(oriData, decData, dims[0], dims[1], dims[2], windowSize0, windowSize1,
                                      windowSize2, windowShift0, windowShift1, windowShift2, prefix );
            break;
        case 4:
            result = 0;
            break;
    }
    return result;
}

#endif  // QCAT_SSIM_MPI_HPP