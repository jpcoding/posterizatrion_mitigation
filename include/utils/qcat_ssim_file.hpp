#ifndef SZ_QCATSSIM_HPP_FILE
#define SZ_QCATSSIM_HPP_FILE

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define K1 0.01
#define K2 0.03

// Code from Sheng Di's QCAT repository.

////////////////// 3D

template <class T>
double SSIM_3d_calcWindow_file(T *data, T *other, size_t size1, size_t size0, int offset0, int offset1, int offset2,
                               int windowSize0, int windowSize1, int windowSize2) {
    int i0, i1, i2, index;
    int np = 0;  // Number of points
    T xMin = data[offset0 + size0 * (offset1 + size1 * offset2)];
    T xMax = data[offset0 + size0 * (offset1 + size1 * offset2)];
    T yMin = other[offset0 + size0 * (offset1 + size1 * offset2)];
    T yMax = other[offset0 + size0 * (offset1 + size1 * offset2)];
    double xSum = 0;
    double ySum = 0;

    for (i2 = offset2; i2 < offset2 + windowSize2; i2++) {
        for (i1 = offset1; i1 < offset1 + windowSize1; i1++) {
            for (i0 = offset0; i0 < offset0 + windowSize0; i0++) {
                np++;
                index = i0 + size0 * (i1 + size1 * i2);
                if (xMin > data[index]) xMin = data[index];
                if (xMax < data[index]) xMax = data[index];
                if (yMin > other[index]) yMin = other[index];
                if (yMax < other[index]) yMax = other[index];
                xSum += data[index];
                ySum += other[index];
            }
        }
    }

    double xMean = xSum / np;
    double yMean = ySum / np;
    double var_x = 0, var_y = 0, var_xy = 0;

    for (i2 = offset2; i2 < offset2 + windowSize2; i2++) {
        for (i1 = offset1; i1 < offset1 + windowSize1; i1++) {
            for (i0 = offset0; i0 < offset0 + windowSize0; i0++) {
                index = i0 + size0 * (i1 + size1 * i2);
                var_x += (data[index] - xMean) * (data[index] - xMean);
                var_y += (other[index] - yMean) * (other[index] - yMean);
                var_xy += (data[index] - xMean) * (other[index] - yMean);
            }
        }
    }

    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma = sqrt(var_x);
    double ySigma = sqrt(var_y);
    double xyCov = var_xy;

    double c1, c2;
    if (xMax - xMin == 0) {
        c1 = K1 * K1;
        c2 = K2 * K2;
    } else {
        c1 = K1 * K1 * (xMax - xMin) * (xMax - xMin);
        c2 = K2 * K2 * (xMax - xMin) * (xMax - xMin);
    }
    double c3 = c2 / 2;

    double luminance = (2 * xMean * yMean + c1) / (xMean * xMean + yMean * yMean + c1);
    double contrast = (2 * xSigma * ySigma + c2) / (xSigma * xSigma + ySigma * ySigma + c2);
    double structure = (xyCov + c3) / (xSigma * ySigma + c3);
    double ssim = luminance * contrast * structure;

    return ssim;
}

template <class T>
double SSIM_3d_windowed_file(const char *oriData_file, const char *decData_file, size_t size2, size_t size1,
                             size_t size0, int windowSize0, int windowSize1, int windowSize2, int windowShift0,
                             int windowShift1, int windowShift2) {
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

    int num_threads = 64;
    omp_set_num_threads(num_threads);
    size_t max_offset2 = size2 - windowSize2;
    size_t max_offset1 = size1 - windowSize1;
    size_t max_offset0 = size0 - windowSize0;

    // std::vector<T> oriData(windowSize2 * windowSize1 * windowSize0);
    // std::vector<T> decData(windowSize2 * windowSize1 * windowSize0);

    std::vector<std::vector<T>> oriData(num_threads, std::vector<T>(windowSize2 * windowSize1 * windowSize0));
    std::vector<std::vector<T>> decData(num_threads, std::vector<T>(windowSize2 * windowSize1 * windowSize0));

    // Read the data for the current window

    // std::vector<std::ifstream> oriData_streams(num_threads);
    // std::vector<std::ifstream> decData_streams(num_threads);
    // for (int i = 0; i < num_threads; i++) {
    //     oriData_streams[i].open(oriData_file, std::ios::binary);
    //     if (!oriData_streams[i]) {
    //         std::cerr << "Error: Couldn't find the file: " << oriData_file << "\n";
    //         exit(0);
    //     }
    //     decData_streams[i].open(decData_file, std::ios::binary);
    //     if (!decData_streams[i]) {
    //         std::cerr << "Error: Couldn't find the file: " << decData_file << "\n";
    //         exit(0);
    //     }
    // }
    // // barrier
    int cur_thread = omp_get_thread_num();

#pragma omp parallel for collapse(3) num_threads(num_threads) reduction(+ : ssimSum, nw)
    for (size_t offset2 = 0; offset2 <= max_offset2; offset2 += offsetInc2) {
        for (size_t offset1 = 0; offset1 <= max_offset1; offset1 += offsetInc1) {
            for (size_t offset0 = 0; offset0 <= max_offset0; offset0 += offsetInc0) {
                nw++;
                // Create thread-local ifstream objects
                std::ifstream oriData_stream(oriData_file, std::ios::binary);
                std::ifstream decData_stream(decData_file, std::ios::binary);
                if (!oriData_stream || !decData_stream) {
                    #pragma omp critical
                    {
                        std::cerr << "Error: Couldn't find the file: " << oriData_file << " or " << decData_file
                                  << "\n";
                        exit(0);
                    }
                }

                // Read the data for the current window
                for (size_t i2 = 0; i2 < windowSize2; i2++) {
                    for (size_t i1 = 0; i1 < windowSize1; i1++) {
                        for (size_t i0 = 0; i0 < windowSize0; i0++) {
                            size_t index = i0 + windowSize0 * (i1 + windowSize1 * i2);
                            size_t oriIndex = offset0 + size0 * (offset1 + size1 * (offset2 + size2 * i2));
                            size_t decIndex = offset0 + size0 * (offset1 + size1 * (offset2 + size2 * i2));
                            oriData_stream.seekg(oriIndex * sizeof(T));
                            decData_stream.seekg(decIndex * sizeof(T));
                            oriData_stream.read(reinterpret_cast<char *>(&oriData[cur_thread][index]),
                                                             sizeof(T));
                            decData_stream.read(reinterpret_cast<char *>(&decData[cur_thread][index]),
                                                             sizeof(T));
                        }
                    }
                }
                ssimSum += SSIM_3d_calcWindow_file(oriData[cur_thread].data(), decData[cur_thread].data(), windowSize1,
                                              windowSize0, 0, 0, 0, windowSize0, windowSize1, windowSize2);
            }
        }
    }
    return ssimSum / nw;
}

template <class T>
double calculateSSIM_file(const char *oriData_file, const char *decData_file, int dim, size_t *dims) {
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
            // result = SSIM_1d_windowed(oriData, decData, dims[0], windowSize0, windowShift0);
            break;
        case 2:
            // result = SSIM_2d_windowed(oriData, decData, dims[0], dims[1], windowSize0, windowSize1, windowShift0,
            // windowShift1);
            break;
        case 3:
            result = SSIM_3d_windowed_file<T>(oriData_file, decData_file, dims[0], dims[1], dims[2], windowSize0,
                                         windowSize1, windowSize2, windowShift0, windowShift1, windowShift2);
            break;
        case 4:
            // result = SSIM_4d_windowed(oriData, decData, dims[0], dims[1], dims[2], dims[3], windowSize0, windowSize1,
            // windowSize2, windowSize3, windowShift0, windowShift1, windowShift2, windowShift3);
            break;
    }
    return result;
}

#endif