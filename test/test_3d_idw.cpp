 #include <assert.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "compensation_idw.hpp"
#include "utils/file_utils.hpp"

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse, double &max_diff) {
    size_t i = 0;
    double Max = ori_data[0];
    double Min = ori_data[0];
    max_diff = fabs(data[0] - ori_data[0]);
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0, l2sum = 0;
    for (i = 0; i < num_elements; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
        l2sum += data[i] * data[i];
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double *diff = (double *)malloc(num_elements * sizeof(double));

    for (i = 0; i < num_elements; i++) {
        diff[i] = data[i] - ori_data[i];
        diff_sum += data[i] - ori_data[i];
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        double err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr) maxpw_relerr = relerr;
        }

        if (max_diff < err) max_diff = err;
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = sum / num_elements;
    double sse = sum; // sum of square error
    double range = Max - Min;
    psnr = 20 * log10(range) - 10 * log10(mse);
    nrmse = sqrt(mse) / range;

    double normErr = sqrt(sum);
    double normErr_norm = normErr / sqrt(l2sum);

    printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("Max absolute error = %.2G\n", max_diff);
    printf("Max relative error = %.2G\n", max_diff / (Max - Min));
    printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("PSNR = %f, NRMSE= %.10G\n", psnr, nrmse);
    printf("normError = %f, normErr_norm = %f\n", normErr, normErr_norm);
    printf("acEff=%f\n", acEff);
    printf("SSE=%f\n", sse);
    printf("MSE=%f\n", mse);
    //        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements, diff_sum / num_elements));
    free(diff);
}

using Real = float;
int main(int argc, char **argv) {
    // prepare original data, dec data, quant index and dims
    int N = atoi(argv[1]);
    std::vector<int> dims(N);
    size_t data_size = 1;
    for (int i = 0; i < N; i++) {
        dims[i] = atoi(argv[i + 2]);
        data_size *= dims[i];
    }
    std::vector<Real> original_data(data_size, 0);
    std::vector<Real> dec_data(data_size, 0);
    std::vector<int> quant_index(data_size, 0);
    readfile(argv[N + 2], data_size, original_data.data());
    readfile(argv[N + 3], data_size, dec_data.data());
    readfile(argv[N + 4], data_size, quant_index.data());


    std::cout << argv[N + 2] << std::endl;
    std::cout << argv[N + 3] << std::endl;
    std::cout << argv[N + 4] << std::endl;
    double psnr, nrmse, max_diff;
    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);

    // get the error
    std::vector<Real> error(data_size, 0);
    double error_max = 0;
    for (int i = 0; i < data_size; i++) {
        error[i] = original_data[i] - dec_data[i];
        error_max = std::max(error_max, (double)std::abs(error[i]));
        quant_index[i] = quant_index[i] - 32768 ;
    }
    std::cout << "error_max = " << error_max << std::endl;
    // get compensation map
    double compensation_factor = 0.9;
    auto timer = Timer();
    timer.start();
    auto compensator = PM::CompensationIDW<Real, int>(N, dims.data(),
                    dec_data.data(), quant_index.data(), error_max*compensation_factor);

    auto compensation_map = compensator.get_compensation_map();
    std::cout << "IDW compensation time = " << timer.stop() << std::endl;

    // write the compensation map to file
    std::string file_basename = fs::path(argv[N + 2]).filename();
    std::cout << "file_basename = " << file_basename << std::endl; 
    writefile((file_basename + "_compensation_map.f32").c_str(), compensation_map.data(), data_size);

    // add the compensation map to the dec data
    for (int i = 0; i < data_size; i++) {
        dec_data[i] += compensation_map[i];
    }
    writefile((file_basename + "_compensated_data.f32").c_str(), dec_data.data(), data_size);
    // calculate PSNR 

    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);
    return 0;
}
