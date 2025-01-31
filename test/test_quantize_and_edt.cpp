#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
// #include "SZ3/api/sz.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include<algorithm>
#include "utils/timer.hpp"
#include "compensation.hpp"
#include <vector>
#include "utils/qcat_ssim.hpp"

using Real = float; 
namespace SZ=SZ3; 

namespace fs = std::filesystem;


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


int main(int argc, char** argv)
{
    int N = atoi(argv[1]);
    std::vector<int> dims(N);
    size_t data_size = 1;
    for (int i = 0; i < N; i++) {
        dims[i] = atoi(argv[i + 2]);
        data_size *= dims[i];
    }

    std::vector<Real> original_data(data_size, 0);
    readfile(argv[N + 2], data_size, original_data.data());
    double rel_eb = atof(argv[N + 3]); 

    int num_threads = 1;
    if (argc >= N + 7) {
        num_threads = atoi(argv[N + 6]);
        std::cout << "num_threads = " << num_threads << std::endl; 
    }   

    // make a copy of the original data
    std::vector<Real> dec_data(data_size, 0);
    std::copy(original_data.begin(), original_data.end(), dec_data.begin());


    std::vector<int> quant_inds(data_size, 0);


    float max = *std::max_element(original_data.begin(), original_data.end());
    float min = *std::min_element(original_data.begin(), original_data.end());
    printf("max: %f, min: %f\n", max, min);
    double eb = rel_eb*(max - min);
    printf("relative eb: %.6f\n",rel_eb);
    printf("absolute eb: %.6f\n", eb);
    // create a linear quantizer
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(eb);
    // iterate the input data and quantize it
    for (size_t i = 0; i < data_size; i++) {
        quant_inds[i] = quantizer.quantize_and_overwrite(dec_data[i],0)-32768;
    }
    // write quantized data 
    // writefile((fs::path(argv[N + 2]).filename().string() + ".qcd").c_str(), dec_data.data(), data_size);
    writefile(argv[N+4], dec_data.data(), data_size);

    writefile("quant_index.i32", quant_inds.data(), data_size);
    // verify the data 
    double psnr, nrmse, max_diff;
    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);
    
        // cast dims to size_t 
        std::vector<size_t> dims_(data_size);
        for (int i = 0; i < N; i++) {
            dims_[i] = dims[i];
        }
        // auto ssim = PM::calculateSSIM(original_data.data(), dec_data.data(), N, dims_.data());
        // printf("SSIM = %f\n", ssim);
    

    // compensation using edt method

    double compensation_factor = 0.9;
    auto timer = Timer(); 
    timer.start();
    auto compensator = PM::Compensation<Real, int>(N, dims.data(),
                    dec_data.data(), quant_inds.data(), max_diff*compensation_factor);
    compensator.set_edt_thread_num(num_threads); 

    auto compensation_map = compensator.get_compensation_map();
    // writefile("compensation_map.f32", compensation_map.data(), data_size);
    // add the compensation map to the dec_data
    for (int i = 0; i < data_size; i++) {
        dec_data[i] += compensation_map[i];
    }
    std::cout << "compensation time = " << timer.stop() << std::endl;

    // verify the compensated data
    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);
    // ssim = PM::calculateSSIM(original_data.data(), dec_data.data(), N, dims_.data());
    // printf("SSIM = %f\n", ssim);
    // write the compensation map to file
    writefile(argv[N+5], dec_data.data(), data_size);
    return 0;
}


