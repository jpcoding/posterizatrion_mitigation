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
#include <vector> 

int main(int argc, char** argv)
{
    int N = atoi(argv[1]); 
    std::vector<size_t> dims(N);
    size_t data_size = 1;
    for (int i = 0; i < N; i++) {
        dims[i] = atoi(argv[i + 2]);
        data_size *= dims[i];
    }
    std::vector<float> original_data(data_size, 0); 
    readfile(argv[N + 2], data_size, original_data.data());
    std::vector<float> dec_data(data_size, 0); 
    readfile(argv[N + 3], data_size, dec_data.data());
    double ssim = 0; 
    ssim = PM::calculateSSIM(original_data.data(), dec_data.data(), N, dims.data());
    std::cout << "ssim = " << ssim << std::endl;
    return 0;
}