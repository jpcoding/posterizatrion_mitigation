#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include<algorithm>
#include <vector>
#include "mpi/qcat_ssim_mpi.hpp"
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
    // std::vector<float> dec_data(data_size, 0); 
    // readfile(argv[N + 3], data_size, dec_data.data());
    std::cout << "loaded data " << std::endl;
    double ssim = 0; 
    calculateSSIM_(original_data.data(), original_data.data(), N, dims.data(), argv[N + 4]);
    std::cout << "ssim = " << ssim << std::endl;
    return 0;
}