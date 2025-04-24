
#include "utils/qcat_ssim.hpp"

#define K1 0.01
#define K2 0.03
const int windowShift0 = 2;
const int windowShift1 = 2;
const int windowShift2 = 2;
const int windowShift3 = 2;


extern "C" 
double calculateSSIM(float *oriData, float *decData, int *dims, int ndims) {
  // const int dim = dims.size();
  const int windowSize0 = 7;
  const int windowSize1 = 7;
  const int windowSize2 = 7;
  const int windowSize3 = 7;
  // int windowShift0 = 2;
  // int windowShift1 = 2;
  // int windowShift2 = 2;
  // int windowShift3 = 2;
  double result = -1;

  switch (ndims) {
  case 1:
    result =
        PM::SSIM_1d_windowed(oriData, decData, dims[0], windowSize0, windowShift0);
    break;
  case 2:
    result = PM::SSIM_2d_windowed(oriData, decData, dims[1], dims[0], windowSize0,
                              windowSize1, windowShift0, windowShift1);
    break;
  case 3:
    result = PM::SSIM_3d_windowed(oriData, decData, dims[2], dims[1], dims[0],
                              windowSize0, windowSize1, windowSize2,
                              windowShift0, windowShift1, windowShift2);
    break;
  case 4:
    result = PM::SSIM_4d_windowed(oriData, decData, dims[3], dims[2], dims[1],
                              dims[0], windowSize0, windowSize1, windowSize2,
                              windowSize3, windowShift0, windowShift1,
                              windowShift2, windowShift3);
    break;
  }
  return result;
}

extern "C" 
double calculateSSIM_double(double *oriData, double *decData, int *dims, int ndims) {
  // const int dim = dims.size();
  const int windowSize0 = 7;
  const int windowSize1 = 7;
  const int windowSize2 = 7;
  const int windowSize3 = 7;
  // int windowShift0 = 2;
  // int windowShift1 = 2;
  // int windowShift2 = 2;
  // int windowShift3 = 2;
  double result = -1;

  switch (ndims) {
  case 1:
    result =
        PM::SSIM_1d_windowed(oriData, decData, dims[0], windowSize0, windowShift0);
    break;
  case 2:
    result = PM::SSIM_2d_windowed(oriData, decData, dims[1], dims[0], windowSize0,
                              windowSize1, windowShift0, windowShift1);
    break;
  case 3:
    result = PM::SSIM_3d_windowed(oriData, decData, dims[2], dims[1], dims[0],
                              windowSize0, windowSize1, windowSize2,
                              windowShift0, windowShift1, windowShift2);
    break;
  case 4:
    result = PM::SSIM_4d_windowed(oriData, decData, dims[3], dims[2], dims[1],
                              dims[0], windowSize0, windowSize1, windowSize2,
                              windowSize3, windowShift0, windowShift1,
                              windowShift2, windowShift3);
    break;
  }
  return result;
}