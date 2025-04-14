#ifndef COMPENSATION_MPI
#define COMPENSATION_MPI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>
#include <locale>
#include <vector>
#include <cmath> 

template <typename T_data, typename T_sign, typename T_distance>
void compensation_idw(T_data* compensation, T_data* quantized_data, T_distance* orig_distance,
                      T_distance* neutral_distance, T_sign* sign_map, size_t block_size,
                      double compensation_magnitude) {
    for (size_t i = 0; i < block_size; i++) {
        double d1 = orig_distance[i] + 0.5;
        double d2 = neutral_distance[i] + 0.5;
        char sign = sign_map[i];
        double magnitude = (1 / d1) / (1 / d1 + 1 / d2);
        // compensation[i] = sign * magnitude * compensation_magnitude;
        // quantized_data[i] += compensation[i];
        quantized_data[i] += sign * magnitude * compensation_magnitude; 
    }
}




template <typename T_data, typename T_sign, typename T_distance, typename T_index> 
void compensation_rbf(T_data* compensation, T_data* quantized_data, T_distance* orig_distance, T_index* indexes,
                   T_distance* neutral_distance, T_index* indexes2, int* dims, 
                   T_sign* sign_map, size_t block_size, double compensation_magnitude) {
    auto rbf = [](double r) -> double {
        // return std::exp(-0.3*r);
        // return (1/r) / (1/r + 1); //?
        return 1 / sqrt(1 + r * r);  // inverse_multiquadric
        // cubic, thin-plate, gaussian, multiquadric, inverse multiquadric
        // thin-plate:
        // return r*r * log(r);
    };
    // calculate the distance between two points
    auto cal_distance = [&](int i, int j) -> double {
        int x1 = i / (dims[1] * dims[2]);
        int y1 = (i / dims[2]) % dims[1];
        int z1 = i % dims[2];
        int x2 = j / (dims[1] * dims[2]);
        int y2 = (j / dims[2]) % dims[1];
        int z2 = j % dims[2];
        return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    };

    for (size_t i = 0; i < block_size; i++) {
        double d1 = orig_distance[i] + 0.5;
        double d2 = neutral_distance[i] + 0.5;
        T_sign sign = sign_map[i];
        double d0 = cal_distance(indexes[i], indexes2[i]);
        double a = rbf(0.5);
        double b = rbf(d0);
        double w0 = a / (a*a - b*b)*sign;
        double w1 = b / (-a*a + b*b)*sign;   
        // compensation[i] =  (w0 * rbf(d1) + w1 * rbf(d2)) * compensation_magnitude;
        quantized_data[i] += (w0 * rbf(d1) + w1 * rbf(d2)) * compensation_magnitude;
    }
}

#endif
