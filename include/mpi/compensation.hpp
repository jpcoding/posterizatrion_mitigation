#ifndef COMPENSATION_MPI
#define COMPENSATION_MPI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>
#include <locale>
#include <vector>

template <typename T_data, typename T_sign, typename T_distance>
void compensation_idw(T_data* compensation, T_data* quantized_data, T_distance* orig_distance,
                      T_distance* neutral_distance, T_sign* sign_map,
                      size_t block_size, double compensation_magnitude) {
    for (size_t i = 0; i < block_size; i++) {
        double d1 = orig_distance[i] + 0.5;
        double d2 = neutral_distance[i] + 0.5;
        char sign = sign_map[i];
        double magnitude = (1 / d1) / (1 / d1 + 1 / d2);
        compensation[i] = sign * magnitude * compensation_magnitude;
        quantized_data[i]+= compensation[i];
    }
}

#endif
