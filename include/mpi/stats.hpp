#ifndef STAT_MPI_HPP
#define STAT_MPI_HPP

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdio>   

template <typename T>
double get_psnr_mpi(T* orig, T* decompressed, size_t local_size, MPI_Comm cart_comm) {
    double local_sse = 0.0;
    double local_min;
    double local_max;
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    local_min = orig[0];
    local_max = orig[0];
    for (size_t i = 0; i < local_size; i++) {
        local_sse += (orig[i] - decompressed[i]) * (orig[i] - decompressed[i]);
        if (orig[i] < local_min) {
            local_min = orig[i];
        }
        if (orig[i] > local_max) {
            local_max = orig[i];
        }
    }
    // printf("Rank %d, local_sse %f\n", rank, local_sse);
    double global_min;
    double global_max;
    double global_sse = 0.0;
    size_t global_size = 0;
    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_size, &global_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, cart_comm);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (rank != 0) {
        return 0;
    }
    double global_range = global_max - global_min;
    double global_mse = global_sse / global_size;
    double nrmse = sqrt(global_mse) / global_range;
    double global_psnr = 20 * log10(global_range) - 10 * log10(global_mse);

    printf(
        "Global min: %f, Global max: %f, global range: %f, global sse: %f, global mse: %f, nrmse: %f, global psnr: "
        "%f\n",
        global_min, global_max, global_range, global_sse, global_mse, nrmse, global_psnr);

    return global_psnr;
}

#endif