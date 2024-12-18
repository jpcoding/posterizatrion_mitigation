#ifndef COMPENSATION_HPP
#define COMPENSATION_HPP

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

#include "compute_grad.hpp"
#include "edt_transform.hpp"
#include "get_boundary.hpp"
#include "utils/file_utils.hpp"
#include "utils/timer.hpp"

namespace PM {
template <typename T_data, typename T_quant>
class Compensation {
   public:
    Compensation(int n, int *dims, T_data *dec_data, T_quant *quant, double compensation_value) {
        N = n;
        this->dec_data = dec_data;
        this->quant_index = quant;
        input_size = 1;
        for (int i = 0; i < n; i++) {
            this->dims.push_back(dims[i]);
            input_size *= dims[i];
        }
        strides.resize(n);
        strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        this->comepnsation_value = compensation_value;
        compensation_map.resize(input_size, 0);
    }

    template <typename T_data_sign>
    char get_sign(T_data_sign data) {
        char sign = (char) (((double) data > 0.0) - ((double) data < 0.0));
        return sign;
    }

    std::tuple<std::array<char, 4>, std::array<int, 4>> check_compensate_direction_distance_2d(size_t index) {
        int x, y;
        int tx, ty;
        char left, right, up, down;
        int d_left, d_right, d_up, d_down;
        x = index / dims[1];
        y = index % dims[1];
        int cur_quant_index = quant_index[index];

        tx = x, ty = y - 1;
        while (ty > 0) {
            int cur_idx = tx * dims[1] + ty;
            if (quant_index[cur_idx] != cur_quant_index) {
                left = get_sign(cur_quant_index - quant_index[cur_idx]);
                break;
            }
            ty--;
        }
        d_left = y - ty - 1;

        tx = x, ty = y + 1;
        while (ty < dims[1]-1) {
            int cur_idx = tx * dims[1] + ty;
            if (quant_index[cur_idx] != cur_quant_index) {
                right = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            ty++;
        }
        d_right = ty - y - 1;
        tx = x - 1, ty = y;
        while (tx > 0) {
            int cur_idx = tx * dims[1] + ty;
            if (quant_index[cur_idx] != cur_quant_index) {
                up = get_sign(cur_quant_index -quant_index[cur_idx]);
                break;
            }
            tx--;
        }
        d_up = x - tx - 1;
        tx = x + 1, ty = y;
        while (tx < dims[0]-1) {
            int cur_idx = tx * dims[1] + ty;
            if (quant_index[cur_idx] != cur_quant_index) {
                down = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            tx++;
        }
        d_down = tx - x - 1;
        std::array<char, 4> compensate_direction{left, right, up, down};
        std::array<int, 4> change_distance{d_left, d_right, d_up, d_down};
        return std::make_tuple(compensate_direction, change_distance);
    }

    std::tuple<std::array<char, 6>, std::array<int, 6>> check_compensate_direction_distance_3d(size_t index) {
        int x, y, z;
        int tx, ty, tz;
        char left, right, up, down, front, back;
        int d_left, d_right, d_up, d_down, d_front, d_back;
        x = index / (dims[1] * dims[2]);
        y = (index / dims[2]) % dims[1];
        z = index % dims[2];
        int cur_quant_index = quant_index[index];

        tx = x, ty = y - 1, tz = z;
        while (ty >= 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                left = get_sign(cur_quant_index -quant_index[cur_idx]);
                break;
            }
            ty--;
        }
        d_left = y - ty - 1;
        tx = x, ty = y + 1, tz = z;
        while (ty < dims[1]) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                right = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            ty++;
        }
        d_right = ty - y - 1;
        tx = x, ty = y, tz = z - 1;
        while (tz >= 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] !=cur_quant_index) {
                up = get_sign(cur_quant_index - quant_index[cur_idx] );
                break;
            }
            tz--;
        }
        d_up = z - tz - 1;
        tx = x, ty = y, tz = z + 1;
        while (tz < dims[2]) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                down = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            tz++;
        }
        d_down = tz - z - 1;
        tx = x - 1, ty = y, tz = z;
        while (tx >= 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                front = get_sign(cur_quant_index - quant_index[cur_idx]);
                break;
            }
            tx--;
        }
        d_front = x - tx - 1;
        tx = x + 1, ty = y, tz = z;
        while (tx < dims[0]) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                back = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            tx++;
        }
        d_back = tx - x - 1;
        std::array<char, 6> compensate_direction{left, right, up, down, front, back};
        std::array<int, 6> change_distance{d_left, d_right, d_up, d_down, d_front, d_back};
        return std::make_tuple(compensate_direction, change_distance);
    }

    double find_opposite_distance_2d(double *distance_array, char *boundary_mao, size_t cur_index, size_t near_index,
                                     T_data *compensation_map, int max_extend = 10){
        int x, y;
        x = cur_index / dims[1];
        y = cur_index % dims[1];
        char cur_sign = get_sign(compensation_map[near_index]);
        double dx, dy;
        int near_x, near_y;
        near_x = near_index / dims[1];
        near_y = near_index % dims[1];
        dx = near_x - x;
        dy = near_y - y;

        double norm = std::sqrt(dx * dx + dy * dy);
        dx = dx / norm;
        dy = dy / norm;

        int max_steps = *std::max_element(dims.begin(), dims.end());
        double distance_to_opposite_boundary = 0;
        for (int i = 1; i < max_steps; i++) {
            int tx = int(x - i * dx);
            int ty = int(y - i * dy);
            size_t global_index = tx * dims[1] + ty;

            if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1]) {
                break;
            }
            char next_sign = get_sign(compensation_map[global_index]);
            if (next_sign != cur_sign and boundary_mao[global_index] == 0) {
                return distance_to_opposite_boundary;
            }
            distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));
        }
        // distance_to_opposite_boundary = 0;
        return distance_to_opposite_boundary;

}

    double find_opposite_distance_3d(double *distance_array, char *boundary_mao, size_t cur_index, size_t near_index,
                                     T_data *compensation_map, int max_extend = 10) {
        int x, y, z;
        x = cur_index / (dims[1] * dims[2]);
        y = (cur_index / dims[2]) % dims[1];
        z = cur_index % dims[2];
        char cur_sign = get_sign(compensation_map[near_index]);
        double dx, dy, dz;
        int near_x, near_y, near_z;
        near_x = near_index / (dims[1] * dims[2]);
        near_y = (near_index / dims[2]) % dims[1];
        near_z = near_index % dims[2];
        dx = near_x - x;
        dy = near_y - y;
        dz = near_z - z;
        double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        dx = dx / norm;
        dy = dy / norm;
        dz = dz / norm;
        int max_steps = *std::max_element(dims.begin(), dims.end());
        double distance_to_opposite_boundary = 0;
        for (int i = 1; i < max_steps; i++) {
            int tx = int(x - i * dx);
            int ty = int(y - i * dy);
            int tz = int(z - i * dz);
            size_t global_index = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1] || tz < 0 || tz >= dims[2]) {
                break;
            }
            char next_sign = get_sign(compensation_map[global_index]);
            if (next_sign != cur_sign and boundary_mao[global_index] == 0) {
                return distance_to_opposite_boundary;
            }
            distance_to_opposite_boundary =
                sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y) + 1.0 * (tz - z) * (tz - z));
        }
        // distance_to_opposite_boundary = 0;
        return distance_to_opposite_boundary;
    }

    std::vector<T_data> get_compensation_map_2d() {
        auto boundary_map = get_boundary(quant_index, N, dims.data());
        // flip the boundary map tag
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0; // boundary lable
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();

        timer.start();
        auto edt_result = NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data());
        std::cout << "edt time = "  << timer.stop() << std::endl;
        auto distance_array = std::get<0>(edt_result);

        auto indexes = std::get<1>(edt_result);
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
                auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
                auto [compensate_direction, change_distance] = check_compensate_direction_distance_2d(i);
                auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
                auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
                int direction = std::distance(change_distance.begin(), min_iter);
                double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
                double grad = grad_computer.get_grad(i);

                if (grad >= 1.0) {
                    sign = 0;
                }
                compensation_map[i] = sign * comepnsation_value;
            }
        }
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                double min_distance = distance_array[i];
                char sign = get_sign(compensation_map[indexes[i]]);
                double distance_to_opposoite_boundary = find_opposite_distance_2d(
                    distance_array.data(), boundary_map.data(), 
                    i, indexes[i], compensation_map.data());
                double width = distance_to_opposoite_boundary + min_distance;
                double magnitude = 1.0 / (width * width) * ((width - min_distance) * (width - min_distance));
                // magnitude = std::exp(-min_distance*0.3);
                compensation_map[i] = sign * magnitude * comepnsation_value;
            }
        }
        return compensation_map;
    }

    std::vector<T_data> get_compensation_map_3d() {
        auto boundary_map = get_boundary(quant_index, N, dims.data());
        // flip the boundary map tag
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();
        timer.start();

        auto edt_result = NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data());
        std::cout << "edt time = "  << timer.stop() << std::endl;
        auto distance_array = std::get<0>(edt_result);
        auto indexes = std::get<1>(edt_result);
        std::cout << "compenstion_value = " << comepnsation_value << std::endl;
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
                auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
                auto [compensate_direction, change_distance] = check_compensate_direction_distance_3d(i);
                auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
                auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
                int direction = std::distance(change_distance.begin(), min_iter);
                double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
                double grad = grad_computer.get_grad(i);
                if (grad >= 1) {
                    sign = 0;
                }
                if(i == 3315383)
                {
                    std::cout << "grad = " << grad << std::endl;
                    std::cout << "sign = " << sign << std::endl;
                    std::cout << "change distance = " << std::endl;
                    for(int j = 0; j < 6; j++)
                    {
                        std::cout << " " << change_distance[j] << " ";
                    }
                    std::cout <<  std::endl;
                    std::cout << "compensate_direction = " << std::endl;
                    for(int j = 0; j < 6; j++)
                    {
                        std::cout << " " << (int) compensate_direction[j] << " ";
                    }
                    std::cout <<  std::endl;
                    std::cout << "sign = " << (int) sign << std::endl;
                    std::cout << "compensation = " << sign * comepnsation_value << std::endl;
                }
                compensation_map[i] = sign * comepnsation_value;
            }
        }
        std::cout << "compensation_map[0] = " << compensation_map[0] << std::endl;

        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                double min_distance = distance_array[i];
                char sign = get_sign(compensation_map[indexes[i]]);
                // std::cout << "sign = " << (int) sign << std::endl;
                double distance_to_opposoite_boundary = find_opposite_distance_3d(
                    distance_array.data(), boundary_map.data(), 
                    i, indexes[i], compensation_map.data());
                double width = distance_to_opposoite_boundary + min_distance;
                double magnitude = 1.0 / (width * width) * (width - min_distance) * (width - min_distance);
                compensation_map[i] = sign * magnitude * comepnsation_value;
            }
        }
        std::cout << "compensation_map[0] = " << compensation_map[0] << std::endl;

        return compensation_map;
    }

    std::vector<T_data> get_compensation_map() {
        if (N == 2) {
            return get_compensation_map_2d();
        } else if (N == 3) {
            return get_compensation_map_3d();
        }
    }

   private:
    int N;
    std::vector<int> dims;
    std::vector<size_t> strides;
    T_data *dec_data;
    T_quant *quant_index;
    size_t input_size;
    double comepnsation_value;
    std::vector<T_data> compensation_map;
    char *boundary_map;
};
}  // namespace PM

#endif  // COMPENSATION_HPP