#ifndef COMPENSATION_HPP
#define COMPENSATION_HPP

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "compute_grad.hpp"
#include "edt_transform_omp.hpp"
#include "edt_transform.hpp"
#include "get_boundary.hpp"
#include "utils/file_utils.hpp"
#include "utils/timer.hpp"

namespace PM {
template <typename T_data, typename T_quant>
class Compensation {
   public:
    Compensation(int n, int *dims, T_data *dec_data, T_quant *quant, double compensation_value) {
        this->N = n;
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

    ~Compensation() {}

    void set_edt_thread_num(int num_threads) { this->edt_thread_num = num_threads; } 

    template <typename T_data_sign>
    char get_sign(T_data_sign data) {
        char sign = (char)(((double)data > 0.0) - ((double)data < 0.0));
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
        while (ty < dims[1] - 1) {
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
                up = get_sign(cur_quant_index - quant_index[cur_idx]);
                break;
            }
            tx--;
        }
        d_up = x - tx - 1;
        tx = x + 1, ty = y;
        while (tx < dims[0] - 1) {
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
        char left = 0, right = 0, up = 0, down = 0, front = 0, back = 0;
        int d_left = 0, d_right = 0, d_up = 0, d_down = 0, d_front = 0, d_back = 0;
        x = index / (dims[1] * dims[2]);
        y = (index / dims[2]) % dims[1];
        z = index % dims[2];
        int cur_quant_index = quant_index[index];

        tx = x, ty = y - 1, tz = z;
        while (ty > 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                left = get_sign(cur_quant_index - quant_index[cur_idx]);
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
        while (tz > 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                up = get_sign(cur_quant_index - quant_index[cur_idx]);
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
        while (tx > 0) {
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

    double find_opposite_distance_2d(double *distance_array, size_t *index_array, char *boundary_map, size_t cur_index,
                                     size_t near_index, T_data *compensation_map, int max_extend = 10) {
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

        dx = dx * 1.0 / norm;
        dy = dy * 1.0 / norm;
        int tx, ty;

        // if(cur_index == 6045)
        // {
        //     std::cout << "### x = " << x << " y = " << y << std::endl;
        //     std::cout << "### near_x = " << near_x << " near_y = " << near_y << std::endl;
        //     std::cout << "### dx = " << dx << " dy = " << dy << std::endl;
        // }

        int max_steps = *std::max_element(dims.begin(), dims.end());
        double distance_to_opposite_boundary = 0;
        for (int i = 1; i < max_steps; i++) {
            tx = std::round(x - i * dx);
            ty = std::round(y - i * dy);
            size_t global_index = tx * dims[1] + ty;
            // if(cur_index == 6045)
            // {
            //     std::cout << "### dy " << dy <<  std::endl;
            //     std::cout << "### dy " << dy*i <<  std::endl;
            //     std::cout << "### y " << y <<  std::endl;
            //     std::cout << "### calcl " << y*1.0 - i * dy <<  std::endl;
            //     std::cout << "### calcl " << int(y - i * dy) <<  std::endl;
            //     std::cout << "### i  = " << i  <<  std::endl;
            //     std::cout << "### tx = " << tx << " ty = " << ty << std::endl;
            //     std::cout << "### global_index = " << global_index << std::endl;
            // }

            if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1]) {
                break;
            }
            char next_sign = get_sign(compensation_map[index_array[global_index]]);
            if (next_sign != cur_sign) {
                // distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));
                return distance_to_opposite_boundary + 1;
            }
            distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));
        }
        // double prev_distance = distance_array[cur_index];
        // for(int i = 1; i < max_steps; i++)
        // {
        //     tx = std::round(x - i * dx);
        //     ty = std::round(y - i * dy);
        //     size_t global_index = tx * dims[1] + ty;
        //     if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1]) {
        //         break;
        //     }
        //     double next_distance = distance_array[global_index];
        //     if (next_distance < prev_distance) {
        //         distance_to_opposite_boundary =
        //             sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));
        //         return distance_to_opposite_boundary;
        //     }

        // }
        // distance_to_opposite_boundary = 0;
        distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));

        return distance_to_opposite_boundary;
    }

    double find_opposite_distance_3d(double *distance_array, size_t *index_array, char *boundary_mao, size_t cur_index,
                                     size_t near_index, T_data *compensation_map, int max_extend = 10) {
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
        int tx, ty, tz;
        for (int i = 1; i < max_steps; i++) {
            tx = int(x - i * dx);
            ty = int(y - i * dy);
            tz = int(z - i * dz);
            size_t global_index = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1] || tz < 0 || tz >= dims[2]) {
                break;
            }
            char next_sign = get_sign(compensation_map[index_array[global_index]]);
            if (next_sign != cur_sign and boundary_mao[global_index] == 0) {
                // distance_to_opposite_boundary =
                // sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y) + 1.0 * (tz - z) * (tz - z));
                return distance_to_opposite_boundary + 1;
                // return distance_to_opposite_boundary;
            }
            distance_to_opposite_boundary =
                sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y) + 1.0 * (tz - z) * (tz - z));
        }
        // distance_to_opposite_boundary = 0;
        distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));

        return distance_to_opposite_boundary;
    }

    std::vector<T_data> get_compensation_map_2d_() {
        auto boundary_map = get_boundary(quant_index, N, dims.data());
        // flip the boundary map tag
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;  // boundary lable
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();

        timer.start();
        auto edt_omp = PM2::EDT_OMP();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data());
        auto distance_array = std::get<0>(edt_result);
        auto indexes = std::get<1>(edt_result);
        printf("distance address  = %ld \n", &distance_array[0]);
        printf("index address  = %ld \n", &indexes[0]); 
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
                    distance_array.data(), indexes.data(), boundary_map.data(), i, indexes[i], compensation_map.data());
                double width = distance_to_opposoite_boundary + min_distance;
                double magnitude = 1.0 / (width * width) * ((width - min_distance) * (width - min_distance));
                // double rel_R = (min_distance -0.5)/(width-0.5);
                // double magnitude  = (1-rel_R)*(1-rel_R);

                compensation_map[i] = sign * magnitude * comepnsation_value;
            }
        }
        return compensation_map;
    }

    std::vector<T_data> get_compensation_map_2d() {
        auto boundary_map = get_boundary(quant_index, N, dims.data());
        // flip the boundary map tag
        std::vector<bool> boundary_mask(input_size, false);
        size_t edge_point_count = 0; 
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;  // boundary lable
                boundary_mask[i] = true; // boundary lable
                edge_point_count++;
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();
        // std::cout << "edge point count = " << edge_point_count << std::endl; 
        if (edge_point_count <= compensation_map.size()*0.1)
        {
            return compensation_map; 
        } 

        // timer.start();
        auto edt_omp = PM2::EDT_OMP();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data());
        // std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array = std::get<0>(edt_result);
        auto indexes = std::get<1>(edt_result);
        std::vector<int> sign_map(input_size, 0);
        auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
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
                sign_map[i] = sign; 
            }
        }
        // complete the sign map
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                char sign = get_sign(compensation_map[indexes[i]]);
                sign_map[i] = sign;
            }
        }
        // get the second boundry map 
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());
        // filp and remove the boundary points 
        for (int i = 0; i < input_size; i++) {
            if (boundary_map2[i] == 1 && boundary_mask[i] == false) { 
                boundary_map2[i] = 0;  // boundary lable
            } else {
                boundary_map2[i] = 1;
            }
        }
        // get the second edt map 
        timer.start();
        edt_omp.reset_timer();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map2.data(), N, dims.data());
        // std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array2 = std::get<0>(edt_result2);
        auto indexes2 = std::get<1>(edt_result2);

        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                double distance1 = distance_array[i] + 0.5; 
                double distance2 = distance_array2[i] + 0.5 ;
                char sign = sign_map[i];
                double width = distance2 + distance1;
                // double relative_r = (distance1 ) / (width);
                // double magnitude = (1 - relative_r) * (1 - relative_r);
                // double magnitude = std::pow(1 - relative_r, 1.5);
                double magnitude = (1/distance1) / (1/distance1 + 1/distance2);
                compensation_map[i] = sign * magnitude * comepnsation_value;   
            }
            // new logic 
            // if(boundary_map[i] == 1)
            // {
            //     double distance1 = distance_array[i] + 0.5; 
            //     double distance2 = distance_array2[i] + 0.5 ;
            //     char sign = sign_map[i];
            //     int old_x = indexes[i] / dims[1], old_y = indexes[i] % dims[1]; 
            //     int new_x = indexes2[i]/ dims[1], new_y = indexes2[i] % dims[1]; 
            //     int cur_x = i / dims[1], cur_y = i % dims[1]; 
            //     double dot_product = (old_x - cur_x) * (new_x - cur_x) + (old_y - cur_y) * (new_y - cur_y); 
            //     double magnitude = 0; 
            //     if (dot_product <=0  && distance2 <= distance1)
            //     {
            //         double width = distance2 + distance1;
            //         double relative_r = (distance1 ) / (width);
            //         magnitude = std::pow(1 - relative_r, 1.5);
            //     }
            //     else if ( dot_product <=0 && distance2 > distance1)
            //     {
            //         magnitude = std::exp(-0.3*distance1); 
            //         // what if using idw here 
            //     }
            //     else if (dot_product > 0 && distance2 <= distance1)
            //     {
            //         double width = distance2 + distance1;
            //         double relative_r = (distance1 ) / (width);
            //         magnitude = std::pow(1 - relative_r, 1.5); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
               
            //     }
            //     else if (dot_product > 0 && distance2 > distance1)
            //     {
            //         magnitude = std::exp(-0.3*distance1); 
            //     }
            //     else{
            //         magnitude = std::exp(-0.3*distance1);  
            //     }
            //     compensation_map[i] = sign * magnitude * comepnsation_value; 
             
            // }
        }
        return compensation_map;
    }


    std::vector<T_data> get_compensation_map_2d(std::vector<double> &distance_array,
                                                std::vector<double> &distance_array2,
                                                std::vector<int> &sign_map) {
    auto boundary_map = get_boundary(quant_index, N, dims.data());
    // flip the boundary map tag
    std::vector<bool> boundary_mask(input_size, false);
    sign_map.resize(input_size, 0); 
    size_t edge_point_count = 0; 
    for (int i = 0; i < input_size; i++) {
        if (boundary_map[i] == 1) {
            boundary_map[i] = 0;  // boundary lable
            boundary_mask[i] = true; // boundary lable
            edge_point_count++;
        } else {
            boundary_map[i] = 1;
        }
    }
    auto timer = Timer();
    // std::cout << "edge point count = " << edge_point_count << std::endl; 
    if (edge_point_count == 0)
    {
        distance_array.resize(input_size, std::numeric_limits<double>::max());
        distance_array2.resize(input_size, std::numeric_limits<double>::max());

        return compensation_map; 
    } 

    // timer.start();
    auto edt_omp = PM2::EDT_OMP();
    edt_omp.set_num_threads(edt_thread_num);
    auto edt_result1 = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data());
    // std::cout << "edt time = " << timer.stop() << std::endl;
    distance_array = std::move(std::get<0>(edt_result1));
    auto indexes = std::move(std::get<1>(edt_result1));
    // std::vector<int> sign_map(input_size, 0);
    
    auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
    for (size_t i = 0; i < input_size; i++) {
        if (boundary_map[i] == 0)  // boundary points
        {
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
            sign_map[i] = sign; 
        }
    }
    // complete the sign map
    for (size_t i = 0; i < input_size; i++) {
        if (boundary_map[i] == 1)  // non-boundary points ·
        {
            char sign = get_sign(compensation_map[indexes[i]]);
            sign_map[i] = sign;
        }
    }
    // get the second boundry map 
    auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());
    // filp and remove the boundary points 
    for (int i = 0; i < input_size; i++) {
        if (boundary_map2[i] == 1 && boundary_mask[i] == false) { 
            boundary_map2[i] = 0;  // boundary lable
        } else {
            boundary_map2[i] = 1;
        }
    }
    // get the second edt map 
    timer.start();
    edt_omp.reset_timer();
    auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map2.data(), N, dims.data());
    distance_array2 = std::move(std::get<0>(edt_result2));
    // auto indexes2 = std::move(std::get<1>(edt_result2));

    for (size_t i = 0; i < input_size; i++) {
        if (boundary_map[i] == 1)  // non-boundary points ·
        {
            double distance1 = distance_array[i] + 0.5; 
            double distance2 = distance_array2[i] + 0.5 ;
            char sign = sign_map[i];
            double width = distance2 + distance1;
            // double relative_r = (distance1 ) / (width);
            // double magnitude = (1 - relative_r) * (1 - relative_r);
            // double magnitude = std::pow(1 - relative_r, 1.5);
            double magnitude = (1/distance1) / (1/distance1 + 1/distance2);
            compensation_map[i] = sign * magnitude * comepnsation_value;   
        }
        // new logic 
        // if(boundary_map[i] == 1)
        // {
        //     double distance1 = distance_array[i] + 0.5; 
        //     double distance2 = distance_array2[i] + 0.5 ;
        //     char sign = sign_map[i];
        //     int old_x = indexes[i] / dims[1], old_y = indexes[i] % dims[1]; 
        //     int new_x = indexes2[i]/ dims[1], new_y = indexes2[i] % dims[1]; 
        //     int cur_x = i / dims[1], cur_y = i % dims[1]; 
        //     double dot_product = (old_x - cur_x) * (new_x - cur_x) + (old_y - cur_y) * (new_y - cur_y); 
        //     double magnitude = 0; 
        //     if (dot_product <=0  && distance2 <= distance1)
        //     {
        //         double width = distance2 + distance1;
        //         double relative_r = (distance1 ) / (width);
        //         magnitude = std::pow(1 - relative_r, 1.5);
        //     }
        //     else if ( dot_product <=0 && distance2 > distance1)
        //     {
        //         magnitude = std::exp(-0.3*distance1); 
        //         // what if using idw here 
        //     }
        //     else if (dot_product > 0 && distance2 <= distance1)
        //     {
        //         double width = distance2 + distance1;
        //         double relative_r = (distance1 ) / (width);
        //         magnitude = std::pow(1 - relative_r, 1.5); 
        //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
            
        //     }
        //     else if (dot_product > 0 && distance2 > distance1)
        //     {
        //         magnitude = std::exp(-0.3*distance1); 
        //     }
        //     else{
        //         magnitude = std::exp(-0.3*distance1);  
        //     }
        //     compensation_map[i] = sign * magnitude * comepnsation_value; 
            
        // }
    }
    return compensation_map;
}


    std::vector<T_data> get_compensation_map_3d() {
        auto boundary_map = get_boundary(quant_index, N, dims.data());

        // write boundary map to file 
        // writefile("boundary.int8", boundary_map.data(), input_size);
        // flip the boundary map tag
        std::vector<bool> boundary_mask(input_size, false);
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;  // boundary lable
                boundary_mask[i] = true; // boundary lable
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();

        timer.start();
        auto edt_omp = PM2::EDT_OMP();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data(),edt_thread_num);
        std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array = std::get<0>(edt_result);
        auto indexes = std::get<1>(edt_result);

        // writefile("distance.f64", distance_array.data(), distance_array.size());
        std::vector<int> sign_map(input_size, 0);  
        auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
                auto [compensate_direction, change_distance] = 
                            check_compensate_direction_distance_3d(i);
                auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
                auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
                int direction = std::distance(change_distance.begin(), min_iter);
                double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
                double grad = grad_computer.get_grad(i);
                if (grad >= 1.0) {
                    sign = 0;
                }
                compensation_map[i] = sign * comepnsation_value;
                sign_map[i] = sign; 
            }
        }
        // complete the sign map
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                char sign = get_sign(compensation_map[indexes[i]]);
                sign_map[i] = sign;
            }
        }
        // dump the sign map 
        // writefile("sign.int8", sign_map.data(), input_size);
        // get the second boundry map 
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());

        // filp and remove the boundary points 
        for (int i = 0; i < input_size; i++) {
            if (boundary_map2[i] == 1 && boundary_mask[i] == false) { 
                boundary_map2[i] = 0;  // boundary lable
            } else {
                boundary_map2[i] = 1;
            }
        }
        // writefile("boundary2.int8", boundary_map2.data(), boundary_map2.size());
        // get the second edt map 

        timer.start();
        edt_omp.reset_timer(); 
        auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map2.data(), 
                                            N, dims.data(), edt_thread_num);
        std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array2 = std::get<0>(edt_result2);
        auto indexes2 = std::get<1>(edt_result2);
        // dump the distance array
        // writefile("distance1.f32", distance_array.data(), input_size);
        // writefile("distance2.f32", distance_array2.data(), input_size);
        for (size_t i = 0; i < input_size; i++) {
            // old method 
            // if (boundary_map[i] == 1)  // non-boundary points ·
            if(1)
            {
                double distance1 = distance_array[i] + 0.5; 
                double distance2 = distance_array2[i] + 0.5 ;
                char sign = sign_map[i];
                double width = distance2 + distance1;
                double relative_r = (distance1 ) / (width);
                // double magnitude = (1 - relative_r) * (1 - relative_r);
                // double magnitude = std::pow(1 - relative_r, 1.5);
                double magnitude = (1/distance1) / (1/distance1 + 1/distance2); 
                compensation_map[i] = sign * magnitude * comepnsation_value;  
                // int ii = i / (dims[1]*dims[2]);
                // int jj = (i / dims[2]) % dims[1];
                // int kk = i % dims[2];
                // if( ii == 81  && jj == 35 && kk == 275)
                // {
                //     std::cout << "### collision grad happned " << std::endl;
                // }
            }

            // new method 
            // if(boundary_map[i] == 1)
            // {
            //     double distance1 = distance_array[i] + 0.5; 
            //     double distance2 = distance_array2[i] + 0.5 ;
            //     char sign = sign_map[i];
            //     int old_x = indexes[i] / (dims[1] * dims[2]), old_y = (indexes[i] / dims[2]) % dims[1], old_z = indexes[i] % dims[2]; 
            //     int new_x = indexes2[i]/ (dims[1] * dims[2]), new_y = (indexes2[i] / dims[2]) % dims[1], new_z = indexes2[i] % dims[2]; 
            //     int cur_x = i / (dims[1] * dims[2]), cur_y = (i / dims[2]) % dims[1], cur_z = i % dims[2]; 
            //     double dot_product = (old_x - cur_x) * (new_x - cur_x) +
            //                              (old_y - cur_y) * (new_y - cur_y) + 
            //                                     (old_z - cur_z) * (new_z - cur_z); 
            //     double magnitude = 0; 
            //     if (dot_product <=0  && distance2 <= distance1)
            //     {
            //         double width = distance2 + distance1;
            //         double relative_r = (distance1 ) / (width);
            //         magnitude = std::pow(1 - relative_r, 1.5);
            //     }
            //     else if ( dot_product <=0 && distance2 > distance1)
            //     {
            //         magnitude = std::exp(-0.3*distance1); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
           
            //     }
            //     else if (dot_product > 0 && distance2 <= distance1)
            //     {
            //         // double width = distance2 + distance1;
            //         // double relative_r = (distance1 ) / (width);
            //         // magnitude = std::pow(1 - relative_r, 1.5);    
            //         magnitude = std::exp(-0.3*distance1); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
           
            //     }
            //     else if (dot_product > 0 && distance2 > distance1)
            //     {
            //         magnitude = std::exp(-0.3*distance1); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
            //     }
            //     else{
            //         magnitude = std::exp(-0.3*distance1);  
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
            //     }
            //     compensation_map[i] = sign * magnitude * comepnsation_value;
            // }
        }
        return compensation_map;
    }


    std::vector<T_data> get_compensation_map_3d(std::vector<int> &sign_map) {
        auto boundary_map = get_boundary(quant_index, N, dims.data());

        // write boundary map to file 
        // writefile("boundary.int8", boundary_map.data(), input_size);
        // flip the boundary map tag
        std::vector<bool> boundary_mask(input_size, false);
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;  // boundary lable
                boundary_mask[i] = true; // boundary lable
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();

        timer.start();
        auto edt_omp = PM2::EDT_OMP(); 
        edt_omp.set_num_threads(edt_thread_num); 
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data(),edt_thread_num);
        std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array = std::get<0>(edt_result);
        auto indexes = std::get<1>(edt_result);

        // writefile("distance.f64", distance_array.data(), distance_array.size());
        sign_map.resize(input_size, 0);  
        auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
                auto [compensate_direction, change_distance] = 
                            check_compensate_direction_distance_3d(i);
                auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
                auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
                int direction = std::distance(change_distance.begin(), min_iter);
                double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
                double grad = grad_computer.get_grad(i);
                if (grad >= 1.0) {
                    sign = 0;
                }
                compensation_map[i] = sign * comepnsation_value;
                sign_map[i] = sign; 
            }
        }
        // complete the sign map
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                char sign = get_sign(compensation_map[indexes[i]]);
                sign_map[i] = sign;
            }
        }
        // dump the sign map 
        // writefile("sign.int8", sign_map.data(), input_size);
        // get the second boundry map 
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());

        // filp and remove the boundary points 
        for (int i = 0; i < input_size; i++) {
            if (boundary_map2[i] == 1 && boundary_mask[i] == false) { 
                boundary_map2[i] = 0;  // boundary lable
            } else {
                boundary_map2[i] = 1;
            }
        }
        // writefile("boundary2.int8", boundary_map2.data(), boundary_map2.size());
        // get the second edt map 

        timer.start();
        edt_omp.reset_timer(); 
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform<double, int>(boundary_map2.data(), 
                                            N, dims.data(), edt_thread_num);
        std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array2 = std::get<0>(edt_result2);
        auto indexes2 = std::get<1>(edt_result2);
        // dump the distance array
        // writefile("distance1.f32", distance_array.data(), input_size);
        // writefile("distance2.f32", distance_array2.data(), input_size);
        for (size_t i = 0; i < input_size; i++) {
            // old method 
            // if (boundary_map[i] == 1)  // non-boundary points ·
            if(1)
            {
                double distance1 = distance_array[i] + 0.5; 
                double distance2 = distance_array2[i] + 0.5 ;
                char sign = sign_map[i];
                double width = distance2 + distance1;
                double relative_r = (distance1 ) / (width);
                // double magnitude = (1 - relative_r) * (1 - relative_r);
                // double magnitude = std::pow(1 - relative_r, 1.5);
                double magnitude = (1/distance1) / (1/distance1 + 1/distance2); 
                compensation_map[i] = sign * magnitude * comepnsation_value;  
                // int ii = i / (dims[1]*dims[2]);
                // int jj = (i / dims[2]) % dims[1];
                // int kk = i % dims[2];
                // if( ii == 81  && jj == 35 && kk == 275)
                // {
                //     std::cout << "### collision grad happned " << std::endl;
                // }
            }

            // new method 
            // if(boundary_map[i] == 1)
            // {
            //     double distance1 = distance_array[i] + 0.5; 
            //     double distance2 = distance_array2[i] + 0.5 ;
            //     char sign = sign_map[i];
            //     int old_x = indexes[i] / (dims[1] * dims[2]), old_y = (indexes[i] / dims[2]) % dims[1], old_z = indexes[i] % dims[2]; 
            //     int new_x = indexes2[i]/ (dims[1] * dims[2]), new_y = (indexes2[i] / dims[2]) % dims[1], new_z = indexes2[i] % dims[2]; 
            //     int cur_x = i / (dims[1] * dims[2]), cur_y = (i / dims[2]) % dims[1], cur_z = i % dims[2]; 
            //     double dot_product = (old_x - cur_x) * (new_x - cur_x) +
            //                              (old_y - cur_y) * (new_y - cur_y) + 
            //                                     (old_z - cur_z) * (new_z - cur_z); 
            //     double magnitude = 0; 
            //     if (dot_product <=0  && distance2 <= distance1)
            //     {
            //         double width = distance2 + distance1;
            //         double relative_r = (distance1 ) / (width);
            //         magnitude = std::pow(1 - relative_r, 1.5);
            //     }
            //     else if ( dot_product <=0 && distance2 > distance1)
            //     {
            //         magnitude = std::exp(-0.3*distance1); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
           
            //     }
            //     else if (dot_product > 0 && distance2 <= distance1)
            //     {
            //         // double width = distance2 + distance1;
            //         // double relative_r = (distance1 ) / (width);
            //         // magnitude = std::pow(1 - relative_r, 1.5);    
            //         magnitude = std::exp(-0.3*distance1); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
           
            //     }
            //     else if (dot_product > 0 && distance2 > distance1)
            //     {
            //         magnitude = std::exp(-0.3*distance1); 
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
            //     }
            //     else{
            //         magnitude = std::exp(-0.3*distance1);  
            //         magnitude = (1/distance1) / (1/distance1 + 1/distance2);
            //     }
            //     compensation_map[i] = sign * magnitude * comepnsation_value;
            // }
        }
        return compensation_map;
    }


    // std::vector<T_data> get_compensation_map_3d_() {
    //     auto boundary_map = get_boundary(quant_index, N, dims.data());
    //     // flip the boundary map tag
    //     for (int i = 0; i < input_size; i++) {
    //         if (boundary_map[i] == 1) {
    //             boundary_map[i] = 0;
    //         } else {
    //             boundary_map[i] = 1;
    //         }
    //     }
    //     auto timer = Timer();
    //     timer.start();

    //     auto edt_result = NI_EuclideanFeatureTransform<double, int>(boundary_map.data(), N, dims.data());
    //     // std::cout << "edt time = " << timer.stop() << std::endl;
    //     auto distance_array = std::get<0>(edt_result);
    //     auto indexes = std::get<1>(edt_result);
    //     std::cout << "compenstion_value = " << comepnsation_value << std::endl;

    //     timer.start();
    //     for (size_t i = 0; i < input_size; i++) {
    //         if (boundary_map[i] == 0)  // boundary points
    //         {
    //             auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
    //             auto [compensate_direction, change_distance] = check_compensate_direction_distance_3d(i);
    //             auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
    //             auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
    //             int direction = std::distance(change_distance.begin(), min_iter);
    //             double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
    //             double grad = grad_computer.get_grad(i);
    //             if (grad >= 1) {
    //                 sign = 0;
    //             }
    //             if (i == 3315383) {
    //                 std::cout << "grad = " << grad << std::endl;
    //                 std::cout << "sign = " << sign << std::endl;
    //                 std::cout << "change distance = " << std::endl;
    //                 for (int j = 0; j < 6; j++) {
    //                     std::cout << " " << change_distance[j] << " ";
    //                 }
    //                 std::cout << std::endl;
    //                 std::cout << "compensate_direction = " << std::endl;
    //                 for (int j = 0; j < 6; j++) {
    //                     std::cout << " " << (int)compensate_direction[j] << " ";
    //                 }
    //                 std::cout << std::endl;
    //                 std::cout << "sign = " << (int)sign << std::endl;
    //                 std::cout << "compensation = " << sign * comepnsation_value << std::endl;
    //             }
    //             compensation_map[i] = sign * comepnsation_value;
    //         }
    //     }
    //     std::cout << "edge compensation time = " << timer.stop() << std::endl;

    //     std::cout << "compensation_map[0] = " << compensation_map[0] << std::endl;
    //     timer.start();

    //     for (size_t i = 0; i < input_size; i++) {
    //         if (boundary_map[i] == 1)  // non-boundary points ·
    //         {
    //             double min_distance = distance_array[i];
    //             char sign = get_sign(compensation_map[indexes[i]]);
    //             // std::cout << "sign = " << (int) sign << std::endl;
    //             double distance_to_opposoite_boundary = find_opposite_distance_3d(
    //                 distance_array.data(), indexes.data(), boundary_map.data(), i, indexes[i], compensation_map.data());
    //             double width = distance_to_opposoite_boundary + min_distance;
    //             double magnitude = 1.0 / (width * width) * (width - min_distance) * (width - min_distance);
    //             compensation_map[i] = sign * magnitude * comepnsation_value;
    //         }
    //     }
    //     std::cout << "non-boundary compensation time = " << timer.stop() << std::endl;
    //     std::cout << "compensation_map[0] = " << compensation_map[0] << std::endl;

    //     return compensation_map;
    // }

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
    int edt_thread_num = 8; // the thread number for edt computing
    double edt_time = 0.0; 
    
};
}  // namespace PM

#endif  // COMPENSATION_HPP