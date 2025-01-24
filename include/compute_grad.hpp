#ifndef COMPUTE_GRAD_HPP
#define COMPUTE_GRAD_HPP

#include <cstddef>
#include <limits>
#include<vector>
#include<cmath>
#include<tuple>
#include<functional>
#include<iostream> 

template<typename T_data>
class ComputeGrad
{
    public:
        ComputeGrad(int n, int* dims, T_data* data)
        {
            N = n;
            input_data = data;
            input_size = 1;
            for(int i = 0; i < n; i++)
            {
                this->dims.push_back(dims[i]);
                input_size *= dims[i];
            }
            strides.resize(n);
            strides[n-1] = 1;
            for(int i = n-2; i >= 0; i--)
            {
                strides[i] = strides[i+1] * dims[i+1];
            }
        }

        double get_grad(size_t index)
        {
            if(N ==2)
            {
                int i = index / dims[1];
                int j = index % dims[1];
                if(i ==0 || i == dims[0]-1 || j == 0 || j == dims[1]-1)
                {
                    return std::numeric_limits<double>::infinity();
                }
                else
                {
                    // first dim 
                    double dx =  (1.0*input_data[index + strides[0]] - 1.0*input_data[index - strides[0]])/2.0;
                    // second dim
                    double dy =  (1.0*input_data[index + strides[1]] - 1.0*input_data[index - strides[1]])/2.0;
                    return std::sqrt(dx*dx + dy*dy);
                }   
            }
            else if(N == 3)
            {
                int i = index / (dims[1]*dims[2]);
                int j = (index / dims[2]) % dims[1];
                int k = index % dims[2];
                if(i ==0 || i == dims[0]-1 || j == 0 || j == dims[1]-1 || k == 0 || k == dims[2]-1)
                {
                    return std::numeric_limits<double>::infinity();
                }
                else
                {
                    // first dim 
                    
                    double max_grad = -1;
                    double dx =  (1.0*input_data[index + strides[0]] - 1.0*input_data[index - strides[0]])/2.0;
                    max_grad = std::max(max_grad, std::abs(dx));
                    // second dim
                    double dy =  (1.0*input_data[index + strides[1]] - 1.0*input_data[index - strides[1]])/2.0;
                    max_grad = std::max(max_grad, std::abs(dy));
                    // third dim
                    double dz =  (1.0*input_data[index + strides[2]] - 1.0*input_data[index - strides[2]])/2.0;
                    max_grad = std::max(max_grad, std::abs(dz)); 

                    if( i == 81  && j == 35 && k == 275)
                    {
                        std::cout << "##dx = " << dx << std::endl;
                        std::cout << "##dy = " << dy << std::endl;
                        std::cout << "##dz = " << dz << std::endl; 
                        std::cout << "##max_grad = " << max_grad << std::endl; 
                    }
                    return max_grad; 
                    // std::array<double, 3> grad = {std::abs(dx), std::abs(dy), std::abs(dz)};
                    // return *std::max_element(grad.begin(), grad.end());

                }
            }
            else {
                return -1;
            }
        }



    private:
    int N;
    std::vector<int> dims; 
    std::vector<size_t> strides;
    T_data* input_data; 
    size_t input_size;




};






#endif