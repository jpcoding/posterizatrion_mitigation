#include <cmath>
#include <cstddef>
#include <type_traits>
#include<vector> 
#include<iostream>
#include<limits>
#include"utils/file_utils.hpp"
#include"utils/timer.hpp"

double df(int x, int i, double g_i )
{
    return (double) ((x-i)*(x-i))+ g_i*g_i;
}

int Sep(int i , int u , double g_u, double g_i)
{
    double result = ((u*u - i*i + g_u*g_u - g_i*g_i)/(2*(u-i)));
    if(result == std::numeric_limits<double>::infinity())
    {
        return std::numeric_limits<int>::max()-1; 
    }
    else{
        return (int) result; 
    }
}

std::vector<double> get_edt(char* boundary, int* dims)
{
    int strides[2];
    size_t size = dims[0]*dims[1];
    strides[0] = dims[1];
    strides[1] = 1;
    std::vector<double> distance(size, 0);
    std::vector<double> g(size, 0); 
    for(int i = 0; i < dims[0]; i++)
    {
        if(boundary[i*strides[0]] == 1)
        {
            g[i*strides[0]] = 0;
        }
        else
        {
            g[i*strides[0]] = std::numeric_limits<double>::infinity();
        }

        // forward pass 
        for(int j = 1; j < dims[1]; j++)
        {
            size_t index  = i*strides[0] + j*strides[1];
            if(boundary[index] == 1)
            {
                g[index] = 0;
            }
            else
            {
                g[index] = g[index-strides[1]] + 1;
            }
        }
        // backward pass 
        for(int j = dims[1]-2; j >= 0; j--)
        {
            size_t index = i*strides[0] + j*strides[1]; 
            if(g[index+strides[1]] < g[index])
            {
                g[index] = g[index+strides[1]] + 1;
            } 
        }
    }
    // second dimension  
    for(int y = 0; y < dims[1]; y++)
    {
        int q = 0; 
        std::vector<int> s(dims[0], 0);
        std::vector<int> t(dims[0], 0);
        for (int u = 1; u < dims[0]; u++)
        {

            while (q >=0  && 
                    (df(t[q], s[q], g[s[q]*strides[0] + y*strides[1]]) > 
                    df(t[q], u, g[u*strides[0] + y*strides[1]])))
            {
                q = q -1; 
            }
            if(q<0)
            {
                q = 0;
                s[0] = u; 
            }
            else {
                int w; 
                auto temp = Sep(s[q], u, 
                                g[u*strides[0] + y*strides[1]], 
                                g[s[q]*strides[0] + y*strides[1]]);
                w = temp +1;
                if(w < dims[0])
                {
                    q = q + 1;
                    s[q] = u;
                    t[q] = w;
                }
            }
        }
        for (int u = dims[0]-1; u >=0; u--)
        {

            // std::cout << "sq  " << s[q] << std::endl;
            distance[u*strides[0] + y*strides[1]] = std::sqrt(df(u, s[q],
                                                         g[s[q]*strides[0] + y*strides[1]]));
            if(u == t[q])
            {
                q = q - 1;
            }
        }
    }
    return distance; 
}

// int main()
// {
//     char boundary[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
//     int dims[2] = {3, 3};
//     auto timer = Timer();
//     timer.start(); 
//     std::vector<double> distance = get_edt(boundary, dims);
//     std::cout << "Time taken = " << timer.stop() << std::endl;
//     for(int i = 0; i < distance.size(); i++)
//     {
//         std::cout << "distance[" << i << "] = " << distance[i] << std::endl;
//     }
//     return 0;

// }

int main(int argc, char** argv)
{
    int N = 2;
    int dims[3] = {10000,6700};
    int size = 1;
    for(int i = 0; i < N; i++)
    {
        size *= dims[i];
    }
    std::vector<char> input(size, 0); 
    readfile("/scratch/pji228/useful/direct_quantize/debug/boundary_map_10000x6700.dat", size, input.data());
    
    // flip the input
    for(int i = 0; i< size; i++)
    {
        if(input[i] == 0)
        {
            input[i] = 1;
        }
        else
        {
            input[i] = 0;
        }
    }



    auto timer = Timer(); 
    timer.start();
    std::vector<double> distance = get_edt(input.data(), dims);
    std::cout << "Time taken = " << timer.stop() << std::endl;
    writefile("distance_map.dat", distance.data(), size);
    



    return 0;
}
