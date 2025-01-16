#include <torch/torch.h>
#include <iostream>
#include <cmath>

class RBFInterpolator {
public:
    RBFInterpolator(torch::Tensor y, torch::Tensor d, 
                    double smoothing, std::string kernel, double epsilon, int degree)
        : y_(y), d_(d), smoothing_(smoothing), kernel_(std::move(kernel)), 
          epsilon_(epsilon), degree_(degree) {
        preprocess();
    }

    torch::Tensor interpolate(torch::Tensor x) {
        // Compute kernel matrix for input x with training points y
        torch::Tensor kv = computeKernelMatrix(x, y_);
        torch::Tensor p = computePolynomialMatrix(x);
        torch::Tensor vec = torch::cat({kv, p}, /*dim=*/1);
        return torch::matmul(vec, coeffs_);
    }

private:
    torch::Tensor y_, d_, coeffs_, shift_, scale_;
    double smoothing_, epsilon_;
    std::string kernel_;
    int degree_;

    void preprocess() {
        int p = y_.size(0);  // Number of data points
        torch::Tensor kernelMatrix = computeKernelMatrix(y_, y_);
        torch::Tensor polyMatrix = computePolynomialMatrix(y_);

        // Build the left-hand side matrix (RBF system matrix)
        torch::Tensor lhs = torch::zeros({p + polyMatrix.size(0), p + polyMatrix.size(0)}, torch::kFloat);
        lhs.slice(/*dim=*/0, /*start=*/0, /*end=*/p).slice(/*dim=*/1, /*start=*/0, /*end=*/p) = kernelMatrix + smoothing_ * torch::eye(p);
        lhs.slice(/*dim=*/0, /*start=*/0, /*end=*/p).slice(/*dim=*/1, /*start=*/p, /*end=*/p + polyMatrix.size(0)) = polyMatrix;
        lhs.slice(/*dim=*/0, /*start=*/p, /*end=*/p + polyMatrix.size(0)).slice(/*dim=*/1, /*start=*/0, /*end=*/p) = polyMatrix.transpose(0, 1);

        // Build the right-hand side (data)
        torch::Tensor rhs = torch::zeros({p + polyMatrix.size(0), d_.size(1)}, torch::kFloat);
        rhs.slice(/*dim=*/0, /*start=*/0, /*end=*/p) = d_;

        // Solve the linear system
        coeffs_ = torch::linalg::solve(lhs, rhs);
    }

    torch::Tensor computeKernelMatrix(torch::Tensor a, torch::Tensor b) {
        // Compute pairwise Euclidean distances
        torch::Tensor dist = torch::cdist(a, b);  // Euclidean distance matrix

        if (kernel_ == "thin_plate_spline") {
            return dist.pow(2) * dist.log();
        }
        // Add other kernels if needed (e.g., Gaussian, Cubic)
        return torch::zeros_like(dist);  // Default to zero if kernel is unsupported
    }

    torch::Tensor computePolynomialMatrix(torch::Tensor points) {
        // For simplicity, let's assume degree=0 (constant term).
        // For higher-degree, we can generate terms accordingly.
        return torch::ones({points.size(0), 1}, torch::kFloat);  // Polynomial of degree 0
    }
};

int main() {
    // Example usage of the RBFInterpolator with libtorch
    torch::Device device(torch::kCPU);  // Or torch::kCUDA if you want to use GPU

    // Sample data
    torch::Tensor y = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.5, 0.5}}, torch::kFloat).to(device);
    torch::Tensor d = torch::tensor({{0.0}, {1.0}, {1.0}, {0.0}, {0.5}}, torch::kFloat).to(device);

    // Create the RBFInterpolator
    RBFInterpolator interpolator(y, d, 0.1, "thin_plate_spline", 1.0, 0);

    // Query points
    torch::Tensor x = torch::tensor({{0.25, 0.25}, {0.75, 0.75}, {0.5, 0.5}}, torch::kFloat).to(device);

    // Perform interpolation
    torch::Tensor result = interpolator.interpolate(x);

    // Print the results
    std::cout << "Interpolated values:\n" << result << std::endl;

    return 0;
}
