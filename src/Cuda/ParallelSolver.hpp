#ifndef CUDA_HPP
#define CUDA_HPP

#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <Eigen/Dense>
#include <vector>

namespace CUDA {
    Eigen::VectorXd parallel_LU_pivot(Eigen::MatrixXd A,Eigen::VectorXd b);
}




#endif