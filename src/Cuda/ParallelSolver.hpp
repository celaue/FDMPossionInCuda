#ifndef CUDA_HPP
#define CUDA_HPP

#include <assert.h>
#include <iostream>
#include <cusolverDn.h>
#include <Eigen/Dense>
#include <vector>
#include <cuda_runtime.h>


namespace CUDA {
    //calculate solution of linear system (Ax = b) using LU decompostion with parital pivoting ( LU = PA )
    Eigen::VectorXd parallel_LU_pivot(Eigen::MatrixXd &A,Eigen::VectorXd &b);

    //calculate solution of linear system (Ax = b) using iterative Jacobi method with "error" tolerance
    Eigen::VectorXd parallel_Jacobi_method(Eigen::MatrixXd &A,Eigen::VectorXd &b,double error);
}




#endif