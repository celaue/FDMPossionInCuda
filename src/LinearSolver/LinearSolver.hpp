#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP


#include "Cuda/ParallelSolver.hpp"


class LinearSolver{
    public:
        static Eigen::VectorXd solveLU(Eigen::MatrixXd A,Eigen::VectorXd b,bool parallel=true){
            if(parallel){
                return CUDA::parallel_LU_pivot(A,b);
            }else{
                return A.partialPivLu().solve(b);
            }
        }
        static Eigen::VectorXd solveJacobian(Eigen::MatrixXd &A,Eigen::VectorXd &b,double tol,bool parallel=true){
            if(parallel){
                return CUDA::parallel_Jacobi_method(A,b,tol);
            }else{
                return solveJacobianSequential(A,b,tol);
            }
        }
    
    private:
        static Eigen::VectorXd solveJacobianSequential(Eigen::MatrixXd &A,Eigen::VectorXd &b,double tol); 
};



#endif