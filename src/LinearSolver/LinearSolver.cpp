#include "LinearSolver.hpp"

Eigen::VectorXd LinearSolver::solveJacobianSequential(Eigen::MatrixXd &A,Eigen::VectorXd &b,double tol){
    int n = A.cols();
    std::vector<double> residual(n);
    Eigen::VectorXd x= b;
    unsigned int d_not_tolerent=1;
    double d_marker=0;
    unsigned int d_same=0;
    unsigned int d_pos_of_same=0;
    int stop_after =100000;
    int counter = 0;
    while(counter < stop_after && d_not_tolerent){
        for(int i =0; i<n;i++){
            double new_component=0;
            double zw = 0;
            for(int j=0;j<n;j++){
                zw += A(i,j)*x[j];
            }
            if(A(i,i)!= 0.0){
                new_component = (b[i]- zw)/A(i,i)+x[i];
                residual[i]=new_component-x[i];
            }else{
                residual[i]=0;
            }
            x[i]+=residual[i];
        }
        if(counter%10==0){
            d_not_tolerent = 0;
            for(int i =0; i<n;i++){
                if(std::abs(residual[i])>tol){
                    if(d_marker !=residual[i] || d_pos_of_same != i){
                        d_not_tolerent=1;
                        d_marker = residual[i];
                        d_pos_of_same =i;
                        d_same =0;
                    }else{
                        if(d_same<10){
                            d_not_tolerent=1;
                        }
                        d_same++;
                    }
                }
            }
        }
    }
    return x;
    
    
}