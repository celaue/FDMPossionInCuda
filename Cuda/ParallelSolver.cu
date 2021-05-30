
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <Eigen/Dense>

std::tuple<Eigen::Map<Eigen::MatrixXd>,Eigen::Map<Eigen::VectorXi>> parallel_LU_pivot(Eigen::MatrixXd A);

int main(){
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    parallel_LU_pivot(m);

}
std::tuple<Eigen::Map<Eigen::MatrixXd>,Eigen::Map<Eigen::VectorXi>> parallel_LU_pivot(Eigen::MatrixXd A_){

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    int lda = A_.rows(); //max dim of A (here rows)
    int n = A_.cols(); 
    int m = A_.rows();

    double *A = A_.data();
    // double *x = malloc(n*sizeof(double));
    int *Ipiv = (int*) malloc(lda*sizeof(int)); //Array containing pivot indizes
    double *LU =  (double*)malloc(n*m*sizeof(double));
    int info = 0;     /* host copy of error info */

    double *d_A = NULL; /* device copy of A */
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    double *d_work = NULL; /* device workspace for getrf */

    /* step 1: create cusolver handle, bind a stream */
    std::cout<<"1\n";
    status = cusolverDnCreate(&cusolverH);
    std::cout<<status==CUSOLVER_STATUS_NOT_INITIALIZED<<"2";
    // assert(CUSOLVER_STATUS_SUCCESS == status);

    // cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // assert(cudaSuccess == cudaStat1);

    // std::cout<<"2\n";
    // status = cusolverDnSetStream(cusolverH, stream);
    // assert(CUSOLVER_STATUS_SUCCESS == status);

    // /* step 2: copy A to device */
    // cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * n * m);
    // cudaStat3 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * lda);
    // cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    // assert(cudaSuccess == cudaStat1);
    // assert(cudaSuccess == cudaStat2);
    // assert(cudaSuccess == cudaStat3);
    // assert(cudaSuccess == cudaStat4);

    // cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*n*m, cudaMemcpyHostToDevice);
    // assert(cudaSuccess == cudaStat1);
    // assert(cudaSuccess == cudaStat2);

 
    // /* step 3: query working space of getrf */
    // status = cusolverDnDgetrf_bufferSize(
    //     cusolverH,
    //     m,
    //     n,
    //     d_A,
    //     lda,
    //     &lwork);
    // assert(CUSOLVER_STATUS_SUCCESS == status);

    // cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    // assert(cudaSuccess == cudaStat1);


    
    // status = cusolverDnDgetrf(cusolverH,m,n, d_A, lda,d_work,d_Ipiv,d_info);
    // std::cout<<"3\n";
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(CUSOLVER_STATUS_SUCCESS == status);
    // assert(cudaSuccess == cudaStat1);
    // std::cout<<"4\n";
    // cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost); 
    // cudaStat2 = cudaMemcpy(LU   , d_A   , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    // cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    // assert(cudaSuccess == cudaStat1);
    // assert(cudaSuccess == cudaStat2);
    // assert(cudaSuccess == cudaStat3);


    // /*free GPU ressources*/
    // if (d_A) cudaFree(d_A);
    // if (d_Ipiv) cudaFree(d_Ipiv);
    // if (d_info) cudaFree(d_info);
    // if (d_work) cudaFree(d_work);
    // cudaDeviceReset();

    // /*Transform array into Eigen::Matrix*/
    // Eigen::Map<Eigen::MatrixXd> LU_return(LU,m,n);
    // Eigen::Map<Eigen::VectorXi> Ipiv_return(Ipiv,m);

    // /*Free host data*/
    // if (A) free(A);

    // return std::make_tuple(LU_return,Ipiv_return);



}