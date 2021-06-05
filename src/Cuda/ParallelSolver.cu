

#include "ParallelSolver.hpp"

namespace CUDA {

Eigen::VectorXd parallel_LU_pivot(Eigen::MatrixXd A,Eigen::VectorXd b){
    
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;


    const int m = A.cols();
    const int lda = A.cols();
    const int ldb = b.rows();
 
    Eigen::VectorXd x=Eigen::VectorXd::Zero(m); // x = A\B 
    int info = 0;     // host copy of error info 

    double *d_A = NULL; // device copy of A 
    double *d_b = NULL; // device copy of B 
    int *d_Ipiv = NULL; // pivoting sequence
    int *d_info = NULL; // error info 
    int  lwork = 0;     // size of workspace
    double *d_work = NULL; // device workspace for getrf

    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);
    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /////////////////
    // Copy to GPU //
    /////////////////
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_b, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    cudaStat1 = cudaMemcpy(d_A, A.data(), sizeof(double)*lda*m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_b, b.data(), sizeof(double)*m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);


    /////////////////////
    // Query workspace //
    /////////////////////
    status = cusolverDnDgetrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        &lwork);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    //////////////////////
    // LU factorization //
    //////////////////////
    status = cusolverDnDgetrf(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            d_work,
            d_Ipiv,
            d_info);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);



    /////////////////////
    //  solve A*x = b  //
    /////////////////////
    status = cusolverDnDgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            d_Ipiv,
            d_b,
            ldb,
            d_info);
    
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(x.data(), d_b, sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    /////////////////////
    //  free recourses //
    /////////////////////
    if (d_A    ) cudaFree(d_A);
    if (d_b    ) cudaFree(d_b);
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    cudaDeviceReset();

    return x;
}

}