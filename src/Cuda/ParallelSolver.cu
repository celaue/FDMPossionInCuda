#include "ParallelSolver.hpp"


//Define device variables for stopping condition
__device__ unsigned int d_not_tolerent;
__device__  double d_marker;
__device__ unsigned int d_same;
__device__ unsigned int d_pos_of_same;

__global__ void reset_d_not_tolerent (){
    d_not_tolerent = 0;
}



//Calculate jacobi step for each element seperatly
__global__ void calc_jacobi_step(int n,double *A,double *b,double *x, double *residual){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double new_component=0;
    double zw = 0;
    for(int j=0;j<n;j++){
        zw += A[j*n+i]*x[j];
    }
    if(A[i*n+i]!= 0.0){
        new_component = (b[i]- zw)/A[i*n+i]+x[i];
        residual[i]=new_component-x[i];
    }else{
        residual[i]=0;
    }
    
}


//Check if solution has converged and update new solution
__global__ void update_and_check_tol(double *x, double *residual,double tol){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    x[i]+=residual[i];
}


//external functions
namespace CUDA {

Eigen::VectorXd parallel_LU_pivot(Eigen::MatrixXd &A,Eigen::VectorXd &b){
    
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

    double *d_A = nullptr; // device copy of A 
    double *d_b = nullptr; // device copy of B 
    int *d_Ipiv = nullptr; // pivoting sequence
    int *d_info = nullptr; // error info 
    int  lwork = 0;     // size of workspace
    double *d_work = nullptr; // device workspace for getrf

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




Eigen::VectorXd parallel_Jacobi_method(Eigen::MatrixXd &A,Eigen::VectorXd &b,double error){
    
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;
    cudaError_t cudaStat7 = cudaSuccess;
    int n = A.cols();
    Eigen::VectorXd x_0 = b;

    double *d_A = nullptr; // device copy of A 
    double *d_b = nullptr; // device copy of b
    double  *d_x = nullptr; // iterative solution
    double  *d_residual = nullptr;
    bool  *d_isfinished = nullptr;
    bool  *d_component_finished =nullptr;


    /////////////////
    // Copy to GPU //
    /////////////////
    cudaStat1 = cudaMalloc (&d_A, sizeof(double)*n*n);
    cudaStat2 = cudaMalloc (&d_b, sizeof(double)*n);
    cudaStat3 = cudaMalloc (&d_x, sizeof(double)*n);
    cudaStat4 = cudaMalloc (&d_residual, sizeof(double)*n);
    cudaStat5 = cudaMalloc (&d_isfinished, sizeof(bool));
    // cudaStat6 = cudaMalloc (&d_n, sizeof(int));
    cudaStat7 = cudaMalloc (&d_component_finished, sizeof(bool)*n);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat6);
    assert(cudaSuccess == cudaStat7);
    cudaStat1 = cudaMemcpy(d_A, A.data(), sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_b, b.data(), sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_x, x_0.data(), sizeof(double)*n, cudaMemcpyHostToDevice);
    // cudaStat4 = cudaMemcpy(d_x, &n, sizeof(int), cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    int blockSize = 16; //best performance for 16 threads
    int numBlocks = (n + blockSize - 1) / blockSize;
    int stop_after =100000;
    int counter = 0;
    typeof(d_not_tolerent) h_not_tolerent=1;
    ////////////////////////////
    // Calculate jacobi steps //
    ////////////////////////////
    while(counter < stop_after && h_not_tolerent){
        calc_jacobi_step<<<numBlocks, blockSize>>>(n,d_A,d_b,d_x,d_residual);
        update_and_check_tol<<<numBlocks, blockSize>>>(d_x, d_residual, error);

        if(counter%10 ==0){
            cudaMemcpyFromSymbol(&h_not_tolerent, d_not_tolerent, sizeof(d_not_tolerent)); 
            reset_d_not_tolerent<<<1, 1>>>(); 
        }
        counter++;
    }

    //Copy solution to host
    cudaStat1 = cudaMemcpy(x_0.data(), d_x, sizeof(double)*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    /////////////////////
    //  free recourses //
    /////////////////////
    if (d_A    ) cudaFree(d_A);
    if (d_b    ) cudaFree(d_b);
    if (d_x    ) cudaFree(d_x);
    if (d_residual    ) cudaFree(d_residual);
    if (d_isfinished    ) cudaFree(d_isfinished);
    if (d_component_finished    ) cudaFree(d_component_finished);

    cudaDeviceReset();

    return x_0;
}



}


