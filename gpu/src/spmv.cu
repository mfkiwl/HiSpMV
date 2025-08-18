#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "spmvHelper.h"
#include "nvmlPower.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

double gpuSpMV(const CSRMatrix_t& matrix, const int rows, const int cols, const int nnz, const std::vector<float>& x, std::vector<float>& y,
                  float alpha, float beta, int exec_s, char* filename) {
    // Create sparse matrix handle
    cudaSetDevice(0);

    // Allocate memory on GPU
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, (int*)matrix.row_offsets.data(),
                           (rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, (int*)matrix.col_indices.data(), nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, (float*)matrix.values.data(), nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, (float*)x.data(), cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, (float*)y.data(), rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, rows, cols, nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, rows, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // Run once to get the correct ouput for refernce match
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
    CHECK_CUDA( cudaMemcpy((float*)y.data(), dY, rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaDeviceSynchronize())

    // Run 1000 times to get an estimate of runtime
    auto start_gpu = std::chrono::steady_clock::now();
    // cudaEventRecord(start);
    for(int i = 0; i < 1000; i++)
      CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
    CHECK_CUDA( cudaDeviceSynchronize())
    auto end_gpu = std::chrono::steady_clock::now();
    double time_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu).count() / 1000;
    
    // compute rp time needed to run kerenl for exec_s
    uint32_t rp_time = static_cast<uint32_t>( exec_s * 1e9 / time_gpu ); 

    // Run for exec_s and collect power data every sec
    nvmlAPIRun(filename);
    start_gpu = std::chrono::steady_clock::now();
    for(int i = 0; i < rp_time; i++)
      CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
    CHECK_CUDA( cudaDeviceSynchronize())
    end_gpu = std::chrono::steady_clock::now();
    nvmlAPIEnd();

    std::cout << "using rp time: " << rp_time  << std::endl;
    time_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu).count() / rp_time;
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    return time_gpu;
}

int main(int argc, char** argv) {
    if (argc < 2){
        printf("Invalid Arguments\n");
        return 0; 
    }

    char* filename = argv[1];
    uint16_t exec_s = 1;

    if (argc == 3) 
        exec_s = (uint16_t)atoi(argv[2]);

    // Host COO matrix storage
    std::vector<int> hostRowInd, hostColInd;
    std::vector<float> hostValues;
    int rows, cols, nnz;

    // Load the matrix from .mtx file
    COOMatrix_t matrix = loadMtx(filename);  // Load matrix with loadMtx function
    hostRowInd = matrix.rows;
    hostColInd = matrix.cols;
    hostValues = matrix.values;
    rows = matrix.rows_count;
    cols = matrix.cols_count;
    nnz = matrix.nnz;

    // Generate sequential input vector
    std::vector<float> cpuX(cols);
    generateSequentialVector(cpuX);

    // Generate sequential output vector 
    std::vector<float> cpuY(rows); 
    std::vector<float> gpuY(rows);
    generateSequentialVector(cpuY); 
    generateSequentialVector(gpuY);

    // Set alpha and beta values
    const float alpha = 0.55f;
    const float beta = -2.05f;

    // Perform CPU-based SpMV for verification
    cpuSpMV(rows, nnz, hostRowInd, hostColInd, hostValues, cpuX, cpuY, alpha, beta);

    auto gpuMtx = cooToCsr(matrix);

    double time = gpuSpMV(gpuMtx, rows, cols, nnz, cpuX, gpuY, alpha, beta, exec_s, filename);
    float gflops = (2 * (nnz + rows)) / time;
    std::cout << "Time taken for SpMV: " << time / 1e3 << " us" << std::endl;
    std::cout << "GFLOPs: " << gflops << std::endl;
    // Print error stats
    printErrorStats(cpuY, gpuY);

    return 0;
}

