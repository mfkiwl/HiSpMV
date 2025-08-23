#include "helper_functions.h"
#include <mkl.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// ---------------- Sparse CPU SpMV (reference) ----------------
void cpu_spmv(const CSRMatrix& A, const vector<float> B, vector<float>& C,
              const float alpha, const float beta, const int rp_time) {
    for (int r = 0; r < rp_time; r++) {
        for (int i = 0; i < (int)A.row_offsets.size() - 1; ++i) {
            C[i] *= beta;
            for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
                int colIndex = A.col_indices[j];
                float value = A.values[j];
                C[i] += alpha * value * B[colIndex];
            }
        }
    }
}

// ---------------- MKL SpMV ----------------
double mkl_spmv(const CSRMatrix& matrix, const int rows, const int cols, const int nnz,
                const std::vector<float>& x, std::vector<float>& y,
                float alpha, float beta, int rp_time) {
    sparse_matrix_t A;
    mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, rows, cols,
                            (int*)&matrix.row_offsets[0], (int*)&matrix.row_offsets[1],
                            (int*)&matrix.col_indices[0], (float*)&matrix.values[0]);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < rp_time; ++i) {
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr,
                        (float*)x.data(), beta, (float*)y.data());
    }
    auto end_time = std::chrono::steady_clock::now();

    mkl_sparse_destroy(A);

    double total_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return total_time_ns / rp_time;
}

// ---------------- Naive Dense GEMV (reference) ----------------
// Row-major A, y <- alpha*A*x + beta*y
double naive_gemv(int rows, int cols, const std::vector<float>& A,
                  const std::vector<float>& x, std::vector<float>& y,
                  float alpha, float beta, int rp_time) {
    auto start_time = std::chrono::steady_clock::now();
    for (int r = 0; r < rp_time; ++r) {
        for (int i = 0; i < rows; ++i) {
            float acc = 0.0f;
            const float* Ai = &A[i * cols];
            for (int j = 0; j < cols; ++j) {
                acc += Ai[j] * x[j];
            }
            y[i] = alpha * acc + beta * y[i];
        }
    }
    auto end_time = std::chrono::steady_clock::now();
    double total_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return total_time_ns / rp_time;
}

// ---------------- MKL Dense GEMV ----------------
double mkl_gemv(int rows, int cols, const std::vector<float>& A,
                const std::vector<float>& x, std::vector<float>& y,
                float alpha, float beta, int rp_time) {
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    auto powerLogger = PowerLogger();

    powerLogger.start(); // start measuring power
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < rp_time; ++i) {
        cblas_sgemv(layout, trans, rows, cols, alpha,
                    A.data(), cols, x.data(), 1, beta, y.data(), 1);
    }
    auto end_time = std::chrono::steady_clock::now();
    powerLogger.stop();
    printf("Average Power: %lf\n", powerLogger.getAveragePower());
    double total_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return total_time_ns / rp_time;

    
}

// ---------------- Precision Check ----------------
double computePrecisionLoss(const vector<float>& vectorA, const vector<float>& vectorB) {
    if (vectorA.size() != vectorB.size()) {
        printf("Error: Vector sizes do not match!\n");
        return -1;
    }

    double diffSum = 0.0;
    double refSum = 0.0;
    double maxRelativeError = 0.0;
    int maxRelativeErrorIdx = 0;

    for (size_t i = 0; i < vectorA.size(); ++i) {
        double a = vectorA[i], b = vectorB[i];
        double diff = fabs(a - b);
        double ref = std::min(fabs(a), fabs(b));
        double relativeError = 0.0;

        if (a != 0.0 && b != 0.0) {
            relativeError = (diff / ref);
            if (relativeError > maxRelativeError) {
                maxRelativeErrorIdx = (int)i;
                maxRelativeError = relativeError;
            }
        }

        diffSum += diff;
        refSum += ref;
    }

    printf("Max Relative Error: %f Reference: %f Actual: %f index: %d\n",
           maxRelativeError, vectorA[maxRelativeErrorIdx], vectorB[maxRelativeErrorIdx],
           maxRelativeErrorIdx);
    return (refSum > 0.0) ? (diffSum / refSum) : 0.0;
}

// ---------------- Main ----------------
int main(int argc, char* argv[]) {
    mkl_set_num_threads(24);
    int nthreads = mkl_get_max_threads();
    std::cout << "MKL max threads: " << nthreads << std::endl;

    if (argc < 3) {
        printf("Usage:\n");
        printf("  Sparse SpMV: %s <matrix_file.mtx> <repeat>\n", argv[0]);
        printf("  Dense  GEMV: %s <rows> <cols> <repeat>\n", argv[0]);
        return 0;
    }

    const float alpha = 0.85f;
    const float beta  = -2.06f;

    // ======== Sparse case: input is .mtx file ========
    if (argc == 3) {
        char* filename = argv[1];
        int rp_time = atoi(argv[2]);

        printf("Reading %s\n", filename);

        vector<float> cscValues;
        vector<int> cscRowIndices;
        vector<int> cscColOffsets;
        int rows, cols, nnz;
        readMatrixCSC(filename, cscValues, cscRowIndices, cscColOffsets, rows, cols, nnz);

        CSRMatrix cpuAmtx;
        convertCSCtoCSR(cscValues, cscRowIndices, cscColOffsets,
                        cpuAmtx.values, cpuAmtx.col_indices, cpuAmtx.row_offsets,
                        rows, cols, nnz);

        vector<float> x(cols, 0.0f);
        vector<float> y_cpu(rows, 0.0f);
        vector<float> y_mkl(rows, 0.0f);

        // Deterministic init
        for (int j = 0; j < cols; ++j) x[j] = float(j + 1) / float(j + 2);
        for (int i = 0; i < rows; ++i) {
            float v = -2.0f * (i + 1) / float(i + 2);
            y_cpu[i] = v;
            y_mkl[i] = v;
        }

        printf("\nComputing Single Thread CPU SpMV (reference)... \n");
        auto start_cpu = std::chrono::steady_clock::now();
        cpu_spmv(cpuAmtx, x, y_cpu, alpha, beta, rp_time);
        auto end_cpu = std::chrono::steady_clock::now();
        double time_cpu_s =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count() * 1e-9;
        printf("done (%f msec)\n", time_cpu_s * 1000.0);
        printf("CPU GFLOPS: %f\n", 2.0 * (nnz + rows) / 1e+9 / time_cpu_s);

        printf("\nComputing MKL SpMV... \n");
        double time_mkl_ns = mkl_spmv(cpuAmtx, rows, cols, nnz, x, y_mkl, alpha, beta, rp_time);
        double time_mkl_s  = time_mkl_ns * 1e-9;
        printf("done (%f msec)\n", time_mkl_s * 1000.0);
        printf("MKL GFLOPS: %f\n", 2.0 * (nnz + rows) / 1e+9 / time_mkl_s);

        printf("\nComparing Results (CPU vs MKL)... \n");
        double precisionLoss = computePrecisionLoss(y_cpu, y_mkl);
        printf("Precision Loss: %f\n", precisionLoss);
    }
    // ======== Dense case: rows & cols provided ========
    else if (argc == 4) {
        int rows = atoi(argv[1]);
        int cols = atoi(argv[2]);
        int rp_time = atoi(argv[3]);

        printf("Generating deterministic dense matrix %d x %d\n", rows, cols);

        vector<float> A(rows * cols);
        vector<float> x(cols);
        vector<float> y_ref(rows);   // naive reference output (starts from same initial y)
        vector<float> y_mkl(rows);   // MKL output (starts from same initial y)

        // A[i,j] = (i+1)/(j+2)  (row-major)
        for (int i = 0; i < rows; ++i) {
            const int base = i * cols;
            for (int j = 0; j < cols; ++j) {
                A[base + j] = float(i + 1) / float(j + 2);
            }
        }
        // x[j] = (j+1)/(j+2)
        for (int j = 0; j < cols; ++j) x[j] = float(j + 1) / float(j + 2);
        // y initial = -2*(i+1)/(i+2)
        for (int i = 0; i < rows; ++i) {
            float yi = -2.0f * (i + 1) / float(i + 2);
            y_ref[i] = yi;
            y_mkl[i] = yi;
        }

        printf("\nComputing Naive GEMV (reference)... \n");
        double time_naive_ns = naive_gemv(rows, cols, A, x, y_ref, alpha, beta, 1);
        double time_naive_s  = time_naive_ns * 1e-9;
        printf("done (%f msec)\n", time_naive_s * 1000.0);
        // FLOPs per GEMV ≈ 2*rows*cols (mul+add) — beta*y adds ~rows ops; we can ignore or include:
        const double flops_per_call = 2.0 * (double)rows * (double)cols + (double)rows;
        printf("Naive GEMV GFLOPS: %f\n", flops_per_call / 1e9 / time_naive_s);

        printf("\nComputing MKL GEMV... \n");
        double time_mkl_ns = mkl_gemv(rows, cols, A, x, y_mkl, alpha, beta, rp_time);
        double time_mkl_s  = time_mkl_ns * 1e-9;
        printf("done (%f msec)\n", time_mkl_s * 1000.0);
        printf("MKL GEMV GFLOPS: %f\n", flops_per_call / 1e9 / time_mkl_s);

        printf("\nComparing Results (Naive vs MKL)... \n");
        double precisionLoss = computePrecisionLoss(y_ref, y_mkl);
        printf("Precision Loss: %f\n", precisionLoss);
    }

    return 0;
}