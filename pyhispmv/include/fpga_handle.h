#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include "spmv-helper.h"

namespace py = pybind11;

class FpgaHandle {
public:
    // Maximum HBM Bank size 256 Mega bytes (compile time constants for perfoemance optimization)
    static constexpr unsigned MAX_BUFFER_SIZE_BYTES = 256 * 1024 * 1024;
    static constexpr unsigned CH_WIDTH = 512;
    static constexpr unsigned PES_PER_CH = CH_WIDTH / 64;
    static constexpr unsigned FP32_PER_CH = CH_WIDTH / 32;

private:
    // Xilinx runtime parameters
    xrt::device device;
    xrt::kernel krnl;
    xrt::run run;

    // Hardware configuration
    int num_ch_A;
    int num_ch_B;
    int num_ch_C;
    int urams_per_pe;
    int fp_acc_latency;
    bool dense_overlay;
    bool pre_accumulator;
    bool row_dist_net;

    // vector of HiSpmvHandle instances to store prepared sparse/dense matrix along with hardware info
    std::vector<HiSpmvHandle*> mtx_handles;

    // vector to store buffer objects for A mtx
    std::vector<xrt::bo> mtx_buffers;
    std::vector<xrt::bo> in_buffers;
    std::vector<xrt::bo> bias_buffers;
    std::vector<xrt::bo> out_buffers;

    // loaded matix offsets  
    std::vector<uint32_t> mtx_offsets;

    // variable to store selected mtx
    int selected_matrix;

    void _setInputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num);
    xrt::bo _setOutputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num);

public:
    // Constructor requires xclbin and fpga device to initialize SpMV kernel
    FpgaHandle(const std::string& xclbin_path, int device_id, int a, int b, int c, int urams, int fp_acc_latency, bool dense, bool pre_acc, bool row_dist);

    // Method to create matrix handle for Dense matrix returns index, which can be used to select matrix
    int createDenseMtxHandle(const py::array_t<float>& flattened_dense_values, const int rows, const int cols);

    // Method to create matrix handle for Sparse matrix returns index, which can be used to select matrix
    int createSparseMtxHandle(const py::array_t<int>& coo_rows, const py::array_t<int>& coo_cols, const py::array_t<float>& coo_values, const int rows, const int cols);

    // Method to store multiple sparse/dense matrix in FPGA, returns number of matrices loaded successfully
    void loadMatrices();

    // Set Matrix and it's properties seperately using matrix_idx
    void selectMatrix(const uint32_t matrix_idx);

    // Run SpMV kenrel with given input/output vectors and scalars 
    void runKernel(const py::array_t<float>& x, 
          const py::array_t<float>& bias, 
          py::array_t<float>& y,
        const float alpha, const float beta);
    
    py::array_t<float> runLinear(const int matrix_idx, const py::array_t<float>& x_arr, const py::array_t<float>& bias_arr);
};