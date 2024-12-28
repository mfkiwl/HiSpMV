#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include "spmv-helper.h"

class FpgaHelper {
private:
    // Maximum HBM Bank size 256 Mega bytes
    static constexpr const unsigned MAX_BUFFER_SIZE_BYTES = 256 * 1024 * 1024;

    // Xilinx runtime parameters
    xrt::device device;
    xrt::kernel krnl;
    xrt::run run;

    // vector of SpMVHelper instances to store prepared sparse/dense matrix along with hardware info
    std::vector<SpMVHelper*> matrices;
    int num_ch_A;
    int num_ch_B;
    int num_ch_C;

    // loaded matix offsets  
    std::vector<uint32_t> matrix_offsets;

    void _setInputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num);

    xrt::bo _setOutputBuffer(void* host_aligned_ptr, uint32_t, const int arg_num);

public:
    // Constructor requires xclbin and fpga device to initialize SpMV kernel
    FpgaHelper(const std::string& xclbin_path, const std::string& device_id, const int A_hbm_ch, const int B_hbm_ch, const int C_hbm_ch);

    // Method to store multiple sparse/dense matrix in FPGA, returns number of matrices loaded successfully
    uint32_t loadMatricesToDevice(const std::vector<SpMVHelper*>& inputs);

    // Set Matrix and it's properties seperately using matrix_idx
    void selectMatrix(const uint32_t matrix_idx);

    // Run SpMV kenrel with given input/output vectors and scalars 
    double runKernel(
        std::vector<aligned_vector<float>>& B, 
        std::vector<aligned_vector<float>>& Cin,
        std::vector<aligned_vector<float>>& Cout,
        const float alpha, const float beta);
};