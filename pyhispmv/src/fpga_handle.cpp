#include "fpga_handle.h"

inline void fill(const float* src, std::vector<xrt::bo>& dst, size_t src_size) {
    auto num_ch = dst.size();
    size_t count = src_size / FpgaHandle::FP32_PER_CH;  // Number of 16-element chunks

    // Copy full FP32_PER_CH chunks
    for (size_t i = 0; i < count; ++i) {
        auto* dst_ptr = dst[i%num_ch].map<float*>() + (i / num_ch) * FpgaHandle::FP32_PER_CH;
        std::memcpy(dst_ptr, src + i * FpgaHandle::FP32_PER_CH, FpgaHandle::FP32_PER_CH * sizeof(float));  // Copy 16 elements
    }

    // Copy remaining elements one by one
    for (size_t i = count * FpgaHandle::FP32_PER_CH; i < src_size; ++i) {
        int addr = (i / FpgaHandle::FP32_PER_CH / num_ch) * FpgaHandle::FP32_PER_CH + (i % FpgaHandle::FP32_PER_CH);
        auto* dst_ptr = dst[i%num_ch].map<float*>();
        dst_ptr[addr] = src[i];
    }
}


inline void fill(std::vector<xrt::bo>& src, float* dst, size_t dst_size) {
    auto num_ch = src.size();
    size_t count = dst_size / FpgaHandle::FP32_PER_CH;  // Number of 16-element chunks

    // Copy full FP32_PER_CH chunks
    for (size_t i = 0; i < count; ++i) {
        auto* src_ptr = src[i%num_ch].map<float*>() + (i / num_ch) * FpgaHandle::FP32_PER_CH;
        std::memcpy(dst + i * FpgaHandle::FP32_PER_CH, src_ptr, FpgaHandle::FP32_PER_CH * sizeof(float));  // Copy 16 elements
    }

    // Copy remaining elements one by one
    for (size_t i = count * FpgaHandle::FP32_PER_CH; i < dst_size; ++i) {
        int addr = (i / FpgaHandle::FP32_PER_CH / num_ch) * FpgaHandle::FP32_PER_CH + (i % FpgaHandle::FP32_PER_CH);
        auto* src_ptr = src[i%num_ch].map<float*>();
        dst[i] = src_ptr[addr];
    }
}

FpgaHandle::FpgaHandle(
    const std::string& xclbin_path,
    int id,
    int a, int b, int c,
    int urams, int fp_acc_latency,
    bool dense, bool pre_acc, bool row_dist)
{
    try {
        // Open the FPGA device
        std::cout << "Opening the device with id: " << id << std::endl;

        if (id < 0) {
            throw std::invalid_argument("Device ID must be a non-negative integer.");
        }

        device = xrt::device(id);
        std::cout << "Device name: " << device.get_info<xrt::info::device::name>() << "\n";
        std::cout << "Device BDF: " << device.get_info<xrt::info::device::bdf>() << "\n";
    } catch (const std::runtime_error& e) {
        std::cerr << "XRT error while opening device: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing FPGA device: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Load the xclbin
    try {
        std::cout << "Loading the xclbin: " << xclbin_path << std::endl;

        if (xclbin_path.empty()) {
            throw std::invalid_argument("XCLBIN path is empty.");
        }

        auto uuid = device.load_xclbin(xclbin_path);
        std::cout << "XCLBIN loaded successfully.\n";

        // Initialize the kernel
        std::cout << "Initializing kernel: SpMV\n";
        krnl = xrt::kernel(device, uuid, "SpMV", xrt::kernel::cu_access_mode::exclusive);
        run = xrt::run(krnl);
        std::cout << "Kernel initialized successfully.\n";
    } catch (const std::runtime_error& e) {
        std::cerr << "XRT error while loading SpMV kernel from xclbin: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    } catch (const std::exception& e) {
        std::cerr << "Error loading xclbin or initializing kernel: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Set hardware configuration
    num_ch_A = a;
    num_ch_B = b;
    num_ch_C = c;
    urams_per_pe = urams;
    this->fp_acc_latency = fp_acc_latency;
    dense_overlay = dense;
    pre_accumulator = pre_acc;
    row_dist_net = row_dist;

    // Hold -1 here to indicate a matrix is not selected
    selected_matrix = -1;

    // Initialize member vectors
    mtx_handles.clear();
    mtx_offsets.clear();
    mtx_buffers.clear();
    in_buffers.clear();
    bias_buffers.clear();
    out_buffers.clear();
    // Initialize first matrix offset is always 0
    mtx_offsets.push_back(0);

    // Instantiate device buffers for all A channels
    int arg_num = 0;
    for (int i = 0; i < num_ch_A; i++, arg_num++) {
        xrt::bo device_buffer = xrt::bo(device, (uint32_t)MAX_BUFFER_SIZE_BYTES, krnl.group_id(arg_num));
        mtx_buffers.push_back(device_buffer);
        run.set_arg(arg_num, device_buffer);
    }
    std::cout << "Matrix buffers for A channels created successfully.\n";

    for (int i = 0; i < num_ch_B; i++, arg_num++) {
        xrt::bo device_buffer = xrt::bo(device, (uint32_t)MAX_BUFFER_SIZE_BYTES, krnl.group_id(arg_num));
        in_buffers.push_back(device_buffer);
        run.set_arg(arg_num, device_buffer);
    }
    std::cout << "Vector buffers for B channels created successfully.\n";

    for (int i = 0; i < num_ch_C; i++, arg_num++) {
        xrt::bo device_buffer = xrt::bo(device, (uint32_t)MAX_BUFFER_SIZE_BYTES, krnl.group_id(arg_num));
        bias_buffers.push_back(device_buffer);
        run.set_arg(arg_num, device_buffer);
    }
    std::cout << "Vector buffers for C in channels created successfully.\n";

    for (int i = 0; i < num_ch_C; i++, arg_num++) {
        xrt::bo device_buffer = xrt::bo(device, (uint32_t)MAX_BUFFER_SIZE_BYTES, krnl.group_id(arg_num));
        out_buffers.push_back(device_buffer);
        run.set_arg(arg_num, device_buffer);
    }
    std::cout << "Vector buffers for C out channels created successfully.\n";

    // Allocate memory for input, output and bias buffers before hand
    std::cout << "FPGA handle initialized with configuration:\n"
              << "  num_ch_A: " << num_ch_A << "\n"
              << "  num_ch_B: " << num_ch_B << "\n"
              << "  num_ch_C: " << num_ch_C << "\n"
              << "  ch_width: " << CH_WIDTH << "\n"
              << "  urams_per_pe: " << urams_per_pe << "\n"
              << "  fp_acc_latency: " << fp_acc_latency << "\n"
              << "  dense_overlay: " << (dense_overlay ? "true" : "false") << "\n"
              << "  pre_accumulator: " << (pre_accumulator ? "true" : "false") << "\n"
              << "  row_dist_net: " << (row_dist_net ? "true" : "false") << "\n";
}

int FpgaHandle::createSparseMtxHandle(
    const py::array_t<int>& coo_rows, 
    const py::array_t<int>& coo_cols, 
    const py::array_t<float>& coo_values, 
    const int rows, 
    const int cols) 
{
    // Convert PyArray to standard vectors
    auto rows_buffer = coo_rows.request();
    const int* rows_ptr = static_cast<int*>(rows_buffer.ptr);
    std::vector<int> rows_vec(rows_ptr, rows_ptr + rows_buffer.shape[0]);

    auto cols_buffer = coo_cols.request();
    const int* cols_ptr = static_cast<int*>(cols_buffer.ptr);
    std::vector<int> cols_vec(cols_ptr, cols_ptr + cols_buffer.shape[0]);

    auto values_buffer = coo_values.request();
    const float* values_ptr = static_cast<float*>(values_buffer.ptr);
    std::vector<float> values_vec(values_ptr, values_ptr + values_buffer.shape[0]);

    // Create and prepare sparse matrix
    auto* sparse_handle = new HiSpmvHandle(
        num_ch_A, num_ch_B, num_ch_C, 
        CH_WIDTH, urams_per_pe, fp_acc_latency, 
        dense_overlay, pre_accumulator, row_dist_net);

    double prep_time = sparse_handle->prepareSparseMtxForFPGA(rows, cols, rows_vec, cols_vec, values_vec);
    std::cout << "Sparse matrix preparation completed in " << prep_time << " ns.\n";

    // Get prepared data
    auto& prep_data = sparse_handle->getPreparedMtx();

    // Check if the data can fit and populate buffers
    uint32_t offset_size = mtx_offsets.back();
    uint32_t buffer_size = prep_data[0].size() * sizeof(uint64_t);

    if (offset_size + buffer_size > MAX_BUFFER_SIZE_BYTES) {
        delete sparse_handle;
        return -1;
    }

    for (int i = 0; i < num_ch_A; i++) {
        std::memcpy(mtx_buffers[i].map<char*>() + offset_size, prep_data[i].data(), buffer_size);
    }

    //offset for next handle
    mtx_offsets.push_back(offset_size + buffer_size);

    // Store handle and return index
    mtx_handles.push_back(sparse_handle);
    return mtx_handles.size() - 1;
}

int FpgaHandle::createDenseMtxHandle(
    const py::array_t<float>& flattened_dense_values, 
    const int rows, 
    const int cols) 
{
    // Convert PyArray to standard vector
    auto buffer = flattened_dense_values.request();
    const float* data_ptr = static_cast<float*>(buffer.ptr);
    std::vector<float> dense_values(data_ptr, data_ptr + buffer.shape[0]);

    // Create and prepare dense matrix
    auto* dense_handle = new HiSpmvHandle(
        num_ch_A, num_ch_B, num_ch_C, 
        CH_WIDTH, urams_per_pe, fp_acc_latency, 
        dense_overlay, pre_accumulator, row_dist_net);

    double prep_time = dense_handle->prepareDenseMtxForFPGA(rows, cols, dense_values);
    std::cout << "Dense matrix preparation completed in " << prep_time << " ns.\n";

    // Get prepared data
    auto& prep_data = dense_handle->getPreparedMtx();

    // Check if the data can fit and populate buffers
    uint32_t offset_size = mtx_offsets.back();
    uint32_t buffer_size = prep_data[0].size() * sizeof(uint64_t);

    if (offset_size + buffer_size > MAX_BUFFER_SIZE_BYTES) {
        delete dense_handle;
        return -1;
    }

    for (int i = 0; i < num_ch_A; i++) {
        std::memcpy(mtx_buffers[i].map<char*>() + offset_size, prep_data[i].data(), buffer_size);
    }

    //offset for next handle
    mtx_offsets.push_back(offset_size + buffer_size);

    // Store handle and return index
    mtx_handles.push_back(dense_handle);
    return mtx_handles.size() - 1;
}

void FpgaHandle::loadMatrices() {
    // Sync the matrix buffers after loading all data
    for (int i = 0; i < num_ch_A; i++) {
        mtx_buffers[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Convert offset size to offset length
    for (auto& offset : mtx_offsets) {
        offset /= PES_PER_CH * sizeof(uint64_t);  
    }
    
    std::cout << "Matrix data successfully synced to the device." << std::endl;
}

void FpgaHandle::selectMatrix(const uint32_t matrix_idx) {
    if (matrix_idx >= mtx_handles.size()) {
        std::cerr << "Matrix idx out of range" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto* handle = mtx_handles[matrix_idx];
    int arg_num = num_ch_A + num_ch_B + 2*num_ch_C + 2;
    run.set_arg(arg_num++, mtx_offsets[matrix_idx]);
    run.set_arg(arg_num++, handle->getRunLength()); 
    run.set_arg(arg_num++, handle->getRowsPerPE());
    run.set_arg(arg_num++, handle->getInputLength());
    run.set_arg(arg_num++, handle->getRowTiles());
    run.set_arg(arg_num++, handle->getColTiles());
    run.set_arg(arg_num++, handle->getTotTiles());
    run.set_arg(arg_num++, (uint32_t)1); // repeat time
    run.set_arg(arg_num++, handle->isDense());
    selected_matrix = matrix_idx;
}


void FpgaHandle::runKernel(
    const py::array_t<float>& x_arr, 
    const py::array_t<float>& bias_arr, 
    py::array_t<float>& y_arr,
    const float alpha, const float beta) 
{
    assert(selected_matrix != -1 && "Run Kernel called before selecting a matrix");
    auto* handle = mtx_handles[selected_matrix];
    auto [rows, cols] = handle->getMatrixDims();
    auto [pad_rows, pad_cols] = handle->getPaddedMatrixDims();
    auto row_buf_size = (pad_rows / num_ch_C) * sizeof(float);
    auto col_buf_size = (pad_cols / num_ch_B) * sizeof(float);

    auto x = x_arr.request();
    const auto* x_ptr = static_cast<const float*>(x.ptr);
    auto bias = bias_arr.request();
    const auto* bias_ptr = static_cast<const float*>(bias.ptr);
    auto y = y_arr.request();
    float* y_ptr = static_cast<float*>(y.ptr);

    fill(x_ptr, in_buffers, cols);
    for(auto& bo: in_buffers) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, col_buf_size, 0);

    fill(bias_ptr, bias_buffers, rows);
    for(auto& bo: bias_buffers) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, row_buf_size, 0);

    int arg_num = num_ch_A + num_ch_B + 2*num_ch_C;
    run.set_arg(arg_num++, alpha);
    run.set_arg(arg_num++, beta);

    run.start();
    run.wait();

    for(auto& bo: out_buffers) bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, row_buf_size, 0);
    fill(out_buffers, y_ptr, rows);
}

py::array_t<float> FpgaHandle::runLinear(const int matrix_idx, const py::array_t<float>& x_arr, const py::array_t<float>& bias_arr) {
    // get the handle for selected matrix
    assert(matrix_idx != -1 && "Run Kernels called before selecting a matrix");
    auto* handle = mtx_handles[matrix_idx];
    auto [rows, cols] = handle->getMatrixDims();
    auto [pad_rows, pad_cols] = handle->getPaddedMatrixDims();
    auto row_buf_size = (pad_rows / num_ch_C) * sizeof(float);
    auto col_buf_size = (pad_cols / num_ch_B) * sizeof(float);

    auto x = x_arr.request();
    const auto* x_ptr = static_cast<const float*>(x.ptr);
    auto bias = bias_arr.request();
    const auto* bias_ptr = static_cast<const float*>(bias.ptr);
    int num_vecs = x.shape[0] / cols;

    //output array
    auto y_arr = py::array_t<float>(num_vecs*rows);
    auto y = y_arr.request();
    float* y_ptr = static_cast<float*>(y.ptr);
    int vec_idx = 0;

    fill(x_ptr + (vec_idx * cols), in_buffers, cols);
    for(auto& bo: in_buffers) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, col_buf_size, 0);

    fill(bias_ptr, bias_buffers, rows);
    for(auto& bo: bias_buffers) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, row_buf_size, 0);

    int arg_num = num_ch_A + num_ch_B + 2*num_ch_C;
    run.set_arg(arg_num++, 1.0f);
    run.set_arg(arg_num++, 1.0f);
    run.set_arg(arg_num++, mtx_offsets[matrix_idx]);
    run.set_arg(arg_num++, handle->getRunLength()); 
    run.set_arg(arg_num++, handle->getRowsPerPE());
    run.set_arg(arg_num++, handle->getInputLength());
    run.set_arg(arg_num++, handle->getRowTiles());
    run.set_arg(arg_num++, handle->getColTiles());
    run.set_arg(arg_num++, handle->getTotTiles());
    run.set_arg(arg_num++, (uint32_t)1); // repeat time
    run.set_arg(arg_num++, handle->isDense());

    run.start();
    
    // Enter this loop only if the num_vecs is greater than 1 
    for (vec_idx = 1; vec_idx < num_vecs; vec_idx++) {
        // prepare next input, py_arr to aligned host addr while waiting for run
        fill(x_ptr + (vec_idx * cols), in_buffers, cols);

        run.wait();

        for(auto& bo: out_buffers) bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, row_buf_size, 0);
        for(auto& bo: in_buffers) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, col_buf_size, 0);

        run.start();

        // store current output after launching next run, alighned host ptr to py arr
        fill(out_buffers, y_ptr + (vec_idx - 1) * rows, rows);
    }

    // stop the run and collect the final output vec_idx = 1 in case of single vec or num_vecs if multiple vecs
    run.wait();

    for(auto& bo: out_buffers) bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, row_buf_size, 0);
    fill(out_buffers, y_ptr + (vec_idx - 1) * rows, rows);

    return y_arr;  // Return the output py arr
}