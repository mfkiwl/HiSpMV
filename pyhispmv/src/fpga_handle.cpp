#include "fpga_handle.h"

FpgaHandle::FpgaHandle(
    const std::string& xclbin_path,
    int id,
    int a, int b, int c,
    int width, int urams, int fp_acc_latency,
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
    ch_width = width;
    urams_per_pe = urams;
    this->fp_acc_latency = fp_acc_latency;
    dense_overlay = dense;
    pre_accumulator = pre_acc;
    row_dist_net = row_dist;

    pes_per_ch = ch_width / 64;
    fp32_per_ch = ch_width / 32;

    // Hold -1 here to indicate a matrix is not selected
    selected_matrix = -1;

    // Initialize member vectors
    mtx_handles.clear();
    mtx_offsets.clear();
    mtx_buffers.clear();

    // Initialize first matrix offset is always 0
    mtx_offsets.push_back(0);

    // Instantiate device buffers for all A channels
    for (int i = 0; i < num_ch_A; i++) {
        xrt::bo device_buffer = xrt::bo(device, (uint32_t)MAX_BUFFER_SIZE_BYTES, krnl.group_id(i));
        mtx_buffers.push_back(device_buffer);
    }
    std::cout << "Matrix buffers for A channels created successfully.\n";


    std::cout << "FPGA handle initialized with configuration:\n"
              << "  num_ch_A: " << num_ch_A << "\n"
              << "  num_ch_B: " << num_ch_B << "\n"
              << "  num_ch_C: " << num_ch_C << "\n"
              << "  ch_width: " << ch_width << "\n"
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
        ch_width, urams_per_pe, fp_acc_latency, 
        dense_overlay, pre_accumulator, row_dist_net);

    double prep_time = sparse_handle->prepareSparseMtxForFPGA(rows, cols, rows_vec, cols_vec, values_vec);
    std::cout << "Sparse matrix preparation completed in " << prep_time << " ms.\n";

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
        ch_width, urams_per_pe, fp_acc_latency, 
        dense_overlay, pre_accumulator, row_dist_net);

    double prep_time = dense_handle->prepareDenseMtxForFPGA(rows, cols, dense_values);
    std::cout << "Dense matrix preparation completed in " << prep_time << " ms.\n";

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

    // Set the buffers as arguments for the kernel
    for (int i = 0; i < num_ch_A; i++) {
        run.set_arg(i, mtx_buffers[i]);
    }

    // Convert offset size to offset length
    for (auto& offset : mtx_offsets) {
        offset /= pes_per_ch * sizeof(uint64_t);  
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
    run.set_arg(arg_num++, handle->getVectLength());
    run.set_arg(arg_num++, handle->getRowTiles());
    run.set_arg(arg_num++, handle->getColTiles());
    run.set_arg(arg_num++, handle->getTotTiles());
    run.set_arg(arg_num++, (uint32_t)1); // repeat time
    run.set_arg(arg_num++, handle->isDense());
    selected_matrix = matrix_idx;
}


void FpgaHandle::_setInputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num) {
    // Allocate memory on the device memory
    xrt::bo device_buffer = xrt::bo(device, host_aligned_ptr, buffer_size, xrt::bo::flags::device_only, krnl.group_id(arg_num));

    // Sync buffer to device
    device_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Set the buffer to appropriate argument number
    run.set_arg(arg_num, device_buffer); 
}

xrt::bo FpgaHandle::_setOutputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num) {
    // Allocate memory on the device memory
    xrt::bo device_buffer = xrt::bo(device, host_aligned_ptr, buffer_size, krnl.group_id(arg_num));

    // Set the buffer to appropriate argument number
    run.set_arg(arg_num, device_buffer); 

    return device_buffer;
}

double FpgaHandle::runKernel(
    const py::array_t<float>& x_arr, 
    const py::array_t<float>& bias_arr, 
    py::array_t<float>& y_arr,
    const float alpha, const float beta) 
{
    assert(selected_matrix != -1 && "Run Kernel called before selecting a matrix");
    auto start_cpu = std::chrono::steady_clock::now();
    int arg_num = num_ch_A;

    // Get raw data pointer and construct std::vector for x (input vector)
    auto x = x_arr.request();
    auto* x_data_ptr = static_cast<float*>(x.ptr);
    std::vector<float> x_vec(x_data_ptr, x_data_ptr + x.shape[0]);

    // Get raw data pointer and construct std::vector for bias
    auto bias = bias_arr.request();
    auto* bias_data_ptr = static_cast<float*>(bias.ptr);
    std::vector<float> bias_vec(bias_data_ptr, bias_data_ptr + bias.shape[0]);

    // get the handle for selected matrix
    auto* handle = mtx_handles[selected_matrix];

    // Prepare input vectors for all channels using the prepared method
    auto input_vectors_x = handle->prepareInputVector(x_vec);
    auto input_vectors_bias = handle->prepareInputVector(bias_vec);

    // Set the input buffers for all channels for x (num_ch_B) and bias (num_ch_C)
    // Moving input data (x)
    for (int i = 0; i < num_ch_B; i++) {
        uint32_t buffer_size_x = input_vectors_x[i].size() * sizeof(float);
        _setInputBuffer(input_vectors_x[i].data(), buffer_size_x, arg_num++);
    }

    // Moving bias data (num_ch_C)
    for (int i = 0; i < num_ch_C; i++) {
        uint32_t buffer_size_bias = input_vectors_bias[i].size() * sizeof(float);
        _setInputBuffer(input_vectors_bias[i].data(), buffer_size_bias, arg_num++);
    }

    // Prepare output vector y
    auto output_vectors_y = handle->allocateOutputVector();

    // Prepare output buffers for each channel (num_ch_C output buffers)
    std::vector<xrt::bo> output_buffers;

    // Allocate output buffers for each channel
    for (int i = 0; i < num_ch_C; i++) {
        uint32_t y_buffer_size = output_vectors_y[i].size() * sizeof(float);
        auto device_buffer_y = _setOutputBuffer(output_vectors_y[i].data(), y_buffer_size, arg_num++);
        output_buffers.push_back(device_buffer_y);
    }

    // Set alpha and beta scalars to kernel arguments
    run.set_arg(arg_num++, alpha);
    run.set_arg(arg_num++, beta);

    // Start the kernel execution
    run.start();
    run.wait();

    // Sync the output buffers from device to host
    for (int i = 0; i < num_ch_C; i++) {
        output_buffers[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // This assumes that `output_vectors_y` holds the results in a channel-wise format.
    auto y = y_arr.request();
    auto* y_ptr = static_cast<float*>(y.ptr);

    for (int i = 0; i < y.shape[0]; i++) {
        int ch = (i / fp32_per_ch) % num_ch_C;
        int addr = (i / (fp32_per_ch * num_ch_C)) * fp32_per_ch + (i % fp32_per_ch);
        y_ptr[i] = output_vectors_y[ch][addr];  // Assign the value using calculated index
    }

    auto end_cpu = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
}

// double FpgaHelper::runKernel(
//     std::vector<aligned_vector<float>>& B, 
//     std::vector<aligned_vector<float>>& Cin,
//     std::vector<aligned_vector<float>>& Cout,
//     const float alpha, const float beta) 
// {
//     auto start_cpu = std::chrono::steady_clock::now();
//     int arg_num = num_ch_A;
    
//     // Move B data from host to fpga
//     for(int i = 0; i < num_ch_B; i++) {
//         uint32_t buffer_size = B[i].size() * sizeof(float);
//         float* host_ptr = B[i].data();

//         // Copy data from host memory to device buffer
//         _setInputBuffer(host_ptr, buffer_size, arg_num++);
//     }

//     // Move C data from host to fpga
//     for(int i = 0; i < num_ch_C; i++) {
//         uint32_t buffer_size = Cin[i].size() * sizeof(float);
//         float* host_ptr = Cin[i].data();

//         // Copy data from host memory to device buffer
//         _setInputBuffer(host_ptr, buffer_size, arg_num++);
//     }

//     // Add C out to args
//     std::vector<xrt::bo> output_buffers;
//     for(int i = 0; i < num_ch_C; i++) {
//         uint32_t buffer_size = Cout[i].size() * sizeof(float);
//         float* host_ptr = Cout[i].data();
//         output_buffers.push_back(_setOutputBuffer(host_ptr, buffer_size, arg_num++));
//     }

//     run.set_arg(arg_num++, alpha);
//     run.set_arg(arg_num++, beta);


//     run.start();
//     run.wait();

//     for(int i = 0; i < num_ch_C; i++) {
//         auto device_buffer = output_buffers[i];
//         device_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
//     }
//     auto end_cpu = std::chrono::steady_clock::now();
//     return std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
// }