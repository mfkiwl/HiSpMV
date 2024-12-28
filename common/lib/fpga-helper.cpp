#include "fpga-helper.h"

FpgaHelper::FpgaHelper(const std::string& xclbin_path, const std::string& device_id, const int A_hbm_ch, const int B_hbm_ch, const int C_hbm_ch) {
    try {
        std::cout << "Opening the device with id: " << device_id << std::endl;
        int id = std::stoi(device_id);

        if (id < 0) {
            throw std::invalid_argument("Device ID must be a non-negative integer.");
        }

        device = xrt::device(id);
        std::cout << "Device name:     " << device.get_info<xrt::info::device::name>() << "\n";
        std::cout << "Device BDF:      " << device.get_info<xrt::info::device::bdf>() << "\n";
    }  catch (const std::runtime_error& e) {
        std::cerr << "XRT error while opening device: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Step 2: Load the xclbin
    try {
        std::cout << "Loading the xclbin: " << xclbin_path << std::endl;

        if (xclbin_path.empty()) {
            throw std::invalid_argument("XCLBIN path is empty.");
        }

        auto uuid = device.load_xclbin(xclbin_path);
        std::cout << "XCLBIN loaded successfully.\n";

        std::cout << "Initializing kernel: SpMV\n";
        krnl = xrt::kernel(device, uuid, "SpMV", xrt::kernel::cu_access_mode::exclusive);
        run = xrt::run(krnl);
        // run.stop();
        std::cout << "Kernel initialized successfully.\n";

    } catch (const std::runtime_error& e) {
        std::cerr << "XRT error while loading SpMV kernel from xclbin: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }
    num_ch_A = A_hbm_ch;
    num_ch_B = B_hbm_ch;
    num_ch_C = C_hbm_ch;
}

void FpgaHelper::_setInputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num) {
    // Allocate memory on the device memory
    xrt::bo device_buffer = xrt::bo(device, host_aligned_ptr, buffer_size, xrt::bo::flags::device_only, krnl.group_id(arg_num));

    // Sync buffer to device
    device_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Set the buffer to appropriate argument number
    run.set_arg(arg_num, device_buffer); 
}

xrt::bo FpgaHelper::_setOutputBuffer(void* host_aligned_ptr, uint32_t buffer_size, const int arg_num) {
    // Allocate memory on the device memory
    xrt::bo device_buffer = xrt::bo(device, host_aligned_ptr, buffer_size, krnl.group_id(arg_num));

    // Set the buffer to appropriate argument number
    run.set_arg(arg_num, device_buffer); 

    return device_buffer;
}

uint32_t FpgaHelper::loadMatricesToDevice(const std::vector<SpMVHelper*>& inputs) {
    matrices = inputs;
    std::vector<xrt::bo> input_buffers;
    for(int i = 0; i < num_ch_A; i++) {
        xrt::bo device_buffer = xrt::bo(device, (uint32_t)MAX_BUFFER_SIZE_BYTES, krnl.group_id(i));
        input_buffers.push_back(device_buffer);
    }

    //First matrix has offset of 0
    matrix_offsets.push_back(0);

    for(auto* mtx: matrices) {
        auto& prep_mtx = mtx->getPreparedMtx();

        if (prep_mtx.size() != num_ch_A) {
            std::cerr << "The specified number of hbm channels and prepared matrix channels doesn't match!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // All channels have same size
        uint32_t buffer_size = prep_mtx[0].size() * sizeof(uint64_t);
        uint32_t offset = matrix_offsets.back();
        uint32_t new_size = offset + buffer_size;

        // Make sure adding this matrix will not exceed max capacity
        if (new_size > MAX_BUFFER_SIZE_BYTES)
            break;

        // Fill in data from prepared matrix data for all channels
        for(int i = 0; i < num_ch_A; i++) {
            xrt::bo device_buffer = input_buffers[i];
            std::memcpy(device_buffer.map<char*>() + offset, prep_mtx[i].data(), buffer_size);
        }

        // Update matrix offsets
        matrix_offsets.push_back(new_size);
    }   
        
    // Sync Data transfer and set the buffers as kernel argument
    for(int i = 0; i < num_ch_A; i++) {
        xrt::bo device_buffer = input_buffers[i];
        device_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run.set_arg(i, device_buffer); 
    }
    return matrix_offsets.size() - 1;
}

void FpgaHelper::selectMatrix(const uint32_t matrix_idx) {
    if (matrix_idx > matrix_offsets.size() - 2) {
        std::cerr << "Matrix idx out of range" << std::endl;
        exit(EXIT_FAILURE);
    }
    SpMVHelper* mtx = matrices[matrix_idx];
    int arg_num = num_ch_A + num_ch_B + 2*num_ch_C + 2;
    run.set_arg(arg_num++, matrix_offsets[matrix_idx]);
    run.set_arg(arg_num++, mtx->getRunLength()); 
    run.set_arg(arg_num++, mtx->getRowsPerPE());
    run.set_arg(arg_num++, mtx->getVectLength());
    run.set_arg(arg_num++, mtx->getRowTiles());
    run.set_arg(arg_num++, mtx->getColTiles());
    run.set_arg(arg_num++, mtx->getTotTiles());
    run.set_arg(arg_num++, (uint32_t)1); // repeat time
    run.set_arg(arg_num++, mtx->isDense());
}

double FpgaHelper::runKernel(
    std::vector<aligned_vector<float>>& B, 
    std::vector<aligned_vector<float>>& Cin,
    std::vector<aligned_vector<float>>& Cout,
    const float alpha, const float beta) 
{
    auto start_cpu = std::chrono::steady_clock::now();
    int arg_num = num_ch_A;
    
    // Move B data from host to fpga
    for(int i = 0; i < num_ch_B; i++) {
        uint32_t buffer_size = B[i].size() * sizeof(float);
        float* host_ptr = B[i].data();

        // Copy data from host memory to device buffer
        _setInputBuffer(host_ptr, buffer_size, arg_num++);
    }

    // Move C data from host to fpga
    for(int i = 0; i < num_ch_C; i++) {
        uint32_t buffer_size = Cin[i].size() * sizeof(float);
        float* host_ptr = Cin[i].data();

        // Copy data from host memory to device buffer
        _setInputBuffer(host_ptr, buffer_size, arg_num++);
    }

    // Add C out to args
    std::vector<xrt::bo> output_buffers;
    for(int i = 0; i < num_ch_C; i++) {
        uint32_t buffer_size = Cout[i].size() * sizeof(float);
        float* host_ptr = Cout[i].data();
        output_buffers.push_back(_setOutputBuffer(host_ptr, buffer_size, arg_num++));
    }

    run.set_arg(arg_num++, alpha);
    run.set_arg(arg_num++, beta);


    run.start();
    run.wait();

    for(int i = 0; i < num_ch_C; i++) {
        auto device_buffer = output_buffers[i];
        device_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    auto end_cpu = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
}