#include <thread>
#include <atomic>
#include <regex>

#include "spmv.h"
#include "spmv-helper.h"

#define FREQ 225000000U // Assume frequency is 225MHz

using namespace std;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");
DEFINE_string(device, "", "xilinx fpga device id");
DEFINE_string(exec_ms, "1", "specify execution time for GFLOPs benchmarking in ms");
DEFINE_string(power_s, "0", "specify execution time for Power in Seconds");

vector<float> generateVector(int size) {
    vector<float> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = 1.0 * (i + 2) / (i + 1);  // Vector values generated from loop indices
    }
    return vec;
}

bool isXclbin(const string& filename) {
    if (filename.empty())
      return false;

    // Find the last dot in the filename
    size_t dotPosition = filename.rfind('.');
    if (dotPosition == string::npos) {
      // No dot found, so no extension
      return false;
    }

    // Extract the extension and compare it
    string fileExtension = filename.substr(dotPosition + 1);
    return fileExtension == "xclbin";
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  const float alpha = 0.55;
  const float beta = -2.05;
  auto handle = HiSpmvHandle(NUM_A_CH, NUM_B_CH, NUM_C_CH, CH_WIDTH, URAMS_PER_PE, FP_ACC_LATENCY,
    #ifdef BUILD_DENSE_OVERLAY 
    true,
    #else
    false,
    #endif
    #ifdef BUILD_PRE_ACCUMULATOR 
    true,
    #else
    false,
    #endif
    #ifdef BUILD_ROW_DIST_NETWORK 
    true
    #else
    false
    #endif
  );
  handle.displayConfig();

  if (argc == 2) { 
    char* filename = argv[1];
    cout << endl << "Preparing A Mtx..." << endl;
    double time_gen_ns = handle.prepareSparseMtxForFPGA(filename);
    cout << "Pre-processing Time: " << time_gen_ns * 1e-9 << " secs\n";
  }

#ifdef BUILD_DENSE_OVERLAY
  else if (argc == 3) {
    int rows = stoi(argv[1]);
    int cols = stoi(argv[2]);
    cout << endl << "Preparing A Mtx..." << endl;
    double time_gen_ns = handle.prepareDenseMtxForFPGA(rows, cols, generateVector(rows*cols));
    cout << "Pre-processing Time: " << time_gen_ns * 1e-9 << " secs\n";
  }
#endif

  else {
    cerr << "Sparse Mode Usage: " << argv[0] << "<sparse mtx> <rp_time>" << endl;
#ifdef BUILD_DENSE_OVERLAY
    cerr << "Dense Mode Usage: " << argv[0] << "<dense number_of_rows> <dense number_of_cols> <rp_time>" << endl;
#endif
    return 1;
  }

  auto [rows, cols] = handle.getMatrixDims();
  int nnz = handle.getNNZ();

  vector<float>cpuBinVect = generateVector(cols);
  vector<float>cpuCinVect = generateVector(rows);
  vector<float>cpuCoutVect(rows, 0.0f);

  cout << endl << "Computing on CPU... "  << endl;

  double time_cpu_ns = handle.cpuSequential(cpuBinVect, cpuCinVect, alpha, beta, cpuCoutVect);
  cout << "CPU TIME: "   << time_cpu_ns * 1e-6 << " ms\n";
  cout << "CPU GFLOPS: " << 2.0 * (nnz + rows) / time_cpu_ns << "\n";

  cout << "Preparing FPGA C Vec..."  << endl;
  auto fpgaCinVect = handle.prepareBiasVector(cpuCinVect.data());

  cout << endl << "Preparing FPGA B Vec..."  << endl;
  auto fpgaBinVect = handle.prepareInputVector(cpuBinVect.data());

  auto fpgaAinMtx = handle.getPreparedMtx();
  auto fpgaCoutVect = handle.allocateOutputVector();

  cout << "Matrix A Length: " << handle.getRunLength() << endl;
  cout << "Approx. Clock Cycles: " << handle.getTotalCycles() << endl;

  // Use XRT to tun on FPGA if using xclbin for hw and hw_emu
  if (isXclbin(FLAGS_bitstream)) {

    if (FLAGS_device.empty()) {
      cerr << "Please specify --device = <device id> when running xclbin" << endl;
      return 1;
    }

    int device_id = stoi(FLAGS_device);
    if (device_id < 0) {
      cerr << "Invlid Device ID: " << device_id << endl;
      return 1;
    }

    int exec_ms = stoi(FLAGS_exec_ms);
    if (exec_ms <= 0) {
      cerr << "Invlid exec time: " << exec_ms << endl;
      return 1;
    }

    int power_s = stoi(FLAGS_power_s);
    if (power_s < 0) {
      cerr << "Invlid power time: " << power_s << endl;
      return 1;
    }

    // Set rp_time to run to appropriate value to make it matche exec_ms
    float rp_time = (FREQ / static_cast<float>(handle.getTotalCycles()) / 1000) * exec_ms;
    uint16_t max_rp_time = 1 << 15;

    //If rp_time is too big to fit in uint16_t call the kernel multiple times
    uint16_t safe_rp_time = (rp_time > max_rp_time) ? max_rp_time : static_cast<uint16_t>(rp_time);
    auto num_samples = (static_cast<uint16_t>(rp_time) != safe_rp_time) ? (rp_time / safe_rp_time) : 1.0f; 

    // compute how many execs need to be done run the kernel for power_s secs
    if (power_s) num_samples *= (power_s * 1000.0 / exec_ms);
    num_samples = static_cast<int>(num_samples);
    cout << "Using Repeat Time: " << safe_rp_time << endl;
    cout << "Using Num samples: " << num_samples << endl;

    cout << endl << "Computing on FPGA..."  << endl;
    
    double time_fpga_ns = handle.fpgaRun(FLAGS_bitstream, device_id, fpgaBinVect, fpgaCinVect, alpha, beta, safe_rp_time, fpgaCoutVect, num_samples, power_s);
    
    cout << "Total Kernel Runtime: " << time_fpga_ns * 1e-6 << "ms \n";
    time_fpga_ns /= safe_rp_time;
    cout << "FPGA TIME: " << time_fpga_ns * 1e-3 << "us \n";
    cout << "FPGA GFLOPS: " << 2.0 * (nnz + rows) / time_fpga_ns << "\n";
  }

  // Use TAPA invoke for csim and fast-cosim
  else {
    tapa::invoke(
      SpMV, FLAGS_bitstream, 
      tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAinMtx).reinterpret<channelA_t>(),
      tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinVect).reinterpret<channelB_t>(),
      tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinVect).reinterpret<channelC_t>(),
      tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutVect).reinterpret<channelC_t>(),
      alpha, beta, 
      0, handle.getRunLength(),
      handle.getRowsPerPE(), handle.getInputLength(),
      handle.getRowTiles(), handle.getColTiles(),
      handle.getTotTiles(), 1,
      handle.isDense());
  }

  cout <<  endl << "Comparing Results... "  << endl;
  handle.printErrorStats(cpuCoutVect, handle.collectOutputVector(fpgaCoutVect));
  return 0;
}