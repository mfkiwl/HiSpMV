#include <thread>
#include <atomic>
#include <regex>

#include "spmv.h"
#include "spmv-helper.h"

using namespace std;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

std::vector<float> generateVector(int size) {
    std::vector<float> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = 1.0 * (i + 2) / (i + 1);  // Vector values generated from loop indices
    }
    return move(vec);
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  uint16_t rp_time;
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
  hamdle.displayConfig();

  if (argc == 3) { 
    char* filename = argv[1];
    rp_time = (uint16_t)stoi(argv[2]);
    cout << endl << "Preparing A Mtx..." << endl;
    double time_gen_ns = handle.prepareSparseMtxForFPGA(filename);
    cout << "Pre-processing Time: " << time_gen_ns * 1e-9 << " secs\n";
  }

  else if (argc == 4) {
    int rows = stoi(argv[1]);
    int cols = stoi(argv[2]);
    rp_time = (uint16_t)stoi(argv[3]);
    cout << endl << "Preparing A Mtx..." << endl;
    double time_gen_ns = handle.prepareDenseMtxForFPGA(rows, cols, generateVector(rows*cols));
    cout << "Pre-processing Time: " << time_gen_ns * 1e-9 << " secs\n";
  }

  else {
    cerr << "Sparse Mode Usage: " << argv[0] << "<sparse mtx> <rp_time>" << endl;
    cerr << "Dense Mode Usage: " << argv[0] << "<dense number_of_rows> <dense number_of_cols> <rp_time>" << endl;
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
  auto fpgaCinVect = handle.prepareBiasVector(cpuCinVect);

  cout << endl << "Preparing FPGA B Vec..."  << endl;
  auto fpgaBinVect = handle.prepareInputVector(cpuBinVect);


  cout << endl << "Computing on FPGA..."  << endl;
  auto fpgaAinMtx = handle.getPreparedMtx();
  auto fpgaCoutVect = handle.allocateOutputVector();
  double time_fpga_ns = tapa::invoke(
    SpMV, FLAGS_bitstream, 
    tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAinMtx).reinterpret<channelA_t>(),
    tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinVect).reinterpret<channelB_t>(),
    tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinVect).reinterpret<channelC_t>(),
    tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutVect).reinterpret<channelC_t>(),
    alpha, beta, 
    0, handle.getRunLength(),
    handle.getRowsPerPE(), handle.getVectLength(),
    handle.getRowTiles(), handle.getColTiles(),
    handle.getTotTiles() * rp_time, rp_time,
    handle.isDense());
  time_fpga_ns /= rp_time;

  cout << "FPGA Time: " << time_fpga_ns * 1e-3 << "us \n";
  cout << "FPGA GFLOPS: " << 2.0 * (nnz + rows) / time_fpga_ns << "\n";
  

  cout <<  endl << "Comparing Results... "  << endl;
  handle.printErrorStats(cpuCoutVect, handle.collectOutputVector(fpgaCoutVect));
  return 0;
}