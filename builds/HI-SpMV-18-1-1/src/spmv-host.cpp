#include "spmv.h"
#include "spmv-helper.h"
#include <thread>
#include <atomic>
#include <regex>

using namespace std;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  uint16_t rp_time;
  const float alpha = 0.55;
  const float beta = -2.05;
  auto my_helper = SpMVHelper(NUM_A_CH, NUM_B_CH, NUM_C_CH, CH_WIDTH, URAMS_PER_PE, FP_ACC_LATENCY,
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

  if (argc == 3) { 
    char* filename = argv[1];
    rp_time = (uint16_t)stoi(argv[2]);
    cout << endl << "Preparing A Mtx..." << endl;
    double time_gen_ns = my_helper.prepareSparseMtxForFPGA(filename);
    cout << "Pre-processing Time: " << time_gen_ns * 1e-9 << " secs\n";
  }

  else if (argc == 4) {
    int rows = stoi(argv[1]);
    int cols = stoi(argv[2]);
    rp_time = (uint16_t)stoi(argv[3]);
    cout << endl << "Preparing A Mtx..." << endl;
    double time_gen_ns = my_helper.prepareDenseMtxForFPGA(rows, cols);
    cout << "Pre-processing Time: " << time_gen_ns * 1e-9 << " secs\n";
  }

  else {
    cerr << "Sparse Mode Usage: " << argv[0] << "<sparse mtx> <rp_time>" << endl;
    cerr << "Dense Mode Usage: " << argv[0] << "<dense number_of_rows> <dense number_of_cols> <rp_time>" << endl;
    return 1;
  }

  auto [rows, cols] = my_helper.getPaddedMatrixDims();
  int nnz = my_helper.getNNZ();

  vector<float>cpuBinVect = move(my_helper.generateVector(cols));
  vector<float>cpuCinVect = move(my_helper.generateVector(rows));
  vector<float>cpuCoutVect(rows, 0.0f);

  cout << endl << "Computing on CPU... "  << endl;

  double time_cpu_ns = my_helper.cpuSequential(cpuBinVect, cpuCinVect, alpha, beta, cpuCoutVect);
  cout << "CPU TIME: "   << time_cpu_ns * 1e-6 << " ms\n";
  cout << "CPU GFLOPS: " << 2.0 * (nnz + rows) / time_cpu_ns << "\n";

  vector<aligned_vector<float>>fpgaBinVect(NUM_B_CH, aligned_vector<float>(cols/NUM_B_CH, 0));
  vector<aligned_vector<float>>fpgaCinVect(NUM_C_CH, aligned_vector<float>(rows/NUM_C_CH, 0));
  vector<aligned_vector<float>>fpgaCoutVect(NUM_C_CH, aligned_vector<float>(rows/NUM_C_CH, 0));

  cout << "Preparing FPGA C Vec..."  << endl;
  for (int i = 0; i < rows; i++) {
    int ch = (i / FP32_PER_CH) % NUM_C_CH;
    int addr = (i / (FP32_PER_CH * NUM_C_CH)) * FP32_PER_CH + (i % FP32_PER_CH);
    fpgaCinVect[ch][addr] = cpuCinVect[i];
  }

  cout << endl << "Preparing FPGA B Vec..."  << endl;
  for (int i = 0; i < cols; i++) {
    int ch = (i / FP32_PER_CH) % NUM_B_CH;
    int addr = (i / (NUM_B_CH * FP32_PER_CH)) * FP32_PER_CH + (i % FP32_PER_CH); 
    fpgaBinVect[ch][addr] = cpuBinVect[i];
  }

  cout << endl << "Computing on FPGA..."  << endl;

  auto fpgaAinMtx = my_helper.getPreparedMtx();
  double time_fpga_ns = tapa::invoke(
    SpMV, FLAGS_bitstream, 
    tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAinMtx).reinterpret<channelA_t>(),
    tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinVect).reinterpret<channelB_t>(),
    tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinVect).reinterpret<channelC_t>(),
    tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutVect).reinterpret<channelC_t>(),
    alpha, beta, 
    (uint32_t) 0, (uint32_t)my_helper.getRunLength(),
    (uint32_t)my_helper.getRowsPerPE(), (uint32_t)my_helper.getVectLength(),
    (uint16_t)my_helper.getRowTiles(), (uint16_t)my_helper.getColTiles(),
    (uint32_t)my_helper.getTotTiles() * rp_time, (uint16_t)rp_time,
    my_helper.isDense());
  time_fpga_ns /= rp_time;

  cout << "FPGA Time: " << time_fpga_ns * 1e-3 << "us \n";
  cout << "FPGA GFLOPS: " << 2.0 * (nnz + rows) / time_fpga_ns << "\n";

  cout <<  endl << "Comparing Results... "  << endl;
  my_helper.printErrorStats(cpuCoutVect, fpgaCoutVect);
  return 0;
}