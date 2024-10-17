#include "helper_functions.h"
#include "spmv.h"
#include <thread>
#include <atomic>
#include <regex>

using namespace std;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

const char* GetFilename(const char* filePath) {
    const char* lastSlash = strrchr(filePath, '/');
    const char* lastBackslash = strrchr(filePath, '\\');

    const char* lastSeparator = std::max(lastSlash, lastBackslash);
    
    if (lastSeparator) {
        // Move the pointer to the character right after the separator
        lastSeparator++;
        
        // Find the last "." after the separator
        const char* lastDot = strrchr(lastSeparator, '.');
        if (lastDot && lastDot > lastSeparator) {
            // Calculate the length of the substring
            size_t length = lastDot - lastSeparator;
            
            // Allocate memory for the substring and copy it
            char* baseFilename = new char[length + 1];
            strncpy(baseFilename, lastSeparator, length);
            baseFilename[length] = '\0';
            
            char* newFileName = new char[256];
            sprintf(newFileName, "./power_logs/%s.log", baseFilename);
            std::cout << newFileName << std::endl;
            return newFileName;
        }
    }
    
    // No separator found, return the whole path
    return filePath;
}

std::string exec(const char* cmd) {
    char buffer[2048];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return "ERROR";
    }
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr) {
            result += buffer;
        }
    }
    pclose(pipe);
    // printf("ran cmd\n");
    return result;
}

std::atomic<bool> keepCollecting(true);

void dataCollectionThread(const char* outputFilePath) {
    const char* command = "xbutil query -d 0000:d8:00.1";

    std::ofstream outputFile(outputFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    std::regex cardPowerRegex("Card Power\\(W\\)\\s+(\\d+)");

    while (keepCollecting) {
        for (int i = 0; i < 3; ++i) {
            std::string output = exec(command);
            
            std::smatch match;
            if (std::regex_search(output, match, cardPowerRegex)) {
                std::string cardPower = match[1];
                outputFile << cardPower << std::endl;
            }
        }
    }

    outputFile.close();
}


void stopDataCollection() {
    keepCollecting = false;
}

// Function to perform GeMV: y_out = alpha * A * x + beta * y_in
void gemv(const vector<vector<float>>& A, const vector<float>& x, const vector<float>& y_in, vector<float>& y_out, float alpha, float beta) {
    int rows = A.size();
    int cols = A[0].size();

    // Ensure matrix A dimensions match the vector sizes
    if (x.size() != cols || y_in.size() != rows || y_out.size() != rows) {
        cerr << "Dimension mismatch" << endl;
        return;
    }

    // Perform the GeMV operation
    for (int i = 0; i < rows; ++i) {
        float dot_product = 0.0f;
        for (int j = 0; j < cols; ++j) {
            dot_product += A[i][j] * x[j];
        }
        y_out[i] = alpha * dot_product + beta * y_in[i];
    }
}

void cpu_spmv(const CSRMatrix& A, const vector<float> B, vector<float>& Cin, const float alpha, const float beta, vector<float>& Cout) {  // Initialize result vector C with zeros

    // Perform matrix-vector multiplication
    for (int i = 0; i < A.row_offsets.size() - 1; ++i) {
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
            int colIndex = A.col_indices[j];
            float value = A.values[j];
            Cout[i] += value * B[colIndex];
        }
        Cout[i] = (alpha*Cout[i]) + (beta*Cin[i]);
    }
}

double computePrecisionLoss(const vector<float>& vectorA, const vector<aligned_vector<float>>& vectorB) 
{
  if (vectorA.size() != vectorB[0].size()*NUM_C_CH) {
      cout << "Error: Vector sizes do not match!" << endl;
      return -1;
  }

  double diffSum = 0.0;
  double refSum = 0.0;
  double maxRelativeError = 0.0;
  int maxRelativeErrorIdx = 0;

  for (size_t i = 0; i < vectorA.size(); ++i) {
    int ch = (i / FP32_PER_CH) % NUM_C_CH;
    int addr = (i / (FP32_PER_CH * NUM_C_CH)) * FP32_PER_CH + (i % FP32_PER_CH);
    double diff = fabs(vectorA[i] - vectorB[ch][addr]);
    double ref = min(fabs(vectorA[i]), (float)fabs(vectorB[ch][addr]));
    double relativeError = 0.0;

    if ((vectorA[i] != 0.0 ) && (vectorB[ch][addr] != 0.0)) {
        relativeError = (diff / ref);
        if (relativeError > maxRelativeError) {
          maxRelativeErrorIdx = i;
          maxRelativeError = relativeError;
        }
    }

    if (relativeError > 0.01)
      clog << "Relative Error: " << relativeError <<  " CPU: " << vectorA[i] << " FPGA: " << vectorB[(i / FP32_PER_CH) % NUM_C_CH][(i/ (FP32_PER_CH* NUM_C_CH)) * FP32_PER_CH + (i % FP32_PER_CH)] << "\t i: " << i << endl;

    diffSum += diff;
    refSum += ref;   
  }

  clog << "Max Relative Error: " << maxRelativeError <<  " CPU: " << vectorA[maxRelativeErrorIdx] << " FPGA: " << vectorB[(maxRelativeErrorIdx / FP32_PER_CH) % NUM_C_CH][(maxRelativeErrorIdx/ (FP32_PER_CH * NUM_C_CH)) * FP32_PER_CH + (maxRelativeErrorIdx % FP32_PER_CH)] << "\t i: " << maxRelativeErrorIdx << endl;

  return diffSum/refSum;
}

  // Function to generate a matrix based on loop iterator values
vector<vector<float>> generate_matrix(int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = (float)(i + j + 1) / max(rows, cols);  // Matrix values generated from loop indices
        }
    }
    return matrix;
}

// Function to generate a vector based on loop iterator values
vector<float> generate_vector(int size) {
    vector<float> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = 1.0 * (i+1) / (i+2);;  // Vector values generated from loop indices
    }
    return vec;
}

void compute_max_errors(const vector<float>& expected, const vector<float>& actual) {
    if (expected.size() != actual.size()) {
        cerr << "Vectors must have the same size" << endl;
        return;
    }

    float max_abs_error = 0.0f;
    float max_abs_reference = 0.0f;
    float max_rel_error = 0.0f;

    for (size_t i = 0; i < expected.size(); ++i) {
        float abs_error = fabs(actual[i] - expected[i]);
        float rel_error = 0.0f;

        if (fabs(expected[i]) > 1e-6) {  // Avoid division by zero for relative error
            rel_error = abs_error / fabs(expected[i]);
        }

        // Update max absolute and relative errors
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_abs_reference = expected[i];
        }
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }
    }

        // Output the results
    cout << "Maximum Absolute Error: " << max_abs_error <<  " \t Expected: " << max_abs_reference << endl;
    cout << "Maximum Relative Error: " << max_rel_error << endl;
    
}

void test_sparse_mode(char* filename, uint16_t rp_time) {

  cout << endl << "=======================================================" << endl;
  cout << "===================  SPARSE MODE  =====================" << endl;
  cout << "=======================================================" << endl << endl;

  clog << "Reading A" << filename << endl;

  vector<float> cscValues;
  vector<int> cscRowIndices;
  vector<int> cscColOffsets;
  int rows, cols, nnz;
  readMatrixCSC(filename, cscValues, cscRowIndices, cscColOffsets, rows, cols, nnz);

  CSRMatrix cpuAmtx;
  convertCSCtoCSR(cscValues, cscRowIndices, cscColOffsets, cpuAmtx.values, cpuAmtx.col_indices, cpuAmtx.row_offsets, rows, cols, nnz);

  int Window = B_WINDOW;
  int Depth = D;

  int numTilesCols = (cols - 1)/Window + 1;
  int numTilesRows = (rows - 1)/Depth + 1;

  cout << "Rows: " << rows << "\t Cols: " << cols  << "\t NNZ: " << nnz << endl;
  cout << "Window: " << Window << "\t Depth: " << Depth << endl;
  cout << "Numtiles: " << numTilesRows << ", " << numTilesCols << endl << endl; 

  uint32_t numRowsPerPE = (rows - 1) / NUM_PES + 1;
  uint32_t B_len = (cols - 1) / (NUM_B_CH * FP32_PER_CH) + 1;
  uint32_t numTiles = (numTilesRows * numTilesCols);
  
  vector<float>cpuBinVect = generate_vector(B_len * (NUM_B_CH * FP32_PER_CH));
  vector<float>cpuCinVect = generate_vector(numRowsPerPE * NUM_PES);
  vector<float>cpuCoutVect(numRowsPerPE * NUM_PES, 0);

  vector<aligned_vector<float>>fpgaBinVect(NUM_B_CH, aligned_vector<float>(B_len * FP32_PER_CH, 0));
  vector<aligned_vector<float>>fpgaCinVect(NUM_C_CH, aligned_vector<float>(numRowsPerPE * NUM_PES/NUM_C_CH, 0));
  vector<aligned_vector<float>>fpgaCoutVect(NUM_C_CH, aligned_vector<float>(numRowsPerPE * NUM_PES/NUM_C_CH, 0));

  clog << "Preparing FPGA C Vec..."  << endl;
  for (int i = 0; i < numRowsPerPE * NUM_PES; i++) {
    int ch = (i / FP32_PER_CH) % NUM_C_CH;
    int addr = (i / (FP32_PER_CH * NUM_C_CH)) * FP32_PER_CH + (i % FP32_PER_CH);
    fpgaCinVect[ch][addr] = cpuCinVect[i];
  }

  cout << endl << "Preparing FPGA B Vec..."  << endl;
  for (int i = 0; i < (B_len * NUM_B_CH * FP32_PER_CH); i++) {
    int ch = (i / FP32_PER_CH) % NUM_B_CH;
    int addr = (i / (NUM_B_CH * FP32_PER_CH)) * FP32_PER_CH + (i % FP32_PER_CH); 
    fpgaBinVect[ch][addr] = cpuBinVect[i];
  }

  const float alpha = 0.55;
  const float beta = -2.05;
  
  cout << endl << "Preparing A Mtx..." << endl;

  auto start_gen = std::chrono::steady_clock::now();
  vector<vector<CSRMatrix>> tiledMatrices = tileCSRMatrix(cpuAmtx, rows, cols, Depth, Window, numTilesRows, numTilesCols);

  vector<aligned_vector<uint64_t>> fpgaAinMtx = prepareAmtx(tiledMatrices, numTilesRows, numTilesCols, Depth, Window, rows, cols, nnz);

  auto end_gen = std::chrono::steady_clock::now();
  double time_gen = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen - start_gen).count();
  time_gen *= 1e-9;
  cout << "Pre-processing Time: " << time_gen*1000 << " msec\n";

  cout << "Rows Per PE: " << numRowsPerPE << "\t B length: " << B_len << endl;
  cout << "NNZ: " << nnz << endl;
  cout << "Numtiles: " << numTiles << endl << endl; 



  cout <<  endl << "Computing CPU SpMV... "  << endl;
  auto start_cpu = std::chrono::steady_clock::now();
  cpu_spmv(cpuAmtx, cpuBinVect, cpuCinVect, alpha, beta, cpuCoutVect);
  // gemv(A, cpuBinVect, cpuCinVect, cpuCoutVect, alpha, beta);
  auto end_cpu = std::chrono::steady_clock::now();
  double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
  time_cpu *= 1e-9;
  cout << "done (" << time_cpu*1000 << " msec)\n";
  cout <<"CPU GFLOPS: " << 2.0 * (nnz + rows) / 1e+9 / time_cpu << "\n";
 
 
  uint32_t A_len = fpgaAinMtx[0].size()/ PES_PER_CH;
  cout <<  endl << "Computing FPGA SpMV... "  << endl;

  // const char* outputFilePath = GetFilename(filename);
  // // std::thread collectionThread(dataCollectionThread, outputFilePath);
  double time_taken = tapa::invoke(
    SpMV, FLAGS_bitstream, 
    tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAinMtx).reinterpret<channelA_t>(),
    tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinVect).reinterpret<channelB_t>(),
    tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinVect).reinterpret<channelC_t>(),
    tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutVect).reinterpret<channelC_t>(),
    alpha, beta, 
    (uint32_t) 0, (uint32_t)A_len,
    (uint32_t)numRowsPerPE, (uint32_t)B_len,
    (uint16_t)numTilesRows, (uint16_t)numTilesCols,
    (uint32_t)(numTiles*rp_time), (uint16_t)rp_time,
    false);
  // clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;
  // stopDataCollection();
  // collectionThread.join();
  time_taken *= (1e-9); // total time in second
  time_taken /= rp_time;
    printf("Kernel time:%f\n", time_taken*1000);
  float gflops =
    2.0 * (nnz + rows)
    / 1e+9
    / time_taken
    ;
  printf("GFLOPS:%f \n", gflops);
 
  cout <<  endl << "Comparing Results... "  << endl;
  double precisionLoss = computePrecisionLoss(cpuCoutVect, fpgaCoutVect);
  cout << "Precision Loss: " << precisionLoss << endl;
} 

void test_dense_mode(int rows, int cols, uint16_t rp_time) {
  cout << "=======================================================" << endl;
  cout << "====================  DENSE MODE  =====================" << endl;
  cout << "=======================================================" << endl;
  int Window = B_WINDOW;
  int Depth = D;

  int numTilesCols = (cols - 1)/Window + 1;
  int numTilesRows = (rows - 1)/Depth + 1;

  cout << "Rows: " << rows << "\t Cols: " << cols << endl ; //<< "\t NNZ: " << nnz << endl;
  cout << "Window: " << Window << "\t Depth: " << Depth << endl;
  cout << "Numtiles: " << numTilesRows << ", " << numTilesCols << endl << endl; 
  cout << "Preparing A Mtx... " << endl; 

  uint32_t numRowsPerPE = (rows - 1) / NUM_PES + 1;
  uint32_t B_len = (cols - 1) / (NUM_B_CH * FP32_PER_CH) + 1;
  uint32_t numTiles = (numTilesRows * numTilesCols);
  
  vector<vector<float>> A = generate_matrix(numRowsPerPE * NUM_PES, B_len * (NUM_B_CH * FP32_PER_CH));
  vector<float>cpuBinVect = generate_vector(B_len * (NUM_B_CH * FP32_PER_CH));
  vector<float>cpuCinVect = generate_vector(numRowsPerPE * NUM_PES);
  vector<float>cpuCoutVect(numRowsPerPE * NUM_PES, 0);

  vector<aligned_vector<float>>fpgaBinVect(NUM_B_CH, aligned_vector<float>(B_len * FP32_PER_CH, 0));
  vector<aligned_vector<float>>fpgaCinVect(NUM_C_CH, aligned_vector<float>(numRowsPerPE * NUM_PES/NUM_C_CH, 0));
  vector<aligned_vector<float>>fpgaCoutVect(NUM_C_CH, aligned_vector<float>(numRowsPerPE * NUM_PES/NUM_C_CH, 0));

  clog << "Preparing FPGA C Vec..."  << endl;
  for (int i = 0; i < numRowsPerPE * NUM_PES; i++) {
    int ch = (i / FP32_PER_CH) % NUM_C_CH;
    int addr = (i / (FP32_PER_CH * NUM_C_CH)) * FP32_PER_CH + (i % FP32_PER_CH);
    fpgaCinVect[ch][addr] = cpuCinVect[i];
  }

  cout << endl << "Preparing FPGA B Vec..."  << endl;
  for (int i = 0; i < (B_len * NUM_B_CH * FP32_PER_CH); i++) {
    int ch = (i / FP32_PER_CH) % NUM_B_CH;
    int addr = (i / (NUM_B_CH * FP32_PER_CH)) * FP32_PER_CH + (i % FP32_PER_CH); 
    fpgaBinVect[ch][addr] = cpuBinVect[i];
  }

  const float alpha = 0.85;
  const float beta = -2.06;
  
  cout << endl << "Preparing A Mtx..." << endl;
  uint32_t A_len = numRowsPerPE * (B_len * NUM_B_CH * FP32_PER_CH) / 2;
  vector<aligned_vector<uint64_t>>fpgaAinMtx(NUM_A_CH, aligned_vector<uint64_t>(A_len * PES_PER_CH, 0));
  
  cout << "Run Length:" << A_len << endl;

  for (int i = 0; i < numRowsPerPE * NUM_PES; i++) {
    for (int j = 0; j < numTilesCols; j++) {
      int offset = j * (numRowsPerPE) * (B_WINDOW / 2);
      for (int jj = 0; (jj < B_WINDOW) && ((j * B_WINDOW + jj) <  B_len * (NUM_B_CH * FP32_PER_CH)); jj+=2) {
        int pe = (i % NUM_PES);
        int ch = (pe / PES_PER_CH);
        int inter_ch_pe = (pe % PES_PER_CH);
        int addr = offset + (jj/2) * numRowsPerPE + (i / NUM_PES);
        uint64_t temp = 0;
        uint32_t val0 = *(uint32_t*)&(A[i][j * B_WINDOW + jj]);
        uint32_t val1 = *(uint32_t*)&(A[i][j * B_WINDOW + jj + 1]);
        temp |= val1;
        temp <<= 32;
        temp |= val0;
        fpgaAinMtx[ch][addr * PES_PER_CH + inter_ch_pe] = temp;
      }
    }
  }

  cout <<  endl << "Computing CPU GeMV... "  << endl;
  auto start_cpu = std::chrono::steady_clock::now();
  gemv(A, cpuBinVect, cpuCinVect, cpuCoutVect, alpha, beta);
  auto end_cpu = std::chrono::steady_clock::now();
  double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
  time_cpu *= 1e-9;
  cout << "done (" << time_cpu*1000 << " msec)\n";
  cout <<"CPU GFLOPS: " << 2.0 * (rows * (cols + 1)) / 1e+9 / time_cpu << "\n";

  cout <<  endl << "Computing FPGA SpMV... "  << endl;
  // const char* outputFilePath = GetFilename(filename);
  // // std::thread collectionThread(dataCollectionThread, outputFilePath);
  double time_taken = tapa::invoke(
    SpMV, FLAGS_bitstream, 
    tapa::read_only_mmaps<uint64_t, NUM_A_CH>(fpgaAinMtx).reinterpret<channelA_t>(),
    tapa::read_only_mmaps<float, NUM_B_CH>(fpgaBinVect).reinterpret<channelB_t>(),
    tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinVect).reinterpret<channelC_t>(),
    tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutVect).reinterpret<channelC_t>(),
    alpha, beta, 
    (uint32_t) 0, (uint32_t)A_len,
    (uint32_t)numRowsPerPE, (uint32_t)B_len,
    (uint16_t)numTilesRows, (uint16_t)numTilesCols,
    (uint32_t)(numTiles*rp_time), (uint16_t)rp_time,
    true);
  // clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;
  // stopDataCollection();
  // collectionThread.join();
  time_taken *= (1e-9); // total time in second
  time_taken /= rp_time;
    printf("Kernel time:%f\n", time_taken*1000);
  float gflops =
    2.0 * (rows * (cols + 1))
    / 1e+9
    / time_taken
    ;
  printf("GFLOPS:%f \n", gflops);
 
  cout <<  endl << "Comparing Results... "  << endl;
  double precisionLoss = computePrecisionLoss(cpuCoutVect, fpgaCoutVect);
  cout << "Precision Loss: " << precisionLoss << endl;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  if (argc < 2) {
    cerr << "Usage 1: " << argv[0] << "<sparse mtx> <rp_time>" << endl;
    cerr << "Usage 2: " << argv[0] << "<dense number_of_rows> <dense number_of_cols> <rp_time>" << endl;
    cerr << "Usage 3: " << argv[0] << "<sparse mtx> <dense number_of_rows> <dense number_of_cols> <rp_time>" << endl;
    return 1;
  }
  
  if (argc == 3) { 
    char* filename = argv[1];
    uint16_t rp_time = (uint16_t)stoi(argv[2]);
    test_sparse_mode(filename, rp_time);
  }

  else if (argc == 4) {
    int rows = stoi(argv[1]);
    int cols = stoi(argv[2]);
    uint16_t rp_time = (uint16_t)stoi(argv[3]);
    test_dense_mode(rows, cols, rp_time);
  }

  else if (argc == 5) {
    char* filename = argv[1];
    int rows = stoi(argv[2]);
    int cols = stoi(argv[3]);
    uint16_t rp_time = (uint16_t)stoi(argv[4]);
    test_sparse_mode(filename, rp_time);
    test_dense_mode(rows, cols, rp_time);
  }
  return 0;
}