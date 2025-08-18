#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <vector>
#include <sstream>
#include <cmath>
#include <cassert>

struct CSRMatrix_t {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;
};

// Structure to hold COO matrix data
struct COOMatrix_t {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
    int rows_count;
    int cols_count;
    int nnz;
};


COOMatrix_t loadMtx(const std::string& mtx_file);
CSRMatrix_t cooToCsr(const COOMatrix_t& coo);
void printErrorStats(const std::vector<float>& cpu_ref, const std::vector<float>& gpu_out);
void generateSequentialVector(std::vector<float>& vec);
void cpuSpMV(int rows, int nnz, const std::vector<int>& rowInd, const std::vector<int>& colInd, 
             const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y, 
             float alpha, float beta);