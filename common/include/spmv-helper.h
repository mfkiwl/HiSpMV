#ifndef SPMV_HELPER_H
#define SPMV_HELPER_H

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <tapa.h>
#include <stdexcept>
#include <cctype>
#include <cassert>
#include <utility>
#include <cmath>
#include <limits>
#include <iomanip>

#define PADDING 1

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

struct CSRMatrix_t {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;
};

struct COOMatrix_t {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
};

inline uint64_t encode(bool tileEnd, bool rowEnd, bool sharedRow, uint16_t row, uint16_t col, uint32_t val)
{
    uint64_t res = 0;
    res |= rowEnd; 
    res <<= 15;
    res |= row & (0x7FFF);
    res <<= 1;
    res |= tileEnd; //will be 47th bit
    res <<= 1;
    res |= sharedRow; // will be 46th nit
    res <<= 14;
    res |= col & (0x3FFF); // 14 bits col
    res <<= 32;
    res |= val; //32 bits val
    return res;
}

class SpMVHelper {
private:
    // Matrix Market Constant parameters
    static constexpr const char* MatrixMarketBanner = "%%MatrixMarket";
    static constexpr const char* MM_MTX_STR         = "matrix";
    static constexpr const char* MM_SPARSE_STR      = "coordinate";
    static constexpr const char* MM_REAL_STR        = "real";
    static constexpr const char* MM_INT_STR         = "integer";
    static constexpr const char* MM_PATTERN_STR     = "pattern";
    static constexpr const char* MM_GENERAL_STR     = "general";
    static constexpr const char* MM_SYMM_STR        = "symmetric";
    static constexpr const char* MM_SKEW_STR        = "skew-symmetric";

    // Configuration parameters
    int num_ch_A;
    int num_ch_B;
    int num_ch_C;
    int ch_width;
    int urams_per_pe;
    int fp_acc_latency;
    bool dense_overlay;
    bool pre_accumulator;
    bool row_dist_net;

    // Derived config parameters
    int num_pes;
    int num_fp32s_b;
    int num_fp32s_c;
    int pes_per_ch;
    int fp32_per_ch;
    int max_depth;
    int max_window;

    // Matrix parameters 
    COOMatrix_t coo_mtx;
    std::vector<std::vector<float>> dense_mtx;
    int rows;
    int cols;
    int nnz;
    int rows_per_pe;
    int b_len;
    int padded_rows;
    int padded_cols;
    int tile_rows;
    int tile_cols;
    int row_tiles;
    int col_tiles; 
    std::vector<std::vector<CSRMatrix_t>> tiled_matrices;
    std::set<int>shared_row_indices;

    //Preprocessing Parameter
    bool use_row_dist = true;
    bool use_pre_accum = true;
    bool use_dense_mode = true;
    bool use_old_balancing_algorithm = false;
    std::vector<aligned_vector<uint64_t>> prep_mtx;
    
    // Method to display the configuration
    void displayConfig() const;

    // Method to load Sparse Matrix from Matrix Market file (.mtx)
    COOMatrix_t loadMtx(const std::string& mtx_file);

    // Method to tile and pad Sparse Matrix in COO format into CSR format
    std::vector<std::vector<CSRMatrix_t>> tileAndPad(COOMatrix_t mtx); 

    //Method to tile and pad Dense Matrix (overload)
    void tileAndPad();

    // Method to balance PE workloads and return shared rows
    std::vector<int> balanceWorkload(const CSRMatrix_t& csr_matrix);

    // Method to compute tile size of a tile
    int computeTileSize(const CSRMatrix_t& csr_matrix, const std::vector<int>& shared_rows);

    //Method to prepare CSR tile to FPGA format
    void prepareTile(const CSRMatrix_t& csr_matrix, const std::vector<int>& shared_rows, const int size, const int offset, std::vector<aligned_vector<uint64_t>>& fpga_mtx);

    //Method to check if the a row is processed in shared mode inany tile
    bool isSharedRow(int row_idx);

public:
    // Constructor to initialize the configuration
    SpMVHelper(int a, int b, int c, int width, int urams, int fp_acc_latency, bool dense, bool pre_acc, bool row_dist);
    
    // Method to prepare Sparse Matrix for FPGA
    double prepareSparseMtxForFPGA(const std::string& mtx_file);

    // Method to prepare Dense Matrix for FPGA
    double prepareDenseMtxForFPGA(const int rows, const int cols);

    //Method to generate a vector with values
    std::vector<float> generateVector(int size);

    // Method to get Matrix Properties
    COOMatrix_t getMatrix();
    std::pair<int, int> getMatrixDims();
    int getNNZ();
    std::pair<int, int> getPaddedMatrixDims();
    int getRowTiles();
    int getColTiles();
    int getTotTiles();
    int getRowsPerPE();
    int getVectLength();
    int getRunLength();
    std::vector<aligned_vector<uint64_t>> getPreparedMtx();
    bool isDense();

    double cpuSequential(const std::vector<float> B, std::vector<float>& Cin, const float alpha, const float beta, std::vector<float>& Cout);
    void printErrorStats(const std::vector<float>& cpu_ref, const std::vector<aligned_vector<float>>& fpga_out);
};

#endif