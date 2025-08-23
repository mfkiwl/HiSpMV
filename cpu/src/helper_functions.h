#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cmath>

class PowerLogger {
public:
    PowerLogger() = default;  // no arguments

    void start();     // mark beginning of measurement
    void stop();      // mark end of measurement
    double getAveragePower() const; // average power in Watts

private:
    const std::string energyFile    = "/sys/class/powercap/intel-rapl:0/energy_uj";
    const std::string maxEnergyFile = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

    long long readEnergy(const std::string& path) const;

    long long energyStart = 0;
    long long energyEnd   = 0;
    std::chrono::steady_clock::time_point timeStart;
    std::chrono::steady_clock::time_point timeEnd;
};

struct CSRMatrix {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;
};

struct rcv{
    int r;
    int c;
    float v;
};

void readMatrixCSC(char * filename, std::vector<float>& values, std::vector<int>& rowIndices, std::vector<int>& colOffsets, 
    int& rows, int& cols, int& nnz);

void convertCSCtoCSR(const std::vector<float>& cscValues, const std::vector<int>& cscRowIndices, const std::vector<int>& cscColOffsets,
    std::vector<float>& csrValues, std::vector<int>& csrColIndices, std::vector<int>& csrRowOffsets, int rows, int cols, int nnz);

void printMatrixCSR(std::vector<float> values, std::vector<int> columns, std::vector<int> rowPtr, 
    int numRows, int numCols);

#endif