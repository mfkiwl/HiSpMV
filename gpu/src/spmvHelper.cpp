#include "spmvHelper.h"

// Function to load COO Matrix from a .mtx file
COOMatrix_t loadMtx(const std::string& mtx_file) {
    // Constants for Matrix Market format validation
    const std::string MatrixMarketBanner = "%%MatrixMarket";
    const std::string MM_MTX_STR = "matrix";
    const std::string MM_SPARSE_STR = "coordinate";
    const std::string MM_REAL_STR = "real";
    const std::string MM_INT_STR = "integer";
    const std::string MM_PATTERN_STR = "pattern";
    const std::string MM_GENERAL_STR = "general";
    const std::string MM_SYMM_STR = "symmetric";
    const std::string MM_SKEW_STR = "skew-symmetric";

    COOMatrix_t mtx;
    std::ifstream file(mtx_file);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open file " + mtx_file);
    }

    // Read header line
    std::string line;
    std::getline(file, line);
    std::istringstream header_stream(line);
    std::string header[5];
    for (int i = 0; i < 5; ++i) {
        header_stream >> header[i];
    }

    // Validate Matrix Market format
    if (header[0] != MatrixMarketBanner || header[1] != MM_MTX_STR) {
        throw std::runtime_error("Error: Not a valid Matrix Market file.");
    }

    std::string matrix_format = header[2];  // "coordinate"
    std::string data_type = header[3];      // "real", "integer", "pattern"
    std::string symmetry = header[4];       // "general", "symmetric", "skew-symmetric"

    if (matrix_format != MM_SPARSE_STR) {
        throw std::runtime_error("Error: Only sparse matrices in 'coordinate' format are supported.");
    }

    if (data_type != MM_REAL_STR && data_type != MM_INT_STR && data_type != MM_PATTERN_STR) {
        throw std::runtime_error("Error: Unsupported data type.");
    }

    if (symmetry != MM_GENERAL_STR && symmetry != MM_SYMM_STR && symmetry != MM_SKEW_STR) {
        throw std::runtime_error("Error: Unsupported symmetry type.");
    }

    // Skip comments
    while (true) {
        std::getline(file, line);
        if (line[0] != '%') {
            break;
        }
    }

    // Parse matrix dimensions and non-zero count
    std::istringstream line_stream(line);
    line_stream >> mtx.rows_count >> mtx.cols_count >> mtx.nnz;

    // Resize the COO matrix data
    int max_entries = (symmetry == MM_GENERAL_STR) ? mtx.nnz : 2 * mtx.nnz;
    mtx.rows.reserve(max_entries);
    mtx.cols.reserve(max_entries);
    mtx.values.reserve(max_entries);

    int count = 0;
    for (std::getline(file, line); !file.eof(); std::getline(file, line)) {
        std::istringstream entry_stream(line);
        int r, c;
        float value;

        entry_stream >> r >> c >> value;
        --r;  // Convert to 0-based index
        --c;  // Convert to 0-based index

        if (data_type == MM_PATTERN_STR) {
            value = 1.0f;
        }

        if (value == 0) {
            continue;
        }

        // Add entry
        mtx.rows.push_back(r);
        mtx.cols.push_back(c);
        mtx.values.push_back(value);
        ++count;

        // Handle symmetry
        if (symmetry == MM_SYMM_STR && r != c) {
            mtx.rows.push_back(c);
            mtx.cols.push_back(r);
            mtx.values.push_back(value);
            ++count;
        } else if (symmetry == MM_SKEW_STR && r != c) {
            mtx.rows.push_back(c);
            mtx.cols.push_back(r);
            mtx.values.push_back(-value);
            ++count;
        }
    }

    mtx.nnz = count;

    std::cout << "\nMatrix Properties:" << std::endl;
    std::cout << "\tRows: " << mtx.rows_count << "\n\tCols: " << mtx.cols_count
              << "\n\tNNZ: " << count << "\n" << std::endl;

    return mtx;
}

CSRMatrix_t cooToCsr(const COOMatrix_t& coo) {
    CSRMatrix_t csr;
    
    // Resize row_offsets (one extra for the last row's offset)
    csr.row_offsets.resize(coo.rows_count + 1, 0);

    // Temporary storage to hold columns and values for each row
    std::vector<std::vector<std::pair<int, float>>> row_data(coo.rows_count);

    // Step 1: Group elements by row
    for (int i = 0; i < coo.nnz; ++i) {
        int row = coo.rows[i];
        row_data[row].emplace_back(coo.cols[i], coo.values[i]);
    }

    // Step 2: Count the non-zero elements per row for row_offsets
    for (int i = 0; i < coo.rows_count; ++i) {
        csr.row_offsets[i + 1] = csr.row_offsets[i] + row_data[i].size();
    }

    // Step 3: Sort each row's elements by column index
    for (int i = 0; i < coo.rows_count; ++i) {
        std::sort(row_data[i].begin(), row_data[i].end()); // Sort by column index
    }

    // Step 4: Fill the col_indices and values arrays
    csr.col_indices.resize(coo.nnz);
    csr.values.resize(coo.nnz);

    int idx = 0;
    for (int i = 0; i < coo.rows_count; ++i) {
        for (const auto& entry : row_data[i]) {
            csr.col_indices[idx] = entry.first;
            csr.values[idx] = entry.second;
            ++idx;
        }
    }

    return csr;
}


void printErrorStats(const std::vector<float>& cpu_ref, const std::vector<float>& gpu_out) 
{
    if (cpu_ref.size() != gpu_out.size()) {
        throw std::runtime_error("Error: Vector sizes do not match!");
    }

    // Compute Relative Errors
    std::vector<double> relative_errors;
    for (size_t i = 0; i < cpu_ref.size(); ++i) {
        double fpga = static_cast<double>(std::fabs(gpu_out[i]));
        double cpu = static_cast<double>(std::fabs(cpu_ref[i]));
        double epsilon = std::numeric_limits<double>::lowest();
        double ref = std::max(cpu, epsilon);
        double rel_err = std::fabs(fpga - ref) / ref;

        if (rel_err != 0) 
            relative_errors.push_back(rel_err);
    }
  
    if (relative_errors.empty()) {
        std::cout << "No mismatch found" << std::endl;
        return;
    }

    if (relative_errors.size() <= 10) {
        std::cout << "Found at most 10 mismatches, Relative Errors:" << std::endl;
        for (auto err : relative_errors) std::cout << "\t" << err << std::endl;
        return;
    }

    // Find min and max relative errors
    double min_error = *std::min_element(relative_errors.begin(), relative_errors.end());
    double max_error = *std::max_element(relative_errors.begin(), relative_errors.end());

    // Parameters for binning
    const int num_bins = 10;
    double bin_width = (max_error - min_error) / num_bins;

    // Ensure valid bin width
    assert(bin_width > 0 && "Bin width must be positive");

    // Count errors in bins using direct computation
    std::vector<int> bin_counts(num_bins, 0);
    for (double error : relative_errors) {
        int bin_index = std::min(static_cast<int>((error - min_error) / bin_width), num_bins - 1);
        bin_counts[bin_index]++;
    }

    // Print histogram
    std::cout << "Relative Error Range:\tCount\n";
    for (int i = 0; i < num_bins; ++i) {
        double bin_start = min_error + i * bin_width;
        double bin_end = bin_start + bin_width;
        std::cout << "[" << std::scientific << std::setprecision(3) << bin_start << ", " << bin_end << "):\t" << bin_counts[i] << "\n";
    }
}

// Function to generate a sequential input vector from loop iterator
void generateSequentialVector(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = 1.0f * (i + 1) / (i+ 2);
    }
}
// CPU SpMV implementation
void cpuSpMV(int rows, int nnz, const std::vector<int>& rowInd, const std::vector<int>& colInd, 
             const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y, 
             float alpha, float beta) {
    for (int i = 0; i < rows; ++i) {
        y[i] *= beta;  // Apply the beta factor to the initial values of y
    }
    
    for (int i = 0; i < nnz; ++i) {
        y[rowInd[i]] += alpha * values[i] * x[colInd[i]];
    }
}