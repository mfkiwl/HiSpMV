#include "spmv-helper.h"


// Constructor implementation
HiSpmvHandle::HiSpmvHandle(int a, int b, int c, int width, int urams, int fp_acc_latency, bool dense, bool pre_acc, bool row_dist)
    : num_ch_A(a), num_ch_B(b), num_ch_C(c),
      ch_width(width), urams_per_pe(urams), fp_acc_latency(fp_acc_latency),
      dense_overlay(dense), pre_accumulator(pre_acc), row_dist_net(row_dist) {
    
    pes_per_ch = ch_width/64; // one encoded element of A is 64 bit
    fp32_per_ch = ch_width/32; // fp32 used for dense arrays
    num_pes = num_ch_A * pes_per_ch;
    num_fp32s_b = (num_ch_B * fp32_per_ch); 
    num_fp32s_c = (num_ch_C * fp32_per_ch); 
    assert((num_pes % num_fp32s_c == 0) && "Number of PEs should be an  integer multiple of Number of FP32 elements in output vector");
    max_window = (num_fp32s_b / 2) * 1024; // 1k fp32 per BRAM36K and 2 ports
    max_depth = num_pes * urams_per_pe * 4096; // Eache URAM has 4K adresses
    displayConfig();
}

// Method to display the configuration
void HiSpmvHandle::displayConfig() const {
    std::cout << "\nHardware Configuration:" << std::endl;
    std::cout << "\tNumber of Channels (A): " << num_ch_A << std::endl;
    std::cout << "\tNumber of Channels (B): " << num_ch_B << std::endl;
    std::cout << "\tNumber of Channels (C): " << num_ch_C << std::endl;
    std::cout << "\tChannel Width: " << ch_width << std::endl;
    std::cout << "\tURAMs per PE: " << urams_per_pe << std::endl;
    std::cout << "\tDense Overlay: " << (dense_overlay ? "Enabled" : "Disabled") << std::endl;
    std::cout << "\tPre-Accumulator: " << (pre_accumulator ? "Enabled" : "Disabled") << std::endl;
    std::cout << "\tRow Distribution Network: " << (row_dist_net ? "Enabled" : "Disabled") << "\n" << std::endl;
}

// Method to load COO Matrix from File
COOMatrix_t HiSpmvHandle::loadMtx(const std::string& mtx_file) {
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

    // Check for supported symmetry types
    if (symmetry != MM_GENERAL_STR && symmetry != MM_SYMM_STR && symmetry != MM_SKEW_STR) {
        throw std::runtime_error("Error: Unsupported symmetry type. Only 'general', 'symmetric', and 'skew-symmetric' are supported.");
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
    line_stream >> rows >> cols >> nnz;

    // Resize the COO matrix data
    int max_entries = (symmetry == MM_GENERAL_STR) ? nnz : 2 * nnz;
    mtx.rows.reserve(max_entries);
    mtx.cols.reserve(max_entries);
    mtx.values.reserve(max_entries);

    int count = 0;
    // Read entries for the sparse matrix
    for (std::getline(file, line); !file.eof(); std::getline(file, line)) {
        std::istringstream entry_stream(line);
        int r, c;
        float value;

        entry_stream >> r >> c >> value;
        --r;  // Convert to 0-based index
        --c;  // Convert to 0-based index

        if (data_type == MM_PATTERN_STR) {
            value = 1.0f;  // For pattern matrices, the values are always 1
        }

        if (value == 0) {
            continue;
        }

        // Add entry (r, c)
        mtx.rows.push_back(r);
        mtx.cols.push_back(c);
        mtx.values.push_back(value);
        ++count;

        // Handle symmetry: Add (c, r) if not on the diagonal
        if (symmetry == MM_SYMM_STR && r != c) {
            mtx.rows.push_back(c);
            mtx.cols.push_back(r);
            mtx.values.push_back(value);
            ++count;
        }

        // For skew-symmetric matrices, negate the value for the reverse entry
        else if (symmetry == MM_SKEW_STR && r != c) {    
            mtx.rows.push_back(c);
            mtx.cols.push_back(r);
            mtx.values.push_back(-value);
            ++count;
        }
    }

    nnz = count;
    std::cout << "\nMatrix Properties:" << std::endl;
    std::cout << "\tRows: " << rows << "\n\tCols: " << cols << "\n\tNNZ: " << count << "\n" << std::endl;
    return mtx;
}

// Method to tile and pad Sparse Matrix in COO format into CSR format
std::vector<std::vector<CSRMatrix_t>> HiSpmvHandle::tileAndPad(COOMatrix_t mtx) {
    tileAndPad();
    std::vector<std::vector<CSRMatrix_t>> csr_matrices(row_tiles, std::vector<CSRMatrix_t>(col_tiles));
    
    // Initialize row_offsets
    for (int i = 0; i < row_tiles; i++) 
        for (int j = 0; j < col_tiles; j++)  
            csr_matrices[i][j].row_offsets = std::vector<int>(tile_rows + 1, 0);

    // Compute row count for each tile
    for (int l = 0; l < mtx.rows.size(); l++) {
        int r = mtx.rows[l];
        int c = mtx.cols[l];
        
        // tile indices along the row and col dimensions
        int row_tile_idx = r / tile_rows; 
        int col_tile_idx = c / tile_cols;
        
        // row_idx within the tile
        int tile_row_idx = r % tile_rows;

        //initially row_offsets stores the nnz of each row
        csr_matrices[row_tile_idx][col_tile_idx].row_offsets[tile_row_idx+1]++; 
    }

    // Convert row counts into row offsets 
    // Step 2: Convert row counts to row offsets
    for (int i = 0; i < row_tiles; i++) {
        for (int j = 0; j < col_tiles; j++) {
            auto& tile = csr_matrices[i][j];
            for (int r = 1; r <= tile_rows; r++) {
                tile.row_offsets[r] += tile.row_offsets[r - 1];
            }
        }
    }

    // Initialize col indices and values storage
    for (int i = 0; i < row_tiles; i++) {
        for (int j = 0; j < col_tiles; j++) {
            int tile_size = csr_matrices[i][j].row_offsets[tile_rows];
            csr_matrices[i][j].col_indices = std::vector<int>(tile_size, 0);
            csr_matrices[i][j].values = std::vector<float>(tile_size, 0.0f);
        }
    }

    /// Step 4: Populate col_indices and values
    std::vector<std::vector<std::vector<std::vector<std::pair<int, float>>>>> row_entries(
        row_tiles, std::vector<std::vector<std::vector<std::pair<int, float>>>>(
                       col_tiles, std::vector<std::vector<std::pair<int, float>>>(tile_rows)));

    for (size_t l = 0; l < mtx.rows.size(); l++) {
        int r = mtx.rows[l];      // Global row index
        int c = mtx.cols[l];      // Global column index
        float v = mtx.values[l];  // Value at (r, c)

        // Determine the tile indices
        int row_tile_idx = r / tile_rows;
        int col_tile_idx = c / tile_cols;

        // Determine local row and column indices within the tile
        int local_row = r % tile_rows;
        int local_col = c % tile_cols;

        // Append the column index and value as a pair
        row_entries[row_tile_idx][col_tile_idx][local_row].emplace_back(local_col, v);
    }

    // Step 5: Sort and populate the CSR arrays
    for (int i = 0; i < row_tiles; i++) {
        for (int j = 0; j < col_tiles; j++) {
            auto& tile = csr_matrices[i][j];

            for (int r = 0; r < tile_rows; r++) {
                int start = tile.row_offsets[r];
                int end = tile.row_offsets[r + 1];

                // Sort row entries by column index
                std::sort(row_entries[i][j][r].begin(), row_entries[i][j][r].end());

                // Populate col_indices and values
                for (int k = start, idx = 0; k < end; k++, idx++) {
                    tile.col_indices[k] = row_entries[i][j][r][idx].first;
                    tile.values[k] = row_entries[i][j][r][idx].second;
                }
            }
        }
    }
    return csr_matrices;
}

// Method to tile and pad Dense Matrix (overload)
std::vector<std::vector<float>> HiSpmvHandle::tileAndPad(const std::vector<float>& dense_vals) {
    tileAndPad();
    assert(row_tiles == 1 && "Too many rows, dense mode cannot support more than one tile along row dimension");
    std::vector<std::vector<float>> mtx(padded_rows, std::vector<float>(padded_cols, 0));
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            mtx[i][j] = dense_vals[i * cols + j];
        }
    }
    return std::move(mtx);
}

void HiSpmvHandle::tileAndPad() {
    //Pad rows and cols
    //Rows should be integer multiple of number of pes
    //Cols should be integer multiple of num_b_ch * fp32_per_ch 
    rows_per_pe = ((rows - 1) / num_pes) + 1;
    b_len = ((cols - 1) / num_fp32s_b) + 1;

    padded_rows = rows_per_pe * num_pes;
    padded_cols = b_len * num_fp32s_b;

    tile_rows = std::min(max_depth, padded_rows);
    tile_cols = std::min(max_window, padded_cols);

    row_tiles = ((padded_rows - 1) / tile_rows) + 1;
    col_tiles = ((padded_cols - 1) / tile_cols) + 1;

    std::cout << "\nPadding and Tiling Info:" << std::endl;
    std::cout << "\tPadded Dims: (" << padded_rows << ", " << padded_cols << ")" << std::endl;
    std::cout << "\tTile Dims: (" << tile_rows << ", " << tile_cols << ")" << std::endl;
    std::cout << "\tNum Tiles: (" << row_tiles << ", " << col_tiles << ")" << std::endl;
    std::cout << "\tRows Per PE: " << rows_per_pe << "\n\tB length: " << b_len << "\n" << std::endl;
}

std::vector<int> HiSpmvHandle::balanceWorkload(const CSRMatrix_t& csr_matrix)
{
    int rows_per_tile_per_pe = tile_rows/num_pes;

    std::vector<std::vector<std::pair<int, int>>> row_counts(num_pes, std::vector<std::pair<int, int>>(rows_per_tile_per_pe, {0, 0}));
    std::vector<std::pair<int, int>> pe_workloads(num_pes, {0, 0});

    for (int ii = 0; ii < tile_rows; ii++) {
        int pe_idx = ii % num_pes;
        int row_idx = ii / num_pes;
        int row_count = csr_matrix.row_offsets[ii+1] - csr_matrix.row_offsets[ii];
        row_counts[pe_idx][row_idx] = {ii, row_count};  // Store as pair
        pe_workloads[pe_idx].first = pe_idx;
        pe_workloads[pe_idx].second += row_count;  // Accumulate workload
    }

    // Sorting row count (second element of the pair) in descinding order
    for (int p = 0; p < num_pes; p++) {
        std::sort(row_counts[p].begin(), row_counts[p].end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    return a.second > b.second;
                });
    }

    // Sorting pe workload (second element of the pair) in ascending order
    std::sort(pe_workloads.begin(), pe_workloads.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second < b.second;
            });

    //The load of the last pe after sorting
    int max_pe_load = pe_workloads.back().second;

    #ifdef DEBUGGING
    std::cout << "Tile: (" << i << ", " << j << ")\t Max PE Workload: "<< max_pe_load << std::endl;
    #endif

    std::vector<int> removed_rows;
    int best_load = max_pe_load;

    for (int k = 0; k < num_pes; ++k) {
        int baseline_pe_idx = pe_workloads[k].first;
        int baseline_pe_load = pe_workloads[k].second;

        std::vector<int> curr_removed_rows;
        int extra_load = 0;

        for (int l = k + 1; l < num_pes; ++l) {
            int current_pe_idx = pe_workloads[l].first;
            int current_pe_load = pe_workloads[l].second;
            int current_pe_removed_load = 0;
            int current_pe_removed_count = 0;

            while ((baseline_pe_load < (current_pe_load - current_pe_removed_load)) && (current_pe_removed_count < rows_per_tile_per_pe)) {
                int curr_pe_largest_row_idx = row_counts[current_pe_idx][current_pe_removed_count].first;
                int curr_pe_largest_row_cnt = row_counts[current_pe_idx][current_pe_removed_count].second;
                current_pe_removed_load += curr_pe_largest_row_cnt;
                current_pe_removed_count++;
                extra_load += ((curr_pe_largest_row_cnt - 1) / num_pes) + 1;
                curr_removed_rows.push_back(curr_pe_largest_row_idx);
            }
        }

        int balanced_load = baseline_pe_load + extra_load;
        if (balanced_load < best_load) {
            best_load = balanced_load;
            removed_rows = std::move(curr_removed_rows);
        }
    }

    std::sort(removed_rows.begin(), removed_rows.end());

    #ifdef DEBUGGING
    std::cout << "Best Load= "<< best_load << std::endl;
    #endif
    
    int tile_load = csr_matrix.row_offsets.back();
    float original_imb = (max_pe_load * num_pes) / tile_load;
    float improved_imb = (best_load * num_pes) / tile_load;

    int improvement = (int)((original_imb - improved_imb) * 100);
    
    if (improvement < 10)
        removed_rows.clear();

    return removed_rows;
}


int HiSpmvHandle::computeTileSize(const CSRMatrix_t& csr_matrix, const std::vector<int>& shared_rows) 
{
    int dep_dist = (pre_accumulator && use_pre_accum) ? 1 : fp_acc_latency + 1;
    std::vector<std::vector<int>> pe_workloads(num_pes, std::vector<int>(dep_dist, 0));
    int num_rows = csr_matrix.row_offsets.size() - 1;
    int num_shared_rows = shared_rows.size();
    std::vector<std::pair<int, int>> shared_rows_sorted(num_shared_rows, {0, 0});
    for(int k = 0; k < num_shared_rows; k++) {
        int row_id = shared_rows[k]; 
        int row_size = csr_matrix.row_offsets[row_id+1] - csr_matrix.row_offsets[row_id];
        shared_rows_sorted[k].first = row_id;
        shared_rows_sorted[k].second = row_size;
    }

    // sort shared rows based on their size
    sort(shared_rows_sorted.begin(), 
        shared_rows_sorted.end(),
        [] (const std::pair<int, int> &a, const std::pair<int, int> &b)
        {
            return a.second > b.second;
    });

    for(int k = 0; k < num_shared_rows; k++) {
        int row_id = shared_rows_sorted[k].first; 
        int row_size = shared_rows_sorted[k].second;
        int load_size = ((row_size - 1)/num_pes) + 1;

        for (int pe = 0; pe < num_pes; pe++) {
            int min_idx = 0;
            int min = pe_workloads[pe][min_idx];

            for (int ii=0; ii<dep_dist; ii++) {
                if (pe_workloads[pe][ii] < min) {
                    min_idx = ii;
                    min = pe_workloads[pe][ii];
                }
            }

            pe_workloads[pe][min_idx] += load_size;
        }
    }
    //schedule remaining rows
    int num_rem_rows = num_rows - num_shared_rows;
    std::vector<std::pair<int, int>> sorted_rem_rows(num_rem_rows, {0, 0});
    for(int k = 0, l = 0, m = 0; k < num_rows; k++) {
        if (!shared_rows.empty() && k == shared_rows[l]) 
            l++;

        else {
            int row_size = csr_matrix.row_offsets[k+1] - csr_matrix.row_offsets[k];
            sorted_rem_rows[m].first = k;
            sorted_rem_rows[m].second = row_size;
            m++;
        }
    }
    sort(sorted_rem_rows.begin(), 
        sorted_rem_rows.end(),
        [] (const std::pair<int, int> &a, const std::pair<int, int> &b)
        {
            return a.second > b.second;
    });

    for(int k = 0; k < num_rem_rows; k++) {
        int row_id = sorted_rem_rows[k].first; 
        int row_size = sorted_rem_rows[k].second;
        int pe = row_id % num_pes;
        int min_idx = 0;
        int min = pe_workloads[pe][min_idx];

        for (int ii=0; ii<dep_dist; ii++) {
            if (pe_workloads[pe][ii] < min) {
                min_idx = ii;
                min = pe_workloads[pe][ii];
            }
        }

        pe_workloads[pe][min_idx] += row_size;
    }
    int max_load = pe_workloads[0][0];
    for(int p = 0; p < num_pes; p++) {
        for(int ii = 0; ii < dep_dist; ii++) {
        if(pe_workloads[p][ii] > max_load) 
            max_load = pe_workloads[p][ii];
        }
    }

    return (max_load + PADDING) * dep_dist;
}

void HiSpmvHandle::prepareTile(const CSRMatrix_t& csr_matrix, const std::vector<int>& shared_rows, const int size, const int offset, std::vector<aligned_vector<uint64_t>>& fpga_mtx)
{
    int dep_dist = (pre_accumulator && use_pre_accum) ? 1 : fp_acc_latency + 1;
    //schedule shared rows first
    std::vector<std::vector<int>> pe_loads(num_pes, std::vector<int>(dep_dist, 0));
    int num_rows = csr_matrix.row_offsets.size() - 1;

    int num_shared_rows = shared_rows.size();
    std::vector<std::pair<int, int>> shared_rows_sorted(num_shared_rows, {0,0});
    for(int k = 0; k < num_shared_rows; k++) {
        int row_id = shared_rows[k]; 
        int row_size = csr_matrix.row_offsets[row_id+1] - csr_matrix.row_offsets[row_id];
        shared_rows_sorted[k].first = row_id;
        shared_rows_sorted[k].second = row_size;
    }
    sort(shared_rows_sorted.begin(), 
        shared_rows_sorted.end(),
        [] (const std::pair<int, int> &a, const std::pair<int, int> &b)
        {
            return a.second > b.second;
    });
    int num_rem_rows = num_rows - num_shared_rows;
    std::vector<std::pair<int, int>> rem_rows_sorted(num_rem_rows, {0,0});
    for(int k = 0, l = 0, m = 0; k < num_rows; k++) {
        if (!shared_rows.empty() && k == shared_rows[l])
            l++;

        else {
            int row_size = csr_matrix.row_offsets[k+1] - csr_matrix.row_offsets[k];
            rem_rows_sorted[m].first = k;
            rem_rows_sorted[m].second = row_size;
            m++;
        }
    }
    sort(rem_rows_sorted.begin(), 
        rem_rows_sorted.end(),
        [] (const std::pair<int, int> &a, const std::pair<int, int> &b)
        {
            return a.second > b.second;
    });

    for(int k = 0; k < num_shared_rows; k++) {
        int row_no = shared_rows_sorted[k].first; 
        int row_size = shared_rows_sorted[k].second;
        int load_size = ((row_size - 1)/num_pes) + 1;
        uint16_t rowl16 = (row_no % num_pes);
        uint16_t rowh16 = (row_no / num_pes);
        int row_start = csr_matrix.row_offsets[row_no];
        int row_end = csr_matrix.row_offsets[row_no+1];

        for (int pe = 0; pe < num_pes; pe++) {
            int min_idx = 0;
            int min = pe_loads[pe][min_idx];

            for (int ii=0; ii<dep_dist; ii++) {
                if (pe_loads[pe][ii] < min) {
                    min_idx = ii;
                    min = pe_loads[pe][ii];
                }
            }

            int ch_no = pe / pes_per_ch;
            int inter_ch_pe = pe % pes_per_ch;
            uint16_t row16 = (pe & 1) ? rowh16 : rowl16;
            for(int l = 0; l < load_size; l++) {
                int ind = row_start + (l * num_pes) + pe;
                int col_id = (ind < row_end) ? csr_matrix.col_indices[ind] : 0; 
                float value = (ind < row_end) ? csr_matrix.values[ind] : 0;
                uint32_t val_bits = *(uint32_t*)&value;
                int addr = offset + (((pe_loads[pe][min_idx] * dep_dist) + min_idx) * pes_per_ch) + inter_ch_pe;
                fpga_mtx[ch_no][addr] = encode(false, true, true, row16, col_id, val_bits);
                pe_loads[pe][min_idx]++;
            } 
        }
    }

    //schedule remaining rows
    for(int k = 0; k < num_rem_rows; k++) {
        int row_no = rem_rows_sorted[k].first; 
        int row_size = rem_rows_sorted[k].second;
        int pe = row_no % num_pes;
        int min_idx = 0;
        int min = pe_loads[pe][min_idx];

        for (int ii=0; ii<dep_dist; ii++) {
            if (pe_loads[pe][ii] < min) {
                min_idx = ii;
                min = pe_loads[pe][ii];
            }
        }

        int ch_no = pe / pes_per_ch;
        int inter_ch_pe = pe % pes_per_ch;
        uint16_t row16 = (row_no / num_pes);
        for(int ind = csr_matrix.row_offsets[row_no]; ind < csr_matrix.row_offsets[row_no+1]; ind++) {
            int col_id = csr_matrix.col_indices[ind];
            float value = csr_matrix.values[ind];
            uint32_t val_bits = *(uint32_t*)&value;
            int addr = offset + (((pe_loads[pe][min_idx] * dep_dist) + min_idx) * pes_per_ch) + inter_ch_pe;
            fpga_mtx[ch_no][addr] = encode(false, true, false, row16, col_id, val_bits);
            pe_loads[pe][min_idx]++;
        }  
    }

    for(int p = 0; p < num_pes; p++) {
        int ch_no = p / pes_per_ch;
        int inter_ch_pe = p % pes_per_ch;
        for(int ii = 0; ii < dep_dist; ii++) {
            while(pe_loads[p][ii] < (size/dep_dist)) {
                bool tileEnd = (pe_loads[p][ii] == (size/dep_dist) - 1) && (ii == dep_dist-1);
                int col_id = 0;
                uint16_t row16 = 0;
                float value = 0;
                uint32_t val_bits = *(uint32_t*)&value;
                int addr = offset + ((pe_loads[p][ii]++) * dep_dist + ii) * pes_per_ch + inter_ch_pe;
                fpga_mtx[ch_no][addr] = encode(tileEnd, false, false, row16, col_id, val_bits);
            }
        }
    }
}

// Function to generate a vector based on loop iterator values

double HiSpmvHandle::prepareSparseMtxForFPGA(const std::string& mtx_file) {
    //Load Sparse Matrix File from ".mtx" file
    this->coo_mtx = loadMtx(mtx_file);
    return prepareSparseMtxForFPGA();
}

double HiSpmvHandle::prepareSparseMtxForFPGA(const int rows, const int cols, const std::vector<int>& coo_rows, const std::vector<int>& coo_cols, const std::vector<float>& coo_vals) {
    this->rows = rows;
    this->cols = cols;
    this->nnz = coo_rows.size();
    this->coo_mtx.rows = std::move(coo_rows);
    this->coo_mtx.cols = std::move(coo_cols);
    this->coo_mtx.values = std::move(coo_vals);
    return prepareSparseMtxForFPGA();
}

double HiSpmvHandle::prepareSparseMtxForFPGA() {
    auto start_gen = std::chrono::steady_clock::now();
    //Tile and Pad Matrix and store tiled matrices in CSR format
    std::vector<std::vector<CSRMatrix_t>> tiled_matrices = tileAndPad(coo_mtx);

    //Determine if we need to process any rows in shared mode and store them
    std::vector<std::vector<std::vector<int>>> shared_rows(row_tiles, std::vector<std::vector<int>>(col_tiles));
    if(row_dist_net && use_row_dist)
#pragma omp parallel for collapse(2) schedule(dynamic)
        for(int i = 0; i < row_tiles; i++) 
            for(int j = 0; j < col_tiles; j++) 
                shared_rows[i][j] = std::move(balanceWorkload(tiled_matrices[i][j]));
    
    //Store shared rows 
    for(int i = 0; i < row_tiles; i++) 
        for(int j = 0; j < col_tiles; j++) 
            for(int local_row: shared_rows[i][j])
                shared_row_indices.insert(i * tile_rows + local_row);
        
    //Compute Size of encoded data for each tile
    std::vector<std::vector<int>> tile_sizes(row_tiles, std::vector<int>(col_tiles));
#pragma omp parallel for collapse(2) schedule(dynamic)
    for(int i = 0; i < row_tiles; i++) 
        for(int j = 0; j < col_tiles; j++) 
            tile_sizes[i][j] = computeTileSize(tiled_matrices[i][j], shared_rows[i][j]);

    //Convert tile sizes to tile offsets
    std::vector<std::vector<int>> tile_offsets(row_tiles, std::vector<int>(col_tiles, 0));
    int total_size = 0;
    for(int i = 0; i < row_tiles; i++) {
        for(int j = 0; j < col_tiles; j++) {
            tile_offsets[i][j] = total_size;
            total_size += pes_per_ch * tile_sizes[i][j];
        }
    }
    
    //Prepare the encoded data for each tile in parallel
    std::vector<aligned_vector<uint64_t>> fpga_mtx(num_ch_A, aligned_vector<uint64_t>(total_size));
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < row_tiles; i++) {
        for (int j = 0; j < col_tiles; j++) {
            prepareTile(tiled_matrices[i][j], shared_rows[i][j], tile_sizes[i][j], tile_offsets[i][j], fpga_mtx);
        }
    }
    prep_mtx = std::move(fpga_mtx);
    auto end_gen = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen - start_gen).count();
}

double HiSpmvHandle::prepareDenseMtxForFPGA(const int rows, const int cols, const std::vector<float>& dense_vals) {
    assert(dense_overlay && "Hardware is not built with Dense Overlay, cannot support dense workload");
    auto start_gen = std::chrono::steady_clock::now();
    this->rows = rows;
    this->cols = cols;
    this->nnz = rows*cols;
    
    dense_mtx = tileAndPad(dense_vals);
    uint32_t A_len = rows_per_pe * padded_cols / 2;

    std::vector<aligned_vector<uint64_t>>fpga_mtx(num_ch_A, aligned_vector<uint64_t>(A_len * pes_per_ch, 0));
    //Prepare the encoded data for each tile
    for (int i = 0; i < padded_rows; i++) {
        for (int j = 0; j < col_tiles; j++) {
            int offset = j * rows_per_pe * (tile_cols / 2);
            for (int jj = 0; (jj < tile_cols) && ((j * tile_cols + jj) < padded_cols); jj+=2) {
                int pe = (i % num_pes);
                int ch = (pe / pes_per_ch);
                int inter_ch_pe = (pe % pes_per_ch);
                int addr = offset + (jj/2) * rows_per_pe + (i / num_pes);
                uint64_t temp = 0;
                uint32_t val0 = *(uint32_t*)&(dense_mtx[i][j * tile_cols + jj]);
                uint32_t val1 = *(uint32_t*)&(dense_mtx[i][j * tile_cols + jj + 1]);
                temp |= val1;
                temp <<= 32;
                temp |= val0;
                fpga_mtx[ch][addr * pes_per_ch + inter_ch_pe] = temp;
            }
        }
    }
    prep_mtx = std::move(fpga_mtx);
    auto end_gen = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen - start_gen).count();
}

const COOMatrix_t HiSpmvHandle::getMatrix() {
    return coo_mtx;
}

std::pair<int, int> HiSpmvHandle::getMatrixDims() {
    return {rows, cols};
}

int HiSpmvHandle::getNNZ() {
    return nnz;
}

std::pair<int, int> HiSpmvHandle::getPaddedMatrixDims() {
    return {padded_rows, padded_cols};
}

uint16_t HiSpmvHandle::getRowTiles() {
    return row_tiles;
}

uint16_t HiSpmvHandle::getColTiles() {
    return col_tiles;
}

uint32_t HiSpmvHandle::getTotTiles() {
    return row_tiles * col_tiles;
}

uint32_t HiSpmvHandle::getRowsPerPE() {
    return rows_per_pe;
}

uint32_t HiSpmvHandle::getVectLength() {
    return b_len;
}

uint32_t HiSpmvHandle::getRunLength() {
    return prep_mtx[0].size()/pes_per_ch;
}

const std::vector<aligned_vector<uint64_t>> HiSpmvHandle::getPreparedMtx() {
    return prep_mtx;
}

bool HiSpmvHandle::isSharedRow(int row_idx) {
    return (shared_row_indices.find(row_idx) != shared_row_indices.end());
}

bool HiSpmvHandle::isDense() {
    return (!dense_mtx.empty());
}

double HiSpmvHandle::cpuSequential(const std::vector<float> B, std::vector<float>& Cin, const float alpha, const float beta, std::vector<float>& Cout) {
    // Perform matrix-vector multiplication
    auto start_cpu = std::chrono::steady_clock::now();
    if (isDense()) 
        for (int i = 0; i < padded_rows; ++i) 
            for (int j = 0; j < padded_cols; ++j) 
                Cout[i] += dense_mtx[i][j] * B[j];
    else {
        for (int i = 0; i < nnz; ++i) {
            int row = coo_mtx.rows[i];
            int col = coo_mtx.cols[i];
            float value = coo_mtx.values[i];
            Cout[row] += value * B[col];
        }
    }

    for (int i = 0; i < Cout.size(); i++)
        Cout[i] = (alpha*Cout[i]) + (beta*Cin[i]);

    auto end_cpu = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
}

void HiSpmvHandle::printErrorStats(const std::vector<float>& cpu_ref, const std::vector<float>& fpga_out) 
{
    if (cpu_ref.size() != fpga_out.size()) {
        throw std::runtime_error("Error: Vector sizes do not match!");
    }

    //Compute Relative Errors
    std::vector<double> relative_errors;
    for (size_t i = 0; i < cpu_ref.size(); ++i) {
        double fpga = static_cast<double>(std::fabs(fpga_out[i]));
        double cpu = static_cast<double>(std::fabs(cpu_ref[i]));
        double epsilon = std::numeric_limits<double>::lowest();
        double ref = std::max(cpu, epsilon);
        double rel_err = std::fabs(fpga - ref) / ref;
        #ifdef DEBUGGING
        if (isSharedRow(i)) 
            std::cout << "[shared mode]";
        else 
            std::cout << "[normal mode]";
        std::cout << "\tRelative Error: " << std::scientific << std::setprecision(3) << rel_err << "\tRow Idx: " << i << "\tCPU Ref: " << cpu_ref[i] << "\tFPGA Out: " << fpga_out[ch][addr] << std::endl;
        #endif
        if (rel_err!= 0) relative_errors.push_back(rel_err);
    }
  
    if (relative_errors.empty()) {
        std::cout << "No mismatch found" << std::endl;
        return;
    }

    if (relative_errors.size() <= 10) {
        std::cout << "Found atmost 10 mismatches, Relative Errors:" << std::endl;
        for(auto err: relative_errors) std::cout << "\t" << err << std::endl;
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

std::vector<aligned_vector<float>> HiSpmvHandle::prepareInputVector(const std::vector<float>& b) {
    assert(b.size() == cols && "Expected input vector to match col dimension of the matrix");
    std::vector<aligned_vector<float>>fpgaBinVect(num_ch_B, aligned_vector<float>(padded_cols/num_ch_B, 0));
    for (int i = 0; i < cols; i++) {
        int ch = (i / fp32_per_ch) % num_ch_B;
        int addr = (i / (fp32_per_ch * num_ch_B)) * fp32_per_ch + (i % fp32_per_ch);
        fpgaBinVect[ch][addr] = b[i];
    }
    return std::move(fpgaBinVect);
}

std::vector<aligned_vector<float>> HiSpmvHandle::prepareBiasVector(const std::vector<float>& c_in) {
    assert(c_in.size() == rows && "Expected bias vector to match row dimension of the matrix");
    std::vector<aligned_vector<float>>fpgaCinVect(num_ch_C, aligned_vector<float>(padded_rows/num_ch_C, 0));
    for (int i = 0; i < rows; i++) {
        int ch = (i / fp32_per_ch) % num_ch_C;
        int addr = (i / (fp32_per_ch * num_ch_C)) * fp32_per_ch + (i % fp32_per_ch);
        fpgaCinVect[ch][addr] = c_in[i];
    }
    return std::move(fpgaCinVect);
}

std::vector<aligned_vector<float>> HiSpmvHandle::allocateOutputVector() {
    std::vector<aligned_vector<float>>fpgaCoutVect(num_ch_C, aligned_vector<float>(padded_rows/num_ch_C, 0));
    return std::move(fpgaCoutVect);
}

std::vector<float> HiSpmvHandle::collectOutputVector(const std::vector<aligned_vector<float>>& fpgaCoutVect) {
    assert(fpgaCoutVect.size() == num_ch_C && "Output Channels doesn't match the expected value");
    int out_size = padded_rows/num_ch_C;
    assert(fpgaCoutVect[0].size() == out_size && "Ouput Size doesn't match the expected size");
    std::vector<float> c_out(rows);
    for (int i = 0; i < rows; i++) {
        int ch = (i / fp32_per_ch) % num_ch_C;
        int addr = (i / (fp32_per_ch * num_ch_C)) * fp32_per_ch + (i % fp32_per_ch);
        c_out[i] = fpgaCoutVect[ch][addr];
    }    
    return std::move(c_out);
}