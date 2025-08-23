#include "helper_functions.h"
#include "mmio.h"

long long PowerLogger::readEnergy(const std::string& path) const {
    std::ifstream file(path);
    long long value = 0;
    file >> value;
    return value;
}

void PowerLogger::start() {
    energyStart = readEnergy(energyFile);
    timeStart = std::chrono::steady_clock::now();
}

void PowerLogger::stop() {
    energyEnd = readEnergy(energyFile);
    timeEnd = std::chrono::steady_clock::now();

    // handle wrap-around
    if (energyEnd < energyStart) {
        long long maxEnergy = readEnergy(maxEnergyFile);
        energyEnd += maxEnergy;
    }
}

double PowerLogger::getAveragePower() const {
    double deltaE = (energyEnd - energyStart) / 1e6; // convert uJ to J
    double deltaT = std::chrono::duration<double>(timeEnd - timeStart).count(); // s
    return (deltaT > 0) ? deltaE / deltaT : 0.0;
}

// function from Serpens to read mtx file
int cmp_by_column_row(const void *aa,
                      const void *bb) {
    rcv * a = (rcv *) aa;
    rcv * b = (rcv *) bb;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    return 0;
}

void sort_by_fn(int nnz_s,
                std::vector<int> & cooRowIndex,
                std::vector<int> & cooColIndex,
                std::vector<float> & cooVal,
                int (* cmp_func)(const void *, const void *)) {
    rcv * rcv_arr = new rcv[nnz_s];
    
    for(int i = 0; i < nnz_s; ++i) {
        rcv_arr[i].r = cooRowIndex[i];
        rcv_arr[i].c = cooColIndex[i];
        rcv_arr[i].v = cooVal[i];
    }
    
    qsort(rcv_arr, nnz_s, sizeof(rcv), cmp_func);
    
    for(int i = 0; i < nnz_s; ++i) {
        cooRowIndex[i] = rcv_arr[i].r;
        cooColIndex[i] = rcv_arr[i].c;
        cooVal[i] = rcv_arr[i].v;
    }
    
    delete [] rcv_arr;
}

void mm_init_read(FILE * f,
                  char * filename,
                  MM_typecode & matcode,
                  int & m,
                  int & n,
                  int & nnz) {

    if (mm_read_banner(f, &matcode) != 0) {
        std::cout << "Could not process Matrix Market banner for " << filename << std::endl;
        exit(1);
    }
    
    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) != 0) {
        std::cout << "Could not read Matrix Market format for " << filename << std::endl;
        exit(1);
    }
}

void load_S_matrix(FILE* f_A,
                   int nnz_mmio,
                   int & nnz,
                   std::vector<int> & cooRowIndex,
                   std::vector<int> & cooColIndex,
                   std::vector<float> & cooVal,
                   MM_typecode & matcode) {
    
    if (mm_is_complex(matcode)) {
        std::cout << "Reading in a complex matrix, not supported yet!" << std::endl;
        exit(1);
    }
    
    if (!mm_is_symmetric(matcode)) {
        std::cout << "It's an NS matrix.\n";
    } else {
        std::cout << "It's an S matrix.\n";
    }
    
    int r_idx, c_idx;
    float value;
    int idx = 0;
    
    for (int i = 0; i < nnz_mmio; ++i) {
        if (mm_is_pattern(matcode)) {
            fscanf(f_A, "%d %d\n", &r_idx, &c_idx);
            value = 1.0;
        }else {
            fscanf(f_A, "%d %d %f\n", &r_idx, &c_idx, &value);
        }
        
        unsigned int * tmpPointer_v = reinterpret_cast<unsigned int*>(&value);;
        unsigned int uint_v = *tmpPointer_v;
        if (uint_v != 0) {
            if (r_idx < 1 || c_idx < 1) { // report error
                std::cout << "idx = " << idx << " [" << r_idx - 1 << ", " << c_idx - 1 << "] = " << value << std::endl;
                exit(1);
            }
            
            cooRowIndex[idx] = r_idx - 1;
            cooColIndex[idx] = c_idx - 1;
            cooVal[idx] = value;
            idx++;
            
            if (mm_is_symmetric(matcode)) {
                if (r_idx != c_idx) {
                    cooRowIndex[idx] = c_idx - 1;
                    cooColIndex[idx] = r_idx - 1;
                    cooVal[idx] = value;
                    idx++;
                }
            }
        }
    }
    nnz = idx;
}

void readMatrixCSC(char* filename, std::vector<float>& values, std::vector<int>& rowIndices, std::vector<int>& colOffsets, int& rows, int& cols, int& nnz) {
    int nnz_mmio;
    MM_typecode matcode;
    FILE * f_A;
    
    if ((f_A = fopen(filename, "r")) == NULL) {
        std::cout << "Could not open " << filename << std::endl;
        exit(1);
    }
    
    mm_init_read(f_A, filename, matcode, rows, cols, nnz_mmio);
    
    if (!mm_is_coordinate(matcode)) {
        std::cout << "The input matrix file " << filename << "is not a coordinate file!" << std::endl;
        exit(1);
    }
    
    int nnz_alloc = (mm_is_symmetric(matcode))? (nnz_mmio * 2): nnz_mmio;
    //std::cout << "Matrix A -- #row: " << rows << " #col: " << cols << std::endl;
    
    std::vector<int> cooRowIndex(nnz_alloc);
    std::vector<int> cooColIndex(nnz_alloc);
    //eleIndex.resize(nnz_alloc);
    values.resize(nnz_alloc);
    
    //std::cout << "Loading input matrix A from " << filename << "\n";
    
    load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, values, matcode);
    
    fclose(f_A);
    
    sort_by_fn(nnz, cooRowIndex, cooColIndex, values, cmp_by_column_row);
    
    // convert to CSC format
    int M_K = cols;
    colOffsets.resize(M_K+1);
    std::vector<int> counter(M_K, 0);
    
    for (int i = 0; i < nnz; i++) {
        counter[cooColIndex[i]]++;
    }
    
    int t = 0;
    for (int i = 0; i < M_K; i++) {
        t += counter[i];
    }
    
    colOffsets[0] = 0;
    for (int i = 1; i <= M_K; i++) {
        colOffsets[i] = colOffsets[i - 1] + counter[i - 1];
    }
    
    rowIndices.resize(nnz);
    
    for (int i = 0; i < nnz; ++i) {
        rowIndices[i] = cooRowIndex[i];
    }
    
    if (mm_is_symmetric(matcode)) {
        //eleIndex.resize(nnz);
        values.resize(nnz);
    }
}

void convertCSCtoCSR(const std::vector<float>& cscValues, const std::vector<int>& cscRowIndices, const std::vector<int>& cscColOffsets,
                     std::vector<float>& csrValues, std::vector<int>& csrColIndices, std::vector<int>& csrRowOffsets, int rows, int cols, int nnz) {
    // allocate memory
    csrValues.resize(nnz);
    csrColIndices.resize(nnz);
    csrRowOffsets.resize(rows + 1);
    std::vector<int> rowCounts(rows, 0);

    for (int i = 0; i < nnz; i++) {
        rowCounts[cscRowIndices[i]]++;
    }

    // convert rowCounts to cumulative sum
    csrRowOffsets[0] = 0;
    for (int i = 0; i < rows; i++) {
      csrRowOffsets[i+1] = csrRowOffsets[i] + rowCounts[i];
    }

    std::vector<int> rowOffset(rows, 0);
    // fill csrValues and csrColIndices
    for (int j = 0; j < cols; j++) {
        for (int i = cscColOffsets[j]; i < cscColOffsets[j + 1]; i++) {
            int row = cscRowIndices[i];
            int index = csrRowOffsets[row] + rowOffset[row];
            csrValues[index] = cscValues[i];
            csrColIndices[index] = j;
            rowOffset[row]++;
        }
    }
}

void printMatrixCSR(std::vector<float> values, std::vector<int> columns, std::vector<int> rowPtr, int numRows, int numCols) {
    // Print the matrix in CSR format
    std::cout << "Matrix in dense format:" << std::endl;
    for (int i = 0; i < numRows; i++) {
        int prev_col = 0;
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
            int col = columns[j];
            float val = values[j];
            for (int k = prev_col; k < col; k++)
                printf("%.4f; ", 0.0);
            printf("%.4f; ",val);
            prev_col = col + 1;
        }
        for (int k = prev_col; k < numCols; k++)
                printf("%.4f; ", 0.0);
        printf("\n");
    }
}

