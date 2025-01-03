import numpy as np
import pyhispmv  # Import the compiled C++ module
import time

# Initialize the FPGA handle
xclbin_path = "../builds/Dense-HiSpMV-24-1-1/SpMV_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin"
device_id = 0  # Example device ID
num_ch_A = 24
num_ch_B = 1
num_ch_C = 1
ch_width = 512
urams_per_pe = 2
fp_acc_latency = 5
dense_overlay = True
pre_accumulator = False
row_dist_net = True

# Create an FpgaHandle object
fpga = pyhispmv.FpgaHandle(
    xclbin_path, device_id, num_ch_A, num_ch_B, num_ch_C, 
    ch_width, urams_per_pe, fp_acc_latency, dense_overlay, pre_accumulator, row_dist_net
)

# Generate a very large dense matrix for testing
rows, cols = 10000, 10000
dense_values = np.random.rand(rows, cols).astype(np.float32)  # Dense matrix of size 10000x10000
x = np.random.rand(cols).astype(np.float32)  # Input vector (10000 elements)
bias = np.random.rand(rows).astype(np.float32)  # Bias vector (10000 elements)
y_dense = np.zeros(rows, dtype=np.float32)  # Output vector for dense matrix (10000 elements)

# Create a dense matrix handle
dense_handle_idx = fpga.create_dense_handle(dense_values.flatten(), rows, cols)

# Create the expected output using NumPy (Python computation)
start_time = time.time()
y_dense_expected = np.dot(dense_values, x) + bias  # Expected result from NumPy computation
end_time = time.time()
print(f"Dense matrix computation time (NumPy): {end_time - start_time:.2f} seconds")

# Generate a very large sparse matrix for testing (COO format)
nnz = 100000  # Number of non-zero elements
coo_rows = np.random.randint(0, rows, size=nnz, dtype=np.int32)
coo_cols = np.random.randint(0, cols, size=nnz, dtype=np.int32)
coo_values = np.random.rand(nnz).astype(np.float32)  # Sparse values (non-zero)

# Create a sparse matrix handle
sparse_handle_idx = fpga.create_sparse_handle(coo_rows, coo_cols, coo_values, rows, cols)

# Create the expected output for the sparse matrix (NumPy computation)
y_sparse = np.zeros(rows, dtype=np.float32)  # Output vector for sparse matrix (10000 elements)
y_sparse_expected = np.zeros(rows, dtype=np.float32)  # Output vector for sparse matrix (10000 elements)
# Compute the expected result for sparse matrix multiplication (using NumPy)
start_time = time.time()
for i in range(nnz):
    y_sparse_expected[coo_rows[i]] += coo_values[i] * x[coo_cols[i]]
y_sparse_expected += bias  # Adding bias
end_time = time.time()
print(f"Sparse matrix computation time (NumPy): {end_time - start_time:.2f} seconds")

# Load both dense and sparse matrices onto the FPGA (only once)
fpga.load_matrices()

# # Select the dense matrix created earlier and run the SpMV kernel
start_time_fpga = time.time()
fpga.select_matrix(dense_handle_idx)
fpga.run_kernel(x, bias, y_dense, 1.0, 1.0)
end_time_fpga = time.time()
fpga_execution_time_dense = end_time_fpga - start_time_fpga
print(f"FPGA execution time for dense matrix: {fpga_execution_time_dense:.4f} seconds")

# Select the sparse matrix created earlier and run the SpMV kernel
start_time_fpga = time.time()
fpga.select_matrix(sparse_handle_idx)
fpga.run_kernel(x, bias, y_sparse, 1.0, 1.0)
end_time_fpga = time.time()
fpga_execution_time_sparse = end_time_fpga - start_time_fpga
print(f"FPGA execution time for sparse matrix: {fpga_execution_time_sparse:.4f} seconds")

# Calculate errors for dense matrix
absolute_error_dense = np.abs(y_dense - y_dense_expected)  # Absolute error
relative_error_dense = absolute_error_dense / np.abs(y_dense_expected)  # Relative error

max_absolute_error_dense = np.max(absolute_error_dense)
max_relative_error_dense = np.max(relative_error_dense)

print(f"Maximum Absolute Error for Dense Matrix: {max_absolute_error_dense:.6f}")
print(f"Maximum Relative Error for Dense Matrix: {max_relative_error_dense:.6f}")

# Calculate errors for sparse matrix
absolute_error_sparse = np.abs(y_sparse - y_sparse_expected)  # Absolute error
relative_error_sparse = absolute_error_sparse / np.abs(y_sparse_expected)  # Relative error

max_absolute_error_sparse = np.max(absolute_error_sparse)
max_relative_error_sparse = np.max(relative_error_sparse)

print(f"Maximum Absolute Error for Sparse Matrix: {max_absolute_error_sparse:.6f}")
print(f"Maximum Relative Error for Sparse Matrix: {max_relative_error_sparse:.6f}")

# Verify the result for dense matrix
print("Dense matrix SpMV result verification:")
if np.allclose(y_dense, y_dense_expected, rtol=1e-3):
    print("Dense matrix result is correct!")
else:
    print("Dense matrix result is incorrect!")

# Verify the result for sparse matrix
print("Sparse matrix SpMV result verification:")
if np.allclose(y_sparse, y_sparse_expected, rtol=1e-3):  # Compare with expected result
    print("Sparse matrix result is correct!")
else:
    print("Sparse matrix result is incorrect!")
