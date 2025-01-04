#include "fpga_handle.h"  // Include your class header

PYBIND11_MODULE(pyhispmv, m) {
    m.doc() = "Python binding for FPGA-based SpMV kernel";

    py::class_<FpgaHandle>(m, "FpgaHandle")
        // Constructor
        .def(py::init<const std::string&, int, int, int, int, int, int, int, bool, bool, bool>(),
             py::arg("xclbin_path"), py::arg("device_id"),
             py::arg("num_ch_A"), py::arg("num_ch_B"), py::arg("num_ch_C"),
             py::arg("ch_width"), py::arg("urams_per_pe"), py::arg("fp_acc_latency"),
             py::arg("dense_overlay"), py::arg("pre_accumulator"), py::arg("row_dist_net"))
        
        // Method to create matrix handle for dense matrix
        .def("create_dense_handle", &FpgaHandle::createDenseMtxHandle,
             py::arg("flattened_dense_values"), py::arg("rows"), py::arg("cols"),
             "Creates a matrix handle for a dense matrix")

        // Method to create matrix handle for sparse matrix
        .def("create_sparse_handle", &FpgaHandle::createSparseMtxHandle,
             py::arg("coo_rows"), py::arg("coo_cols"), py::arg("coo_values"),
             py::arg("rows"), py::arg("cols"),
             "Creates a matrix handle for a sparse matrix")

        // Method to load matrices onto the FPGA
        .def("load_matrices", &FpgaHandle::loadMatrices, "Loads matrices onto FPGA")

        // Method to select a matrix
        .def("select_matrix", &FpgaHandle::selectMatrix, py::arg("matrix_idx"), "Select a matrix by its index")

        // Method to run the kernel
        .def("run_kernel", &FpgaHandle::runKernel,
             py::arg("x"), py::arg("bias"), py::arg("y"), py::arg("alpha"), py::arg("beta"),
             "Runs the SpMV kernel with the provided input/output vectors and scalars")

        // Method to run the kernel
        .def("run_kernels", &FpgaHandle::runKernels,
             py::arg("x"), py::arg("bias"), py::arg("y"), py::arg("alpha"), py::arg("beta"),
             "Runs the SpMM kernel with the provided input/output vectors and scalars");
}
