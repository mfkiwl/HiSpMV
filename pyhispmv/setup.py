import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Ensure required environment variables are set
required_env_vars = ["XILINX_XRT", "CONDA_PREFIX"]
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable '{var}' is not defined. Please set it before building.")

# Get environment variables
xilinx_xrt = os.environ["XILINX_XRT"]
conda_prefix = os.environ["CONDA_PREFIX"]

ext_modules = [
    Pybind11Extension(
        "pyhispmv",
        [
            "src/pyhispmv_bindings.cpp",  # PyBind11 bindings
            "src/fpga_handle.cpp",        # FpgaHandle implementation
            "../common/src/spmv-helper.cpp",  # spmv-helper implementation
            "../common/src/fpga-power.cpp" # fpga power monitor implementation
        ],
        include_dirs=[
            "include",                     # Headers for FpgaHandle
            "../common/include",           # Headers for spmv-helper
            os.path.join(xilinx_xrt, "include"),  # XRT headers
        ],
        libraries=[
            "tapa", "OpenCL", "pthread", "stdc++", "xrt_coreutil"
        ],
        library_dirs=[
            os.path.join(conda_prefix, "lib"),  # Conda libraries if relevant
            os.path.join(xilinx_xrt, "lib"),    # XRT libraries
        ],
        extra_compile_args=["-O2", "-std=c++17", "-fopenmp"],  
        extra_link_args=["-fopenmp"],  # Link with OpenMP
    ),
]

setup(
    name="pyhispmv",
    version="0.1",
    description="Python bindings for FPGA-based SpMV kernel (HiSpMV)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
