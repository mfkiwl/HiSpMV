# MAD-HiSpMV  
**MAtrix Adaptive Design for Highly Imbalanced SpMV Accelerator (with GeMV Support) on HBM-based FPGAs**

MAD-HiSpMV is a HLS based accelerator designed for **Sparse Matrix–Vector Multiplication (SpMV)** on **HBM-equipped FPGAs**. It incorporates **matrix-adaptive designs** to efficiently handle highly imbalanced sparse workloads.  

With the **dense overlay option**, MAD-HiSpMV also supports **General Matrix–Vector Multiplication (GeMV)**, making it suitable for **mixed sparse–dense workloads** such as hybrid HPC + DNN tasks.

---

## Features
- FPGA-accelerated **SpMV/GeMV** with matrix-adaptive design.
- **Automation tool** to generate accelerator configurations based on input matrix properties.  
- **Dense overlay mode** for GeMV support.  
- Benchmarking support across **FPGA, CPU (Intel MKL), and GPU (NVIDIA cuSPARSE)** with power measurement.  
- Includes prebuilt accelerator designs for **Xilinx Alveo U280** and **U50**.  

---

## Requirements

### Software
- [Vitis HLS 2023.2+](https://www.xilinx.com/products/design-tools/vitis.html)  
- [Xilinx XRT](https://xilinx.github.io/XRT/)  
- [PASTA + AutoBridge (sb-dev branch)](https://github.com/SFU-HiAccel/pasta-hybridbuffer/tree/sb-dev)  
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)  

> ⚠️ Note: The PASTA+AutoBridge repo is **private** until publication. Please request access if needed.

---

## ⚙️ Setup Instructions

1. **Create and activate a Conda environment**  
   Install [PASTA](https://github.com/SFU-HiAccel/pasta-hybridbuffer/tree/sb-dev) following its instructions.  

2. **Clone this repository and set up environment**  
   ```bash
   load_vitis23
   source miniconda3/bin/activate your_conda_env 
   cd HiSpMV 
   source setup
   cd -
   export CONDA_LOC=$(PWD)/miniconda3
   ```
   - `load_vitis23`: loads Vitis HLS & XRT path variables.  
   - `setup`: sets required environment variables for MAD-HiSpMV.  

3. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download benchmarking matrices**  
   ```bash
   python get_tb_matrices.py
   ```

---

## 📂 Repository Structure

```
apps             # Python apps: run SpMV/GeMV + sample DNN model
automation_tool  # Scripts to auto-generate accelerator configs (matrix-adaptive)
builds           # Source code + xclbin for U280/U50 configs, usage reports, floorplans
common           # Common host + kernel source code
cpu              # CPU benchmarking (Intel MKL SpMV/GeMV + power measurement)
gpu              # GPU benchmarking (cuSPARSE SpMV + power measurement)
matrices         # Storage for benchmarking matrices (downloaded by script)
pyhispmv         # pybind11 wrapper to invoke FPGA kernels via XRT
get_tb_matrices.py  # Script to fetch test/benchmarking matrices
requirements.txt # Python dependencies
setup            # Environment setup script
README.md        # Project documentation
```

---

## 🚀 Example Usage

### FPGA Benchmarks (Python Apps)

1. **Build the `pyhispmv` package**  
   ```bash
   cd pyhispmv
   python setup.py build_ext --inplace
   cd ..
   ```

2. **Run SpMV/GeMV tests**  
   - **General test (no arguments):**  
     ```bash
     cd apps
     python general_test.py
     ```

   - **DNN model test (configurable):**  
     ```bash
     cd apps
     python model_test.py \
       --batch_size 1 \
       --input_size 4096 \
       --hidden_size_1 8192 \
       --hidden_size_2 8192 \
       --output_size 1024 \
       --density1 0.1 \
       --density2 0.25
     ```

   - **Note on device selection:**  
     Both scripts require setting `device_id` (the FPGA index).  
     To find available devices, run:  
     ```bash
     xbutil examine
     ```
     Update `device_id` in the scripts to match the U280 board.

---

### CPU Benchmarks (Intel MKL)
```bash
cd cpu
make clean all
./run_spmv.sh   # Run SpMV benchmarks
./run_gemv.sh   # Run GeMV benchmarks
```

---

### GPU Benchmarks (NVIDIA cuSPARSE)
```bash
cd gpu
make clean all
./run_all.sh    # Run all SpMV benchmarks
```

---

### Automation Tool (Matrix-Adaptive Design Generation)

The **automation tool** allows generating accelerator configurations either **automatically (matrix-adaptive)** or **manually (explicit parameters)**.

---

#### Option 1: Automatic Configuration (`main.py`)

`automation_tool/src/main.py` analyzes the input matrix and automatically chooses optimal parameters such as HBM channel usage and optimizations.

**Command:**
```bash
cd automation_tool/src
python main.py <build_dir> --device {U50|U280|V80} [--matrices <file_or_dir>] [--dense-overlay]
```

**Arguments:**
- `build_dir` (positional): Path to the build directory.  
- `--device`: Target device (`U50`, `U280`, or `V80`) **[required]**.  
- `--matrices`: Path to a matrix file or a directory containing matrices.  
- `--dense-overlay`: Enable dense overlay mode (SpMV kernel with GeMV support).  

⚠️ **Important Notes:**  
- In **normal mode** (without `--dense-overlay`), the tool uses the input matrix to **tailor the accelerator design**.  
- In **dense overlay mode**, the design is **not tailored** to the input sparse matrix, and the `--matrices` argument is **ignored**. The generated kernel supports both SpMV and GeMV for mixed workloads.

**Examples:**
- Generate SpMV design for U280 with matrix directory:  
  ```bash
  python main.py ../../builds --device U280 --matrices ../matrices/
  ```
- Generate SpMV+GeMV hybrid design for U50 (no matrices needed):  
  ```bash
  python main.py ../../builds --device U50 --dense-overlay
  ```

---

#### Option 2: Manual Configuration (`spmvcodegen.py`)

`automation_tool/src/rsc/spmvcodegen.py` provides **fine-grained control** over accelerator parameters instead of relying on automation.  

**Command:**
```bash
cd automation_tool/src/
python spmvcodegen.py <output_dir> --device {U50|U280} [options]
```

**Arguments:**
- `output_dir`: Path to the output directory (**warning: will be erased if it exists**).  
- `--device`: Target FPGA device (`U50` or `U280`) **[required]**.  
- `--num-ch-A`: Number of HBM channels for sparse matrix A (default: 16).  
- `--num-ch-x`: Number of HBM channels for input vector x (default: 1).  
- `--num-ch-y`: Number of HBM channels for output vector y (default: 1).  
- `--ch-width`: Width of HBM channels in bits (default: 512).  
- `--urams-per-pe`: URAM banks per PE for output accumulation (default: 2).  
- `--dense-overlay`: Enable dense overlay for GeMV support.  
- `--pre-accumulator`: Enable pre-accumulator optimization.  
- `--row-dist-net`: Enable row distribution network.  
- `--high-freq`: Build hardware for 400 MHz kernel clock.  
**Example (small dense-overlay design):**
```bash
python ../../automation_tool/src/spmvcodegen.py ../ --device U280 \
  --num-ch-A 4 --num-ch-x 1 --num-ch-y 1 --urams-per-pe 1 --row-dist-net --dense-overlay
```

**Example log output:**
```
20250822:204011 [INFO]  Resource: FPGAResource(bram=128, uram=32, dsp=613, lut=134724, reg=135873)
20250822:204011 [INFO]  Successfully Generated Code at ../Dense-HI-SpMV-4-1-1
```

---

### Build and Test the Generated Design

1. **Navigate to the generated design directory**  
   The script automatically names the directory with configuration info:
   ```bash
   cd ../Dense-HI-SpMV-4-1-1
   ```

2. **Build host code**  
   ```bash
   make host
   ```

3. **Run C simulation (HLS source code)**  
   - **Sparse matrix input (SpMV):**
     ```bash
     ./spmv-host ../../matrices/poli_large/poli_large.mtx
     ```
   - **Dense matrix input (dense overlay / GeMV):**
     ```bash
     ./spmv-host 512 512
     ```
     where `512 512` specifies rows and columns of the dense matrix.

4. **Run hardware-software co-simulation**  
   First, synthesize the RTL code:
   ```bash
   make tapa
   ```
   Then run co-simulation using the Vivado TAPA fast cosim:
   ```bash
   ./spmv-host 512 512 --bitstream="spmv.xilinx_u280_gen3x16_xdma_1_202211_1.hw.xo"
   ```

5. **Build final hardware bitstream**  
   ```bash
   make hw
   ```

6. **Run on actual FPGA hardware**  
   ```bash
   ./spmv-host ../../matrices/analytics/analytics.mtx \
       --bitstream="vitis_run_hw/SpMV_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin"
   ```

This workflow covers **dense-overlay design generation**, **C simulation**, **co-simulation**, and **execution on real FPGA hardware**.

---

## 📖 Citation
If you use MAD-HiSpMV in your work, please cite our upcoming publication (to be added here after acceptance).  
