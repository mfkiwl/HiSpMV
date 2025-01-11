import os
import shutil
import math
from pathlib import Path
import argparse
import logging

from commons import SpMVConfig, encodeSpMVConfig, FPGA, FPGAResource
from resource_est import ResourceEstimator
from crossbar import CrossBarGen
from fpgas import U280, U50


# Get the current file's directory
current_dir = Path(__file__).resolve().parent

#automation_tool fir
parent_dir = current_dir.parents[0]

# Navigate two levels up
project_dir = current_dir.parents[1]

logger = logging.getLogger(__name__)

class SpMVCodeGen:
    def __init__(self, spmvConfig, build_dir, fpga):
        self.config = spmvConfig
        self.fpga = fpga
        self.pes_per_ch = self.config.ch_width // 64
        self.num_pes = self.config.num_ch_A * self.pes_per_ch
        self.b_part = (self.config.num_ch_B * self.config.ch_width) // 32 // 2 #BRAM36K 2Ports
        self.b_window = min(self.b_part * 1024, 1 << 14)
        self.w_window = self.b_window // self.b_part // 2
        self.load_group_size = self.pes_per_ch // 4
        self.log2_num_pes = int(math.ceil(math.log2(self.num_pes)))
        self.build_dir = build_dir
        self.asset_dir = os.path.join(parent_dir, "assets")

    def generateAll(self):
        if os.path.exists(self.build_dir):
            shutil.rmtree(self.build_dir)

        # self.copyHostCode()
        self.createMakefile()
        self.createHwDefsHeader()
        self.copyKernelHeader()
        self.createKernelCode()
        self.createLinkConfig()
        resource = ResourceEstimator.getDesignResource(self.config, self.fpga)
        logger.info(f"Resource: {resource}")

    def createLinkConfig(self):
        link_file = os.path.join(self.build_dir, "link_config.ini")

        lines = ["[connectivity]\n"]
        lines.append("\n")
        for i in range(self.config.num_ch_B):
            lines.append(f"sp=SpMV.b_{i}:HBM[{i}]\n")

        for i in range(self.config.num_ch_A):
            lines.append(f"sp=SpMV.A_{i}:HBM[{i+self.config.num_ch_B}]\n")

        for i in range(self.config.num_ch_C):
            lines.append(f"sp=SpMV.c_in_{i}:HBM[{i+self.config.num_ch_A+self.config.num_ch_B}]\n")
        
        for i in range(self.config.num_ch_C):
            lines.append(f"sp=SpMV.c_out_{i}:HBM[{i+self.config.num_ch_A+self.config.num_ch_B+self.config.num_ch_C}]\n")

        with open(link_file, 'w') as file:
            for line in lines:
                file.write(line)

    # def copyHostCode(self):
    #     src_dir = os.path.join(self.build_dir, "src")

    #     if not os.path.exists(src_dir):
    #         os.makedirs(src_dir)

    #     file = 'spmv-host.cpp'
    #     cp_src_file = os.path.join(self.asset_dir, file)
    #     cp_dst_file = os.path.join(src_dir, file)
    #     shutil.copy(cp_src_file, cp_dst_file)

    def createMakefile(self):
        src_dir = os.path.join(self.build_dir, "src")

        if not os.path.exists(src_dir):
            os.makedirs(src_dir)
            
        makefile_content = f"""
platform = {self.fpga.platform}
PROJECT_DIR = {project_dir}
WORK_DIR = $(shell pwd)

include $(PROJECT_DIR)/common/common.mk
"""   
        # Write the content to a Makefile
        file_path = os.path.join(self.build_dir, 'Makefile')
        with open(file_path, 'w') as makefile:
            makefile.write(makefile_content)

    def createHwDefsHeader(self):
        src_dir = os.path.join(self.build_dir, "src")
        header_file_wr = os.path.join(src_dir, "hw_defs.h")

        with open(header_file_wr, 'w') as file:
            file.write(f"#define NUM_A_CH {self.config.num_ch_A}\n")
            file.write(f"#define NUM_B_CH {self.config.num_ch_B}\n")
            file.write(f"#define NUM_C_CH {self.config.num_ch_C}\n")
            file.write(f"#define CH_WIDTH {self.config.ch_width}\n")
            file.write(f"#define URAMS_PER_PE {self.config.urams_per_pe}\n")
            if self.config.dense_overlay:
                file.write("#define BUILD_DENSE_OVERLAY\n")
            if self.config.pre_accumulator:
                file.write("#define BUILD_PRE_ACCUMULATOR\n")
            if self.config.row_dist_net:
                file.write("#define BUILD_ROW_DIST_NETWORK\n")
            file.write(f"#define LOAD_GROUP_SIZE {self.load_group_size}\n\n")
            file.write(f"#define LOG_2_NUM_PES {self.log2_num_pes}\n\n")


    def copyKernelHeader(self):
        src_dir = os.path.join(self.build_dir, "src")
        header_file_rd = os.path.join(self.asset_dir, "spmv.h")
        header_file_wr = os.path.join(src_dir, "spmv.h")

        shutil.copy(header_file_rd, header_file_wr)

    def createKernelCode(self):
        src_dir = os.path.join(self.build_dir, "src")

        base_functions_file = os.path.join(self.asset_dir, "base_functions.cpp")
        top_function_file = os.path.join(self.asset_dir, "top_function.cpp")
        spmv_wr_file = os.path.join(src_dir, "spmv.cpp")

        with open(base_functions_file, 'r') as file:
            base_functions_lines = file.readlines()
        
        with open(top_function_file, 'r') as file:
            top_func_lines = file.readlines()

        with open(spmv_wr_file, 'w') as file:
            for line in base_functions_lines:
                file.write(line)
            file.write("\n")
        
        myCB = CrossBarGen(self.num_pes)
        myCB.buildGraph(False)

        cb_streams_lines = self.generateCBstreams(myCB.depth_dict)
        cb_invoke_lines = self.generateCBinvokes(myCB.graph_dict)

        with open(spmv_wr_file, 'a') as file:
            # for line in swb_lines:
            #     file.write(line)

            for line in top_func_lines:
                if line.find("tapa::task()") != -1:
                    file.write("#ifdef BUILD_ROW_DIST_NETWORK")
                    for l in cb_streams_lines:
                        file.write(l)
                    file.write("#endif\n")
                    file.write("\n")
                
                if line.find("#else") != -1:
                    for l in cb_invoke_lines:
                        file.write(l)

                file.write(line)
    
    def generateCBstreams(self, dict):
        lines = ["\n"]
        for stream, props in dict.items():
            lines.append(f"\ttapa::stream<Cnoc_pkt, {props['depth']}> {stream}(\"{stream}\");" + "\n")
        return lines
    
    def generateCBinvokes(self, dict):
        lines = []
        for node, edges in dict.items():
            id, name, level = node.split(".")
            lines.append(f"\t\t.invoke({name}, {edges['incoming'][0]}, {edges['incoming'][1]}, {edges['outgoing'][0]}, {edges['outgoing'][1]})/*{id}*/\n")
        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to Generate TAPA code for SpMV")
    parser.add_argument("output_dir", help="Path to output directory, can be any location, new or existing [WARNING: anything inside this directory will be erased]")
    parser.add_argument("--num-ch-A", default=16, type=int, help="Number of HBM channels to read sparse matrix A")
    parser.add_argument("--num-ch-x", default=1, type=int, help="Number of HBM channels to read dense vector x")
    parser.add_argument("--num-ch-y", default=1, type=int, help="Number of HBM channels to read/write dense vector y_in/y_out")
    parser.add_argument("--ch-width", default=512, type=int, help="Width of HBM channels")
    parser.add_argument("--urams-per-pe", default=2, type=int, help="URAM banks allocated to store output for each PE")
    parser.add_argument("--dense-overlay", action="store_true", help="Build SpMV kernel with support for GeMV")
    parser.add_argument("--pre-accumulator", action="store_true", help="Build SpMV kernel with Pre-Accumulator")
    parser.add_argument("--row-dist-net", action="store_true", help="Build SpMV kernel with Row Distribution Network")
    # Add an argument with choices 'U280' and 'U50'
    parser.add_argument(
        '--device', 
        choices=['U280', 'U50'],  # Specify the valid choices
        required=True,             # Makes this argument mandatory
        help="Choose either 'U280' or 'U50'"
    )
    args = parser.parse_args()
    
    mySpMV = SpMVConfig(
        num_ch_A = args.num_ch_A,
        num_ch_B = args.num_ch_x,
        num_ch_C = args.num_ch_y,
        ch_width = args.ch_width,
        urams_per_pe = args.urams_per_pe,
        dense_overlay = args.dense_overlay,
        pre_accumulator = args.pre_accumulator,
        row_dist_net = args.row_dist_net
    )

    build_dir = os.path.join(args.output_dir, encodeSpMVConfig(mySpMV))
    
    if args.device == 'U280':
        selected_device = U280
    elif args.device == 'U50':
        selected_device = U50

    assert(mySpMV.num_ch_A%(2*mySpMV.num_ch_C) == 0)
    assert(mySpMV.ch_width == 256 or mySpMV.ch_width == 512)

    myGen = SpMVCodeGen(mySpMV, build_dir, selected_device)
    myGen.generateAll()

    logger.info(f"Succesfully Generated Code at {build_dir}")