import os
import shutil
import math
from commons import SpMVConfig
from crossbar import CrossBarGen

class SpMVCodeGen:
    def __init__(self, spmvConfig, build_dir, parent_dir):
        self.config = spmvConfig
        self.pes_per_ch = self.config.ch_width // 64
        self.num_pes = self.config.num_ch_A * self.pes_per_ch
        self.b_part = (self.config.num_ch_B * self.config.ch_width) // 32 // 2 #BRAM36K 2Ports
        self.b_window = min(self.b_part * 1024, 1 << 14)
        self.w_window = self.b_window // self.b_part // 2
        self.load_group_size = self.pes_per_ch // 2
        self.log2_num_pes = int(math.ceil(math.log2(self.num_pes)))
        self.build_dir = build_dir
        self.parent_dir = parent_dir
        self.temp_dir = os.path.join(parent_dir, "temp")
        self.asset_dir = os.path.join(parent_dir, "assets")

    def generateAll(self):
        self.copyHostCode()
        self.copyMakefile()
        self.writeHwDefsHeader()
        self.writeKernelHeader()
        self.writeKernelCode()
        self.writeLinkConfig()

        if os.path.exists(self.build_dir):
            shutil.rmtree(self.build_dir)
            
        shutil.move(self.temp_dir, self.build_dir)

    def writeLinkConfig(self):
        link_file = os.path.join(self.temp_dir, "link_config.ini")

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

    def copyHostCode(self):
        src_dir = os.path.join(self.temp_dir, "src")

        if not os.path.exists(src_dir):
            os.makedirs(src_dir)

        host_dir = os.path.join(self.asset_dir, "host")
        host_files = os.listdir(host_dir)

        for file in host_files:
            cp_src_file = os.path.join(host_dir, file)
            cp_dst_file = os.path.join(src_dir, file)
            shutil.copy(cp_src_file, cp_dst_file)

    def copyMakefile(self):
        misc_dir = os.path.join(self.asset_dir, "misc")

        # for file in host_files:
        cp_src_file = os.path.join(misc_dir, "Makefile")
        cp_dst_file = os.path.join(self.temp_dir, "Makefile")
        # print(cp_dst_file)
        shutil.copy(cp_src_file, cp_dst_file)

    def writeHwDefsHeader(self):
        src_dir = os.path.join(self.temp_dir, "src")
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


    def writeKernelHeader(self):
        src_dir = os.path.join(self.temp_dir, "src")
        kernel_dir = os.path.join(self.asset_dir, "kernel")

        header_file_rd = os.path.join(kernel_dir, "spmv.h")
        header_file_wr = os.path.join(src_dir, "spmv.h")

        with open(header_file_rd, 'r') as file:
            lines = file.readlines()

        with open(header_file_wr, 'w') as file:
            for line in lines:
                if line.startswith("#define II_DIST"):
                    file.write(f"#define PES_PER_CH {self.pes_per_ch}\n")
                    file.write(f"#define NUM_PES {self.num_pes}\n")
                    file.write(f"#define NUM_PES_HALF {self.num_pes//2}\n")
                    file.write(f"#define LOG_2_NUM_PES {self.log2_num_pes}\n\n")

                    file.write(f"#define B_PART {self.b_part}\n")
                    file.write(f"#define B_WINDOW {self.b_window}\n")
                    file.write(f"#define W_WINDOW {self.w_window}\n")
                    file.write(f"#define LOAD_GROUP_SIZE {self.load_group_size}\n\n")
                file.write(line)

    def writeKernelCode(self):
        src_dir = os.path.join(self.temp_dir, "src")
        kernel_dir = os.path.join(self.asset_dir, "kernel")

        base_functions_file = os.path.join(kernel_dir, "base_functions.cpp")
        top_function_file = os.path.join(kernel_dir, "top_function.cpp")
        spmv_wr_file = os.path.join(src_dir, "spmv.cpp")

        with open(base_functions_file, 'r') as file:
            base_functions_lines = file.readlines()
        
        with open(top_function_file, 'r') as file:
            top_func_lines = file.readlines()

        with open(spmv_wr_file, 'w') as file:
            for line in base_functions_lines:
                file.write(line)
            file.write("\n")
        # swb_lines = self.generateSWB()
        
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


    def generateTA(self):
        lines = ["\n"]
        for i in range(0, self.num_pes, 2):
            lines.append(f"void TreeAdder_{i}(tapa::istream<uint16_v2>& c_row, tapa::istream<float_v2>& c_val, " + "\n")
            lines.append(f"\ttapa::istream<flags_pkt>& c_flags, tapa::ostreams<Cnoc_pkt,2>& c_out)" + "{\n")
            lines.append(f"\tTreeAdder<{i}>(c_row, c_val, c_flags, c_out);" + "\n")
            lines.append("}\n\n")
        return lines
        

    def generateSWB(self):
        lines = ["\n"]
        for i in range(1, self.log2_num_pes-1):
            for j in range(2):
                lines.append(f"void SWB{j}_{i}(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,\n")
                lines.append("\ttapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {\n")
                lines.append(f"\tSWB{j}<{i}>(c_in0, c_in1, c_out0, c_out1);\n")
                lines.append("}\n\n")
        
        return lines