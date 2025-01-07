SRC_DIR = $(WORK_DIR)/src
COMMON_DIR = $(PROJECT_DIR)/common

CC = g++
CFLAGS = -O2 -fopenmp 
LIBRARIES = -std=c++17 -ltapa -lfrt -lglog -lgflags -lOpenCL -lpthread -lxrt_coreutil -L"$(CONDA_PREFIX)/lib/" -L"$(XILINX_XRT)/lib"
INCLUDES = -I"$(XILINX_HLS)/include" -I"$(COMMON_DIR)/include" -I"$(XILINX_XRT)/include" -I"$(SRC_DIR)"
DEFINES = -DTAPA_BUFFER_SUPPORT -DTAPA_BUFFER_EXPLICIT_RELEASE 


host: $(SRC_DIR)
	$(CC) -o $(WORK_DIR)/spmv-host $(CFLAGS) $(SRC_DIR)/spmv.cpp $(COMMON_DIR)/src/spmv-host.cpp $(COMMON_DIR)/src/spmv-helper.cpp $(DEFINES) $(LIBRARIES) $(INCLUDES)

tapa:
	tapac -o $(WORK_DIR)/spmv.$(platform).hw.xo $(SRC_DIR)/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --read-only-args "A*" \
  --read-only-args "b" \
  --read-only-args "c_in*" \
  --write-only-args "c_out*" \
  --work-dir $(WORK_DIR)/spmv.$(platform).hw.xo.tapa \
  --enable-hbm-binding-adjustment \
  --enable-floorplan \
  --enable-buffer-support \
  --enable-synth-util \
  --max-parallel-synth-jobs 24 \
  --floorplan-output constraint.tcl \
  --clock-period 4.25 \
  --enable-buffer-exprel \
  --min-area-limit 0.55 

tapa_fast:
	tapac -o $(WORK_DIR)/spmv.$(platform).hw.xo $(SRC_DIR)/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --work-dir $(WORK_DIR)/spmv.$(platform).hw.xo.tapa \
  --enable-buffer-support \
  --clock-period 4.25 \
  --enable-buffer-exprel

tapa_synth:
	tapac -o $(WORK_DIR)/spmv.$(platform).hw.xo $(SRC_DIR)/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --work-dir $(WORK_DIR)/spmv.$(platform).hw.xo.tapa \
  --enable-buffer-support \
  --clock-period 4.25 \
  --enable-synth-util \
  --max-parallel-synth-jobs 24 \
  --enable-buffer-exprel

hw:
	$(shell sed -i '/PLACEMENT_STRATEGY="EarlyBlockPlacement"/a ROUTE_STRATEGY="AggressiveExplore"' spmv.$(platform).hw_generate_bitstream.sh)
	$(shell sed -i 's/ROUTE_DESIGN\.ARGS\.DIRECTIVE=\$$STRATEGY/ROUTE_DESIGN.ARGS.DIRECTIVE=\$$ROUTE_STRATEGY/' spmv.$(platform).hw_generate_bitstream.sh)
	sh $(WORK_DIR)/spmv.$(platform).hw_generate_bitstream.sh

clean:
	rm -rf $(WORK_DIR)/spmv-host $(WORK_DIR)/spmv.$(platform).hw.xo