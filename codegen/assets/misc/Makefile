CC = g++
CFLAGS = -O2 -fopenmp -g
LIBRARIES = -ltapa -lfrt -lglog -lgflags -lOpenCL -lpthread -I"$(XILINX_HLS)/include" -std=c++17
DEFINES = -DTAPA_BUFFER_SUPPORT -DTAPA_BUFFER_EXPLICIT_RELEASE
platform = xilinx_u280_gen3x16_xdma_1_202211_1

spmv: src/
	$(CC) -o spmv $(CFLAGS) src/*.cpp  $(DEFINES) $(LIBRARIES)

tapa:
	tapac -o spmv.$(platform).hw.xo src/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --read-only-args "A*" \
  --read-only-args "b" \
  --read-only-args "c_in*" \
  --write-only-args "c_out*" \
  --work-dir spmv.$(platform).hw.xo.tapa \
  --enable-hbm-binding-adjustment \
  --enable-floorplan \
  --enable-buffer-support \
  --enable-synth-util \
  --max-parallel-synth-jobs 24 \
  --floorplan-output constraint.tcl \
  --clock-period 4.25 \
  --enable-buffer-exprel 

tapa_fast:
	tapac -o spmv.$(platform).hw.xo src/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --work-dir spmv.$(platform).hw.xo.tapa \
  --enable-buffer-support \
  --clock-period 4.25 \
  --enable-buffer-exprel

tapa_synth:
	tapac -o spmv.$(platform).hw.xo src/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --work-dir spmv.$(platform).hw.xo.tapa \
  --enable-buffer-support \
  --clock-period 4.25 \
  --enable-synth-util \
  --max-parallel-synth-jobs 24 \
  --enable-buffer-exprel

hw:
	sh spmv.$(platform).hw_generate_bitstream.sh

clean:
	rm -rf spmv