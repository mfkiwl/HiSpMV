from commons import FPGA, FPGAResource, HBM

U280 = FPGA(
    available = FPGAResource(bram=2016, uram=960, dsp=9024, lut=1303680, reg=2607360), # Total Available Resources
    fixed = FPGAResource(bram=195.5, uram=0, dsp=4, lut=107629, reg=136086), # Resources not available to the user
    limit = FPGAResource(bram=0.75, uram=0.75, dsp=0.75, lut=0.62, reg=0.62), # Utilization ratio limit (utilized + fixed)/available <= limit
    # HBM has 256b x 32 Channels @ 450MHZ
    # Assuming that we can acheive a kernel frequency of 225MHz, we have 32 x 512b @ 225MHz ~= 460GBPS
    # But realistically HLS designs can only acheive good frequency if the HBM channels are below 28
    hbm=HBM(num_ch=28, ch_width=512, freq=225_000_000), # HBM Specs 
    platform='xilinx_u280_gen3x16_xdma_1_202211_1',
    series='Ultrascale+'
)

U50 = FPGA(
    available = FPGAResource(bram=1344, uram=640, dsp=5952, lut=870720, reg=1743360), # Total Available Resources
    fixed = FPGAResource(bram=179.5, uram=4, dsp=4, lut=112760, reg=142875), # Resources not available to the user
    limit = FPGAResource(bram=0.75, uram=0.75, dsp=0.75, lut=0.70, reg=0.70), # Utilization ratio limit (utilized + fixed)/available <= limit
    hbm=HBM(num_ch=28, ch_width=512, freq=225_000_000), # HBM Specs
    platform='xilinx_u50_gen3x16_xdma_5_202210_1',
    series='Ultrascale+'
)

# Prediction for Versal HBM sereies
# References 
# https://docs.amd.com/v/u/en-US/ds950-versal-overview Table 16 Column VH1782
# https://www.amd.com/en/products/accelerators/alveo/v80/a-v80-p64g-pq-g.html
V80 = FPGA(
    # Key difference in DSP58 in Versal v/s DSP48 in Ultrascale+
    # Can perform fpacc in single cycle in single DSP
    # Can also perform fpmul in single DSP
    # Refenece https://0x04.net/~mwk/xidocs/am/am004-versal-dsp-engine.pdf Figure 33: Floating Point Multiplier and Adder (DSPFP32 Mode)
    available = FPGAResource(bram=3741, uram=1925, dsp=10848,  lut=2574208, reg=5148416), 
    fixed = FPGAResource(bram=195.5, uram=0, dsp=4, lut=107629, reg=136086), # Assume same as U280
    limit = FPGAResource(bram=0.75, uram=0.75, dsp=0.75, lut=0.70, reg=0.70),
    # HBM has 128b x 16 channels @ 3.2GHz 
    # Assuming Max Kernel Freuency of 400MHz we get 256b x 64 channels @ 400MHz ~= 819.2GBPS
    # Assuming we can use all the ports
    hbm=HBM(num_ch=64, ch_width=256, freq=400_000_000),
    # HLS platform is not available currently
    platform='xilinx_v80_platform_placeholder',
    series='Versal'
)