from commons import FPGA, FPGAResource, HBM

U280 = FPGA(
    available = FPGAResource(bram=2016, uram=960, dsp=9024, lut=1303680, reg=2607360), # Total Available Resources
    fixed = FPGAResource(bram=195.5, uram=0, dsp=4, lut=107629, reg=136086), # Resources not available to the user
    limit = FPGAResource(bram=0.75, uram=0.5, dsp=0.5, lut=0.62, reg=0.62), # Utilization ratio limit (utilized + fixed)/available <= limit
    hbm=HBM(num_ch=28, ch_width=512, freq=225_000_000), # HBM Specs
    platform='xilinx_u280_gen3x16_xdma_1_202211_1'
)

U50 = FPGA(
    available = FPGAResource(bram=1344, uram=640, dsp=5952, lut=870720, reg=1743360), # Total Available Resources
    fixed = FPGAResource(bram=179.5, uram=4, dsp=4, lut=112760, reg=142875), # Resources not available to the user
    limit = FPGAResource(bram=0.75, uram=0.5, dsp=0.5, lut=0.70, reg=0.70), # Utilization ratio limit (utilized + fixed)/available <= limit
    hbm=HBM(num_ch=28, ch_width=512, freq=225_000_000), # HBM Specs
    platform='xilinx_u50_gen3x16_xdma_5_202210_1'
)