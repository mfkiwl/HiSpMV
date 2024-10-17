from commons import FPGA, FPGAResource

U280 = FPGA(
    available = FPGAResource(bram=2016, uram=960, dsp=9024, lut=1303680, reg=2607360),
    fixed = FPGAResource(bram=195.5, uram=0, dsp=4, lut=107629, reg=136086)
)