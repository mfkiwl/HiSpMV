import json
from dataclasses import dataclass

@dataclass
class FPGAResource:
    bram: int
    uram: int
    dsp: int
    lut: int
    reg: int

@dataclass
class FPGA:
    available: FPGAResource
    fixed: FPGAResource

@dataclass
class SpMVConfig:
    num_ch_A: int
    num_ch_B: int
    num_ch_C: int
    ch_width: int
    urams_per_pe: int
    dense_overlay: bool
    pre_accumulator: bool
    row_dist_net: bool