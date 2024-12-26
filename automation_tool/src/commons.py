import json
from dataclasses import dataclass
from datetime import datetime
import logging

# Generate a filename with the current date and time
filename = datetime.now().strftime("%Y%m%d%H%M%S.log")

# Define ANSI color code
GREY = '\033[90m'  # Green
YELLOW = '\033[93m' # Yellow
RESET = '\033[0m'   # Reset to default

# Configure the logger
logging.basicConfig(
    format=f'{GREY}%(asctime)s {YELLOW}[%(levelname)s] {RESET} %(message)s',
    datefmt='%Y%m%d:%H%M%S',                        # Timestamp format
    level=logging.INFO                              # Set the logging level
)

@dataclass
class FPGAResource:
    bram: float
    uram: float
    dsp: float
    lut: float
    reg: float

@dataclass
class HBM:
    num_ch: int
    ch_width: int
    freq: int

@dataclass
class FPGA:
    available: FPGAResource
    fixed: FPGAResource
    limit: FPGAResource
    hbm: HBM
    platform: str

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

def encodeSpMVConfig(spmv_config: SpMVConfig) -> str:
    """
    Encodes SpMV configuration into a specific string format.
    """
    base_string = "SpMV"
    prefixes = []

    # Add prefixes based on flags
    if spmv_config.dense_overlay:
        prefixes.append("Dense")
    if spmv_config.pre_accumulator:
        prefixes.append("PA")
    if spmv_config.row_dist_net:
        prefixes.append("HI")

    # Construct the prefix string
    prefix_string = "-".join(prefixes)
    # Construct the final string with postfix
    postfix_string = f"{spmv_config.num_ch_A}-{spmv_config.num_ch_B}-{spmv_config.num_ch_C}"

    # Combine prefix and base string with the postfix
    if prefix_string == "":
        return f"{base_string}-{postfix_string}"
    return f"{prefix_string}-{base_string}-{postfix_string}"
