import os
import subprocess
import csv
import argparse

# Define the configuration and bitstream for each FPGA
FPGA_CONFIGS_AND_BITSTREAMS = {
    'U280': {
        "config": "u280_best_configs.csv",
        "bitstream": "SpMV_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin",
    },
    'U50': {
        "config": "u50_best_configs.csv",
        "bitstream": "SpMV_xilinx_u50_gen3x16_xdma_5_202210_1.xclbin",
    }
}

# Base matrix directory
MATRIX_BASE_DIR = "../../matrices"

def execute_command(command, log_file):
    """
    Executes a single shell command, waits for it to finish, 
    and writes the output to the provided log file.
    """
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        
        # Write the command output to the log file
        with open(log_file, "a") as log:
            log.write(result.stdout)
            log.write(result.stderr)

        # print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {command}")
        print(f"Error: {e}")
        with open(log_file, "a") as log:
            log.write(f"Error: {e}\n")
            log.write(f"Stdout: {e.stdout}\n")
            log.write(f"Stderr: {e.stderr}\n")
        return

def process_config_entry(config, bitstream, device_id, exec_ms, power_s, matrix_files):
    """
    Process each matrix for a specific config and bitstream. First, execute the initial setup commands, 
    then execute the spmv-host command for each matrix in that config.
    """
    # Navigate to the config directory
    os.chdir(config)

    # Run setup commands only once per config
    setup_commands = [
        "make clean host",
        "rm -rf logs",
        "mkdir logs"
    ]
    
    # Execute setup commands
    for command in setup_commands:
        execute_command(command, "setup.log")

    # Process each matrix file in the config
    for filename in matrix_files:
        matrix_name = os.path.splitext(filename)[0]
        log_file = f"logs/{matrix_name}.log"

        # Execute spmv-host command and capture the output
        if 'u50' in bitstream:
            # do not specify device id for u50 to run with tapa invoke, xrt has some bug with u50
            spmv_command = (
            f"./spmv-host {MATRIX_BASE_DIR}/{matrix_name}/{filename} --bitstream=\"{bitstream}\"  --exec_ms={exec_ms} --power_s={power_s}"
            )
        else:
            spmv_command = (
                f"./spmv-host {MATRIX_BASE_DIR}/{matrix_name}/{filename} --bitstream=\"{bitstream}\" --device={device_id} --exec_ms={exec_ms} --power_s={power_s}"
            )
        execute_command(spmv_command, log_file)

    # Return to the original directory after processing the matrices
    os.chdir("..")


def process_csv_file(config_file, bitstream, device_id, exec_ms, power_s):
    """
    Process each entry in a config CSV file and organize the matrix files by config.
    """
    config_dict = {}

    # Read the config CSV file and convert it into a dictionary
    with open(config_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header line

        # Populate the dictionary with config names as keys and list of matrix files as values
        for filename, config, cycle in reader:
            if config not in config_dict:
                config_dict[config] = []
            config_dict[config].append(filename)

    # Process each config in the dictionary
    for config, matrix_files in config_dict.items():
        print(f"Processing config: {config} with bitstream: {bitstream} on device {device_id}")
        process_config_entry(config, bitstream, device_id, exec_ms, power_s, matrix_files)

def main():
    # Parse the FPGA type, device ID, and other arguments
    parser = argparse.ArgumentParser(description="Process config files and bitstreams for specified FPGA type.")
    parser.add_argument('fpga', choices=['U280', 'U50'], help='FPGA type: U280 or U50')
    parser.add_argument('device', type=int, help="Device ID to use (must be specified)")
    args = parser.parse_args()

    fpga_type = args.fpga
    device_id = args.device
    exec_ms = 1000 if fpga_type == 'U280' else 20
    power_s = 60 if fpga_type == 'U280' else 0 

    # Ensure the FPGA type is valid
    if fpga_type not in FPGA_CONFIGS_AND_BITSTREAMS:
        print(f"Error: Invalid FPGA type '{fpga_type}'. Please choose either 'U280' or 'U50'.")
        return

    # Get config file and bitstream for the selected FPGA
    config_file = FPGA_CONFIGS_AND_BITSTREAMS[fpga_type]["config"]
    bitstream = FPGA_CONFIGS_AND_BITSTREAMS[fpga_type]["bitstream"]
    
    print(f"Processing config file: {config_file} with bitstream: {bitstream} on device {device_id}")
    process_csv_file(config_file, bitstream, device_id, exec_ms, power_s)

if __name__ == "__main__":
    main()
