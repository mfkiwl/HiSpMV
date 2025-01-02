from commons import SpMVConfig, encodeSpMVConfig
from spmvcodegen import SpMVCodeGen
from fpgas import U280, U50
from dse import DSE

import argparse
import os
import sys
import logging
import csv

logger = logging.getLogger(__name__)

def list_mtx_files(directory):
    mtx_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".mtx"):
                mtx_files.append(os.path.join(root, filename))
    return mtx_files

def save_to_csv(file_path: str, config: str, output_csv: str):
    # Extract the basename from the file path
    filename = os.path.basename(file_path)
    
    # Prepare data for CSV
    data = {
        "filename": filename,
        "config": config
    }

    # Write to CSV
    with open(output_csv, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        
        # Write header only if the file is new
        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow(data)

def main(build_dir, fpga, matrices, dense_overlay):
    if len(matrices) == 0:
        best_config = DSE.getSingleBestConfig(fpga, dense_overlay)
        config_str = encodeSpMVConfig(best_config)
        build_path = os.path.join(build_dir, config_str)
        myGen = SpMVCodeGen(best_config, build_path, fpga)
        myGen.generateAll()
        sys.exit(0)

    built_configs = set()
    csv_file = os.path.join(build_dir, "best-configs.csv")

    count = 1
    for mtx_file in matrices:
        logger.info(f"Processing Matrix {count} out of {len(matrices)}")
        best_config = DSE.getBestConfig(mtx_file, fpga, dense_overlay)
        config_str = encodeSpMVConfig(best_config)
        save_to_csv(mtx_file, config_str, csv_file)
        count += 1
        if config_str not in built_configs:
            build_path = os.path.join(build_dir, config_str)
            myGen = SpMVCodeGen(best_config, build_path, fpga)
            myGen.generateAll()
            built_configs.add(config_str)
        
    logger.info(f"Succesfully Generated All configs at: {build_dir}")
    logger.info(f"Matrix and Config mapping stored at: {csv_file}")



if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Build process configuration.")

    # Positional argument for the build directory
    parser.add_argument('build_dir', type=str, help="The path to the build directory.")

    # Optional argument for the device (U50 or U280)
    parser.add_argument('--device', choices=['U50', 'U280'], required=True, help="Specify the device, either 'U50' or 'U280'.")

    # Optional argument for matrices (single file or directory)
    parser.add_argument('--matrices', type=str, help="Path to a matrix file or a directory containing matrices.")

    parser.add_argument("--dense-overlay", action="store_true", help="Build SpMV kernel with support for GeMV")

    # Parse the arguments
    args = parser.parse_args()

    if args.device == 'U280':
        selected_device = U280
    elif args.device == 'U50':
        selected_device = U50
    else:
        logger.error("Invalid device. Choose either 'U50' or 'U280'.")
        sys.exit(1)
    
    logger.info(f"Device selected: {args.device}")

    build_dir = args.build_dir
    matrices = args.matrices

    if not os.path.exists(build_dir):
        logger.warning(f"Build directory '{build_dir}' does not exist. Creating it now...")
        try:
            os.makedirs(build_dir)
            logger.info(f"Build directory '{build_dir}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating build directory: {e}")
            sys.exit(1)

    mtx_files = []
    if matrices is not None:
        if not os.path.exists(matrices):
            logger.error(f"The path '{matrices}' does not exist.")
            sys.exit(1)
        
        if os.path.isdir(matrices):
            logger.info(f"Processing matrices from directory: {matrices}")
            mtx_files = list_mtx_files(matrices)
            logger.info(f"Found {len(mtx_files)} matrices")

        elif os.path.isfile(matrices):
            logger.info(f"Processing single matrix file: {matrices}")
            mtx_files.append(matrices)

        else:
            logger.error(f"'{matrices}' is neither a file nor a directory.")
            sys.exit(1)

    # Call main with parsed arguments
    main(build_dir, selected_device, mtx_files, args.dense_overlay)