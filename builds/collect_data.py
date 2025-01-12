import os
import glob
import re
import csv
import argparse

# Define the regex patterns for the required metrics
METRIC_PATTERNS = {
    "Pre-Processing Time": r"Pre-processing Time: ([\d\.]+) secs",
    "CPU Time": r"CPU TIME: ([\d\.]+) ms",
    "CPU GFLOPS": r"CPU GFLOPS: ([\d\.]+)",
    "Matrix A Length": r"Matrix A Length: (\d+)",
    "Approx. Clock Cycles": r"Approx\. Clock Cycles: (\d+)",
    "Repeat Time": r"Using Repeat Time: (\d+)",
    "Num Samples": r"Using Num samples: (\d+)",
    "Average Power": r"Average Power: ([\d\.]+) Watts",
    "Max Power": r"Max Power: ([\d\.]+) Watts",
    "Total Kernel Runtime": r"Total Kernel Runtime: ([\d\.]+)ms",
    "FPGA Time": r"FPGA TIME: ([\d\.]+)us",
    "FPGA GFLOPS": r"FPGA GFLOPS: ([\d\.]+)"
}

SAMPLE_PATTERN = r"sample: ([\d\.]+)"

def process_logs(fpga):
    # File output names based on FPGA target
    output_metrics = f"{fpga}_metrics.csv"
    output_samples = f"{fpga}_power_samples.csv"

    # Collect all logs
    log_files = glob.glob("*/logs/*.log")

    # Store extracted data
    metrics_data = []
    samples_dict = {}

    for log_file in log_files:
        # Initialize log entry
        entry = {"Log File": os.path.basename(log_file)}
        samples = []

        # Read the log file
        with open(log_file, "r") as f:
            content = f.read()

        # Extract metrics
        for metric, pattern in METRIC_PATTERNS.items():
            match = re.search(pattern, content)
            entry[metric] = match.group(1) if match else None

        # Extract samples
        sample_matches = re.findall(SAMPLE_PATTERN, content)
        if sample_matches:
            samples_dict[os.path.basename(log_file)] = list(map(float, sample_matches))

        # Append metrics to data
        metrics_data.append(entry)

    # Write metrics to CSV
    with open(output_metrics, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Log File"] + list(METRIC_PATTERNS.keys()))
        writer.writeheader()
        writer.writerows(metrics_data)

    print(f"Metrics saved to {output_metrics}")

    # Align power samples and write to CSV (only if samples are available)
    if samples_dict:
        max_samples = max(len(samples) for samples in samples_dict.values())
        with open(output_samples, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["Sample Index"] + list(samples_dict.keys()))
            # Write rows
            for i in range(max_samples):
                row = [i + 1]
                for log_file in samples_dict.keys():
                    row.append(samples_dict[log_file][i] if i < len(samples_dict[log_file]) else None)
                writer.writerow(row)
        print(f"Power samples saved to {output_samples}")
    else:
        print("No power samples found. Skipping power samples CSV creation.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process FPGA metrics and power samples.")
    parser.add_argument(
        "fpga", choices=["U50", "U280"], help="Specify the target FPGA: 'U50' or 'U280'."
    )
    args = parser.parse_args()

    # Process logs based on the provided FPGA
    process_logs(args.fpga)

if __name__ == "__main__":
    main()
