import os
import glob
import re
import csv

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

# Initialize CSV file paths
OUTPUT_METRICS = "u280_metrics.csv"
OUTPUT_SAMPLES = "u280_power_samples.csv"
OUTPUT_INCOMPLETE = "u50_metrics.csv"

# Collect all logs
log_files = glob.glob("*/logs/*.log")

# Store extracted data
metrics_data = []
samples_data = []
incomplete_data = []

for log_file in log_files:
    # Initialize log entry
    entry = {"Log File": os.path.basename(log_file)}
    samples = []

    # Read the log file
    with open(log_file, "r") as f:
        content = f.read()

    # Extract metrics
    complete = True
    for metric, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, content)
        if match:
            entry[metric] = match.group(1)
        else:
            entry[metric] = None
            complete = False

    # Extract samples
    sample_matches = re.findall(SAMPLE_PATTERN, content)
    if sample_matches:
        samples.extend(sample_matches)

    # Handle incomplete and complete logs
    if complete:
        metrics_data.append(entry)
    else:
        incomplete_data.append(entry)

    # Add samples if available
    if samples:
        samples_data.append({"Log File": os.path.basename(log_file), "Samples": ",".join(samples)})

# Write metrics to CSV
with open(OUTPUT_METRICS, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Log File"] + list(METRIC_PATTERNS.keys()))
    writer.writeheader()
    writer.writerows(metrics_data)

# Write samples to CSV
with open(OUTPUT_SAMPLES, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Log File", "Samples"])
    writer.writeheader()
    writer.writerows(samples_data)

# Write incomplete metrics to CSV
with open(OUTPUT_INCOMPLETE, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Log File"] + list(METRIC_PATTERNS.keys()))
    writer.writeheader()
    writer.writerows(incomplete_data)

print(f"Metrics saved to {OUTPUT_METRICS}")
print(f"Samples saved to {OUTPUT_SAMPLES}")
print(f"Incomplete metrics saved to {OUTPUT_INCOMPLETE}")
