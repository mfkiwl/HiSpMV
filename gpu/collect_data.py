import os
import csv
import numpy as np

def read_log_file(log_file_path):
    """Reads the content of the log file and returns the relevant performance info including time and rp_time."""
    with open(log_file_path, 'r') as f:
        content = f.readlines()

    # Extract GFLOPs, Time Taken, and rp_time
    performance_info = {}
    for line in content:
        if line.startswith("GFLOPs"):
            performance_info["GFLOPs"] = float(line.split(":")[1].strip())
        if "Time taken for SpMV" in line:
            # Extract time in microseconds (us)
            time_taken_str = line.split(":")[1].strip()
            performance_info["Time Taken"] = float(time_taken_str.split()[0])  # Assuming time_taken is in microseconds
        if "using rp time" in line:
            # Extract rp_time (adjust unit if necessary)
            rp_time_str = line.split(":")[1].strip()
            performance_info["rp_time"] = float(rp_time_str)  # Assuming rp_time is already in the desired unit
    
    return performance_info

def read_power_log_file(power_log_path):
    """Reads the power log file and returns the power values."""
    with open(power_log_path, 'r') as f:
        power_values = [float(line.strip()) for line in f.readlines()]
    
    return power_values

def process_logs(logs_folder, power_logs_folder):
    """Processes both logs and power logs to generate CSVs."""
    summary_data = []
    detailed_data = []
    filenames = []

    # Iterate through the log files in the logs folder
    for log_file in os.listdir(logs_folder):
        if log_file.endswith(".log"):
            # Construct file paths
            log_file_path = os.path.join(logs_folder, log_file)
            power_log_file_path = os.path.join(power_logs_folder, log_file)
            
            # Read the log and power files
            performance_info = read_log_file(log_file_path)
            power_values = read_power_log_file(power_log_file_path)
            
            # Calculate average and max power
            avg_power = np.mean(power_values)
            max_power = np.max(power_values)
            time_taken = performance_info.get("Time Taken", 0)  # Default to 0 if not found
            rp_time = performance_info.get("rp_time", 0)  # Default to 0 if not found
            
            # Store the summary data
            summary_data.append([log_file, performance_info["GFLOPs"], avg_power, max_power, time_taken, rp_time])
            
            # Store the detailed data (just power values)
            detailed_data.append(power_values)
            filenames.append(log_file)
    
    # Ensure all files have the same number of power samples
    max_length = max(len(power_values) for power_values in detailed_data)
    for i in range(len(detailed_data)):
        # Pad with None (or another placeholder) if the list is shorter
        while len(detailed_data[i]) < max_length:
            detailed_data[i].append(None)

    # Write the summary CSV file (renamed to "v100_metrics.csv")
    with open("v100_metrics.csv", mode="w", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(["File", "GFLOPs", "Average Power", "Max Power", "Time Taken", "RP Time"])
        for row in summary_data:
            summary_writer.writerow(row)
    
    # Write the detailed CSV file (renamed to "v100_power_samples.csv")
    with open("v100_power_samples.csv", mode="w", newline="") as detailed_file:
        detailed_writer = csv.writer(detailed_file)
        detailed_writer.writerow(filenames)
        
        # Transpose the detailed data so each row corresponds to one power sample across all files
        for i in range(max_length):  # Use the maximum length
            row = [detailed_data[j][i] for j in range(len(detailed_data))]
            detailed_writer.writerow(row)

# Specify the paths to your logs and power logs directories
logs_folder = "logs"
power_logs_folder = "power_logs"

# Process the logs and power logs
process_logs(logs_folder, power_logs_folder)
