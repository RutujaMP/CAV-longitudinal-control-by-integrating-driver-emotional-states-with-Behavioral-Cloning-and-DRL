import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set the environment variables (Update these to match your local setup if needed)
CODE_DIR = os.getenv('F', r'C:/automatic_vehicular_control/automatic_vehicular_control')
RESULTS_DIR = os.getenv('R', r'C:/automatic_vehicular_control/results')

# Append the code directory to the system path
os.sys.path.append(CODE_DIR)

# Function to read path from initial .npz file
def load_npz_path(file_path):
    try:
        with open(file_path, 'rb') as f:
            path = f.read().decode('utf-8').strip()
            return path
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

# Function to load the referenced .npz file
def load_referenced_npz(file_path):
    referenced_path = load_npz_path(file_path)
    if referenced_path:
        try:
            resolved_path = os.path.abspath(os.path.join(os.path.dirname(file_path), referenced_path))
            if not os.path.exists(resolved_path):
                print(f"File does not exist at path: {resolved_path}")
                return None
            with np.load(resolved_path, allow_pickle=True) as data:
                extracted_data = {key: data[key] for key in data.files}
                return extracted_data
        except Exception as e:
            print(f"Failed to load referenced .npz file {resolved_path}: {e}")
            return None
    else:
        print(f"No valid path found in {file_path}")
        return None

# Function to load results from the directory
def load_results(plots_dir, path_to_key_fn):
    traj_paths = {}
    for path in plots_dir.glob('*'):
        key = path_to_key_fn(path)
        if (tr_path := path / 'trajectories.npz').exists():
            traj_paths[key] = tr_path
    return traj_paths

# Define the key function for the highway ramp
def highway_ramp_key(path):
    method, highway_inflow = re.match(r'(.+) highway_inflow=(\d+)', path.name).groups()
    return method, int(highway_inflow)

# Path to the highway ramp plots
HighwayRampPlots = Path(RESULTS_DIR) / 'highway_ramp' / 'plots'

# Load results
traj_paths = load_results(HighwayRampPlots, highway_ramp_key)

# Plot human speed data for DRL using IDM
plt.figure(figsize=(10, 5))
for k, path in traj_paths.items():
    if k[0] == 'Ours (DRL)':  # DRL using IDM
        referenced_data = load_referenced_npz(path)
        if referenced_data is not None:
            try:
                # Extract relevant data
                step_data = referenced_data["step"]
                speed_data_array = referenced_data["speed"]
                vehicle_type = referenced_data["type"]

                # Check if 'human' is in vehicle_type and create a boolean mask
                human_mask = (vehicle_type == 'human')
                if human_mask.sum() == 0:
                    print(f"No 'human' vehicles found in {path}")
                    continue

                # Filter human speeds
                speed_human = speed_data_array[human_mask]

                # Plot human speed data for DRL using IDM
                plt.plot(step_data[human_mask], speed_human, label='DRL using IDM - Speed Human')
            except Exception as e:
                print(f"Failed to process data from {path}: {e}")
        else:
            print(f"Failed to load referenced data from {path}")

# Final plot adjustments for DRL using IDM
plt.xlabel('Iterations (Step)')
plt.ylabel('Speed')
plt.legend(loc='upper right')
plt.grid()
plt.title('DRL using IDM: Human Speed vs Iterations')
plt.tight_layout()
plt.show()

# Plot human speed data for DRL with driver states
plt.figure(figsize=(10, 5))
for k, path in traj_paths.items():
    if k[0] == 'Ours (Derived)':  # DRL with driver states
        referenced_data = load_referenced_npz(path)
        if referenced_data is not None:
            try:
                # Extract relevant data
                step_data = referenced_data["step"]
                speed_data_array = referenced_data["speed"]
                vehicle_type = referenced_data["type"]

                # Check if 'human' is in vehicle_type and create a boolean mask
                human_mask = (vehicle_type == 'human')
                if human_mask.sum() == 0:
                    print(f"No 'human' vehicles found in {path}")
                    continue

                # Filter human speeds
                speed_human = speed_data_array[human_mask]

                # Plot human speed data for DRL with driver states
                plt.plot(step_data[human_mask], speed_human, label='DRL with driver states - Speed Human')
            except Exception as e:
                print(f"Failed to process data from {path}: {e}")
        else:
            print(f"Failed to load referenced data from {path}")

# Final plot adjustments for DRL with driver states
plt.xlabel('Iterations (Step)')
plt.ylabel('Speed')
plt.legend(loc='upper right')
plt.grid()
plt.title('DRL with driver states: Human Speed vs Iterations')
plt.tight_layout()
plt.show()
