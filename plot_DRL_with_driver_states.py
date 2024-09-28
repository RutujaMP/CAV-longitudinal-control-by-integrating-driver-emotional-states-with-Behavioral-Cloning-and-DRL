import os
import re
import numpy as np
import pandas as pd
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
            # Read the file content and decode it
            path = f.read().decode('utf-8').strip()
            print(f"Path from {file_path}: {path}")
            return path
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

# Function to load the referenced .npz file
def load_referenced_npz(file_path):
    # Load the initial .npz file to get the path
    referenced_path = load_npz_path(file_path)
    if referenced_path:
        try:
            # Resolve the relative path
            referenced_path = os.path.abspath(os.path.join(os.path.dirname(file_path), referenced_path))
            print(f"Resolved path: {referenced_path}")
            # Load the referenced .npz file
            with np.load(referenced_path, allow_pickle=True) as data:
                print(f"Contents of {referenced_path}: {data.files}")
                for key in data.files:
                    print(f"Key: {key}, Data: {data[key]}")
                return data
        except Exception as e:
            print(f"Failed to load referenced .npz file {referenced_path}: {e}")
            return None
    else:
        print(f"No valid path found in {file_path}")
        return None

# Function to load results from the directory
def load_results(plots_dir, path_to_key_fn):
    eval_paths = {}
    traj_paths = {}
    for path in plots_dir.glob('*'):
        key = path_to_key_fn(path)
        if (ev_path := path / 'evaluation.csv').exists():
            eval_paths[key] = ev_path
        if (tr_path := path / 'trajectories.npz').exists():
            traj_paths[key] = tr_path
    return eval_paths, traj_paths

# Function to split trajectories
def split_trajectories(df, min_len=0):
    idxs, = np.where(df.step[:-1].values > df.step[1:].values)
    idxs = [0, *(idxs + 1), len(df)]
    return [df.iloc[start: end] for start, end in zip(idxs[:-1], idxs[1:]) if df.step.iloc[end - 1] >= min_len]

# Define the key function for the highway ramp
def highway_ramp_key(path):
    method, highway_inflow = re.match(r'(.+) highway_inflow=(\d+)', path.name).groups()
    return method, int(highway_inflow)

# Path to the highway ramp plots
HighwayRampPlots = Path(RESULTS_DIR) / 'highway_ramp' / 'plots'

# Load results
eval_paths, traj_paths = load_results(HighwayRampPlots, highway_ramp_key)

# Define constants
inflows = range(1800, 2201, 100)
sim_step = 0.5
warmup_steps = 1000
cache = {}

# Load trajectories
for k, path in traj_paths.items():
    if k not in cache:
        referenced_path = load_npz_path(path)
        if referenced_path:
            resolved_path = os.path.abspath(os.path.join(os.path.dirname(path), referenced_path))
            npz = np.load(resolved_path, allow_pickle=True)
            cache[k] = trajectories = split_trajectories(pd.DataFrame({k: npz[k] for k in npz.files}), 6000)
            assert len(trajectories) == 10

# Plotting results
skip_stat_steps = 3000
xs = np.arange(1800, 2201, 100)

# Correctly map the labels
label_mapping = {
    'Baseline': 'Baseline',
    'Ours (DRL)': 'DRL with IDM',
    'Ours (Derived)': 'DRL with driver states'
}

for original_label, plot_label in label_mapping.items():
    means = []
    stds = []
    for flow in xs:
        dfs = [df.loc[df.step >= warmup_steps + skip_stat_steps] for df in cache.get((original_label, flow), [])]
        if dfs:
            outflows = np.array([len(df.id.unique()) - len(df.id[df.step == 6000].unique()) for df in dfs])
            outflows_per_hour = outflows / sim_step / (6000 - warmup_steps - skip_stat_steps) * 3600
            means.append(outflows_per_hour.mean())
            stds.append(outflows_per_hour.std())
        else:
            means.append(np.nan)
            stds.append(np.nan)
    means = np.array(means)
    plt.plot(xs, means, label=plot_label)
    plt.fill_between(xs, means - stds, means + stds, alpha=0.3)

plt.xlabel('Highway Target Inflow (veh/hr)')
plt.ylabel('Outflow (veh/hr)')
plt.legend()
plt.grid()
plt.show()
