import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_smoothed_rewards(reward_history, output_dir, num_tasks, window_length=51):
    """
    Plot smoothed normalized and original rewards over time.

    Parameters:
    - reward_history: Dictionary containing 'r1', 'r2', and 'time'.
    - output_dir: Directory to save the reward plots.
    - num_tasks: Number of tasks in the workload.
    - window_length: Window size for Savitzky-Golay filter.
    """

    os.makedirs(output_dir, exist_ok=True)

    time_points = np.array(reward_history['time'])
    r1_values = np.array(reward_history['r1'])
    r2_values = np.array(reward_history['r2'])
    r_combined = (r1_values + r2_values) / 2

    if len(time_points) > window_length:
        r1_smooth = savgol_filter(r1_values, window_length, 3)
        r2_smooth = savgol_filter(r2_values, window_length, 3)
        r_combined_smooth = savgol_filter(r_combined, window_length, 3)
    else:
        r1_smooth, r2_smooth, r_combined_smooth = r1_values, r2_values, r_combined

    plt.figure(figsize=(12, 6))
    r1_norm = r1_smooth / max(abs(r1_smooth.max()), abs(r1_smooth.min()))
    r2_norm = r2_smooth / max(abs(r2_smooth.max()), abs(r2_smooth.min()))
    r_combined_norm = r_combined_smooth / max(abs(r_combined_smooth.max()), abs(r_combined_smooth.min()))

    plt.plot(time_points, r1_norm, 'g-', label='R1 (Stage 1)', alpha=0.8)
    plt.plot(time_points, r2_norm, 'b-', label='R2 (Stage 2)', alpha=0.8)
    plt.plot(time_points, r_combined_norm, 'r-', label='R (Combined)', alpha=0.8)
    plt.title(f'Normalized Smoothed Rewards over Time ({num_tasks} tasks)')
    plt.xlabel('Time')
    plt.ylabel('Normalized Reward Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/normalized_rewards_plot_{num_tasks}_tasks.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time_points, r1_smooth, 'g-', label='R1 (Stage 1)', alpha=0.8)
    plt.plot(time_points, r2_smooth, 'b-', label='R2 (Stage 2)', alpha=0.8)
    plt.plot(time_points, r_combined_smooth, 'r-', label='R (Combined)', alpha=0.8)
    plt.title(f'Original Smoothed Rewards over Time ({num_tasks} tasks)')
    plt.xlabel('Time')
    plt.ylabel('Reward Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/original_rewards_plot_{num_tasks}_tasks.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_rewards_data(reward_history, output_dir, num_tasks):
    """
    Save normalized and original rewards data to JSON files.

    Parameters:
    - reward_history: Dictionary containing 'r1', 'r2', and 'time'.
    - output_dir: Directory to save the rewards data.
    - num_tasks: Number of tasks in the workload.
    """
    os.makedirs(output_dir, exist_ok=True)

    r1_values = np.array(reward_history['r1'])
    r2_values = np.array(reward_history['r2'])
    r_combined = (r1_values + r2_values) / 2

    r1_norm = r1_values / max(abs(r1_values.max()), abs(r1_values.min()))
    r2_norm = r2_values / max(abs(r2_values.max()), abs(r2_values.min()))
    r_combined_norm = r_combined / max(abs(r_combined.max()), abs(r_combined.min()))

    rewards_data = {
        'normalized': {
            'R1': r1_norm.tolist(),
            'R2': r2_norm.tolist(),
            'R': r_combined_norm.tolist(),
            'time': reward_history['time']
        },
        'original': {
            'R1': r1_values.tolist(),
            'R2': r2_values.tolist(),
            'R': r_combined.tolist(),
            'time': reward_history['time']
        }
    }

    with open(f"{output_dir}/rewards_data_{num_tasks}_tasks.json", 'w') as f:
        json.dump(rewards_data, f, indent=4)
    print(f"Rewards data saved for {num_tasks} tasks.")