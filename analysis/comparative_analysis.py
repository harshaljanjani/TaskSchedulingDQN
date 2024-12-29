import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def compare_energy_consumption(base_output_dir='results'):
    """
    Compare energy consumption across different scheduling algorithms and save results to CSV.

    Parameters:
    - base_output_dir: Base directory containing results for different task sizes.

    Returns:
    - pd.DataFrame: DataFrame containing energy consumption comparison.
    """

    reference_values = {
        50: {'FCFS': 40.56, 'EDF': 38.24, 'RR': 42.15, 'MOABCQ': 39.87},
        100: {'FCFS': 82.50, 'EDF': 69.70, 'RR': 73.90, 'MOABCQ': 71.43},
        200: {'FCFS': 87.45, 'EDF': 89.12, 'RR': 82.34, 'MOABCQ': 74.67},
        500: {'FCFS': 95.70, 'EDF': 114.2, 'RR': 90.80, 'MOABCQ': 80.18},
        1000: {'FCFS': 113.90, 'EDF': 131.25, 'RR': 120.56, 'MOABCQ': 114.76},
    }

    our_values = {}
    for num_tasks in reference_values.keys():
        task_dir = os.path.join(base_output_dir, f'tasks_{num_tasks}')
        metrics_file = os.path.join(task_dir, 'experiment_details.json')

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                our_values[num_tasks] = data['final_metrics']['power_consumption']

    rows = []
    for num_tasks, ref_values in reference_values.items():
        row = {'Number_of_Tasks': num_tasks, 'Our_Algorithm': our_values.get(num_tasks, 0)}
        row.update(ref_values)

        for algo in ref_values.keys():
            row[f'Improvement_vs_{algo}'] = round(((ref_values[algo] - row['Our_Algorithm']) / ref_values[algo] * 100), 2)
        rows.append(row)

    df = pd.DataFrame(rows)

    output_file = os.path.join(base_output_dir, 'energy_consumption_comparison.csv')
    df.to_csv(output_file, index=False)
    print(f"Energy consumption comparison saved to: {output_file}")

    return df

def plot_energy_comparison(df, base_output_dir='results'):
    """
    Plot energy consumption comparison across different algorithms.

    Parameters:
    - df: DataFrame containing comparison data.
    - base_output_dir: Directory to save the comparison plot.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(df['Number_of_Tasks'], df['Our_Algorithm'], 'o-', label='Our Algorithm', linewidth=2)
    for algo in ['FCFS', 'EDF', 'RR', 'MOABCQ']:
        plt.plot(df['Number_of_Tasks'], df[algo], 's-', label=algo)

    plt.title('Energy Consumption Comparison Across Algorithms')
    plt.xlabel('Number of Tasks')
    plt.ylabel('Energy Consumption (W)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_file = os.path.join(base_output_dir, 'energy_consumption_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {plot_file}")

def plot_final_comparisons(all_final_metrics, base_output_dir):
    """
    Plot comparisons for key metrics across different task sizes.

    Parameters:
    - all_final_metrics: Dictionary containing final metrics for each task size.
    - base_output_dir: Directory to save comparison plots.
    """
    
    metrics_to_compare = ['utilization', 'cpu_utilization', 'memory_utilization', 'response_time']
    task_sizes = sorted(all_final_metrics.keys())

    comparison_dir = os.path.join(base_output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)

    for metric in metrics_to_compare:
        plt.figure(figsize=(10, 6))
        values = [all_final_metrics[ts]['final_metrics'][metric] for ts in task_sizes]
        plt.plot(task_sizes, values, 'o-', label=metric)
        plt.title(f'{metric.replace("_", " ").title()} vs Task Size')
        plt.xlabel('Number of Tasks')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = os.path.join(comparison_dir, f'{metric}_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved for {metric}: {plot_path}")