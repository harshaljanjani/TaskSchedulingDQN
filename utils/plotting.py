import os
import matplotlib.pyplot as plt

def plot_metrics(env, output_dir):
    """
    Plot and save various metrics from the environment, such as energy consumption and reward history.

    Parameters:
    - env: The environment object that contains the metrics and history.
    - output_dir: Directory to save the plots.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    time_points = [entry['time'] for entry in env.energy_consumption_history]
    energy_values = [entry['energy'] for entry in env.energy_consumption_history]
    plt.plot(time_points, energy_values, 'b-', label='Energy Consumption')
    plt.title('Energy Consumption over Time')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'energy_vs_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

    if env.completed_tasks:  
        plt.figure(figsize=(10, 6))
        completion_times = [task.completion_time for task in env.completed_tasks]
        energy_at_completion = []

        for ct in completion_times:
            closest_time_idx = min(range(len(time_points)),
                                   key=lambda i: abs(time_points[i] - ct))
            energy_at_completion.append(energy_values[closest_time_idx])

        plt.scatter(completion_times, energy_at_completion, c='purple', alpha=0.5, label='Energy at Completion')
        plt.xlabel('Task Completion Time')
        plt.ylabel('Energy Consumption')
        plt.title('Energy Consumption vs Task Completion')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'energy_vs_task_completion.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Metrics plots saved in {output_dir}")