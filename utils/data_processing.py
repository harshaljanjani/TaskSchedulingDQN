import os
import json

def save_experiment_results(metrics, num_tasks, all_final_metrics, output_dir):
    """
    Save experiment results to JSON files in the specified directory.

    Parameters:
    - metrics: Dictionary containing detailed metrics of the current experiment.
    - num_tasks: Number of tasks in the experiment.
    - all_final_metrics: Dictionary aggregating results of all experiments.
    - output_dir: Directory to save the results.

    Returns:
    - all_final_metrics: Updated dictionary with the current experiment's results.
    """

    os.makedirs(output_dir, exist_ok=True)

    detailed_filename = os.path.join(output_dir, f'experiment_details.json')
    with open(detailed_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Detailed results saved: {detailed_filename}")

    all_final_metrics[num_tasks] = {
        'final_metrics': metrics['final_metrics'],
        'total_time': metrics['total_time'],
        'total_tasks': num_tasks
    }

    comparative_metrics = {
        'task_size_comparison': {},
        'normalized_metrics': {},
        'performance_ratios': {}
    }

    if len(all_final_metrics) > 1:
        comparative_metrics['task_size_comparison'] = all_final_metrics

        for task_size, data in all_final_metrics.items():
            normalized = {
                metric: value / task_size
                for metric, value in data['final_metrics'].items()
                if isinstance(value, (int, float))
            }
            comparative_metrics['normalized_metrics'][task_size] = normalized

        task_sizes = sorted(all_final_metrics.keys())
        for i in range(len(task_sizes)):
            for j in range(i + 1, len(task_sizes)):
                size1, size2 = task_sizes[i], task_sizes[j]
                ratio_key = f"{size1}_vs_{size2}"
                comparative_metrics['performance_ratios'][ratio_key] = {
                    'time_ratio': all_final_metrics[size1]['total_time'] / all_final_metrics[size2]['total_time'],
                    'efficiency_ratio': (all_final_metrics[size1]['total_time'] / size1) /
                                        (all_final_metrics[size2]['total_time'] / size2)
                }

    comparison_filename = os.path.join(output_dir, 'comparative_metrics.json')
    with open(comparison_filename, 'w') as f:
        json.dump(comparative_metrics, f, indent=4)
    print(f"Comparative metrics saved: {comparison_filename}")

    return all_final_metrics

def analyze_pareto_solutions(all_final_metrics, base_output_dir):
    """
    Perform Pareto analysis on solutions and return a list of Pareto-optimal solutions.

    Parameters:
    - all_final_metrics: Dictionary containing metrics for each task size.
    - base_output_dir: Directory to save Pareto analysis results.

    Returns:
    - pareto_solutions: List of Pareto-optimal solutions.
    """
    
    from analysis.pareto_analysis import find_pareto_optimal, Solution

    objectives = {
        'utilization': 'max',
        'cpu_utilization': 'max',
        'memory_utilization': 'max',
        'response_time': 'min',
        'power_consumption': 'min',
        'error_rate': 'min'
    }

    solutions = [
        Solution(task_size=task_size, metrics=data['final_metrics'])
        for task_size, data in all_final_metrics.items()
    ]

    pareto_solutions = find_pareto_optimal(solutions, objectives)

    pareto_dir = os.path.join(base_output_dir, 'pareto_analysis')
    os.makedirs(pareto_dir, exist_ok=True)

    pareto_results = {
        'pareto_optimal_solutions': [
            {
                'task_size': solution.task_size,
                'metrics': solution.metrics
            }
            for solution in pareto_solutions if solution.is_pareto_optimal
        ]
    }

    with open(os.path.join(pareto_dir, 'pareto_optimal_solutions.json'), 'w') as f:
        json.dump(pareto_results, f, indent=4)
    print(f"Pareto-optimal solutions saved: {os.path.join(pareto_dir, 'pareto_optimal_solutions.json')}")

    return pareto_solutions