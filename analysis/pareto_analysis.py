import os
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Solution:
    task_size: int
    metrics: Dict[str, float]
    is_pareto_optimal: bool = False

def is_dominated(metrics1: Dict[str, float], metrics2: Dict[str, float], objectives: Dict[str, str]) -> bool:
    """
    Check if metrics1 is dominated by metrics2.
    A solution is dominated if another solution is better in at least one objective
    and no worse in all other objectives.

    Parameters:
    - metrics1: Metrics of the first solution.
    - metrics2: Metrics of the second solution.
    - objectives: Dictionary specifying 'min' or 'max' for each metric.

    Returns:
    - bool: True if metrics1 is dominated by metrics2, False otherwise.
    """

    at_least_one_better = False
    for metric, goal in objectives.items():
        if goal == 'min':
            if metrics2[metric] > metrics1[metric]:
                return False
            if metrics2[metric] < metrics1[metric]:
                at_least_one_better = True
        elif goal == 'max':
            if metrics2[metric] < metrics1[metric]:
                return False
            if metrics2[metric] > metrics1[metric]:
                at_least_one_better = True
    return at_least_one_better

def find_pareto_optimal(solutions: List[Solution], objectives: Dict[str, str]) -> List[Solution]:
    """
    Identify Pareto-optimal solutions from the given set of solutions.

    Parameters:
    - solutions: List of Solution objects.
    - objectives: Dictionary specifying 'min' or 'max' for each metric.

    Returns:
    - List[Solution]: Updated list of solutions with Pareto-optimality status.
    """

    for solution in solutions:
        solution.is_pareto_optimal = True
        for other_solution in solutions:
            if solution != other_solution and is_dominated(solution.metrics, other_solution.metrics, objectives):
                solution.is_pareto_optimal = False
                break
    return solutions

def plot_pareto_analysis(solutions: List[Solution], output_dir: str, metrics_to_compare: List[Tuple[str, str]]):
    """
    Generate Pareto analysis plots for given solutions.

    Parameters:
    - solutions: List of Solution objects.
    - output_dir: Directory to save plots.
    - metrics_to_compare: List of metric pairs to plot against each other.
    """

    os.makedirs(output_dir, exist_ok=True)

    for metric1, metric2 in metrics_to_compare:
        plt.figure(figsize=(12, 8))

        x_non_pareto = [s.metrics[metric1] for s in solutions if not s.is_pareto_optimal]
        y_non_pareto = [s.metrics[metric2] for s in solutions if not s.is_pareto_optimal]
        x_pareto = [s.metrics[metric1] for s in solutions if s.is_pareto_optimal]
        y_pareto = [s.metrics[metric2] for s in solutions if s.is_pareto_optimal]

        plt.scatter(x_non_pareto, y_non_pareto, c='gray', alpha=0.5, label='Non-Pareto Optimal')

        plt.scatter(x_pareto, y_pareto, c='red', alpha=0.8, label='Pareto Optimal')

        for s in solutions:
            if s.is_pareto_optimal:
                plt.annotate(f'Tasks: {s.task_size}', (s.metrics[metric1], s.metrics[metric2]),
                             xytext=(5, 5), textcoords='offset points')

        if len(x_pareto) > 1:
            pareto_points = sorted(zip(x_pareto, y_pareto))
            x_line, y_line = zip(*pareto_points)
            plt.plot(x_line, y_line, 'r--', alpha=0.5)

        plt.xlabel(metric1.replace('_', ' ').title())
        plt.ylabel(metric2.replace('_', ' ').title())
        plt.title(f'Pareto Analysis: {metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plot_path = os.path.join(output_dir, f'pareto_{metric1}_vs_{metric2}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_pareto_solutions(all_final_metrics: Dict[int, Dict], base_output_dir: str):
    """
    Perform Pareto analysis and generate visualizations.

    Parameters:
    - all_final_metrics: Dictionary containing metrics for each task size.
    - base_output_dir: Directory to save Pareto analysis results and plots.

    Returns:
    - List[Solution]: List of solutions with updated Pareto-optimality status.
    """
    
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

    solutions = find_pareto_optimal(solutions, objectives)

    metrics_to_compare = [
        ('power_consumption', 'utilization'),
        ('response_time', 'cpu_utilization'),
        ('memory_utilization', 'error_rate')
    ]

    pareto_dir = os.path.join(base_output_dir, 'pareto_analysis')
    plot_pareto_analysis(solutions, pareto_dir, metrics_to_compare)

    pareto_results = {
        'pareto_optimal_solutions': [
            {
                'task_size': s.task_size,
                'metrics': s.metrics
            }
            for s in solutions if s.is_pareto_optimal
        ]
    }

    with open(os.path.join(pareto_dir, 'pareto_analysis_results.json'), 'w') as f:
        json.dump(pareto_results, f, indent=4)

    return solutions