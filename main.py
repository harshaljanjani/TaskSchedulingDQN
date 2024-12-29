import os
from scheduler.cloud_environment import CloudEnvironment
from scheduler.task_scheduler import EnergyAwareScheduler
from analysis.pareto_analysis import analyze_pareto_solutions
from analysis.reward_analysis import plot_smoothed_rewards, save_rewards_data
from analysis.comparative_analysis import (
    compare_energy_consumption,
    plot_energy_comparison,
    plot_final_comparisons
)
from utils.data_processing import save_experiment_results
from utils.plotting import plot_metrics
from keras.backend import clear_session
import time

def run_experiment(num_tasks, episodes=10, base_output_dir='results'):
    """Run a single experiment with the specified number of tasks."""

    output_dir = os.path.join(base_output_dir, f'tasks_{num_tasks}')
    os.makedirs(output_dir, exist_ok=True)

    env = CloudEnvironment()
    env.generate_workload(num_tasks=num_tasks)

    state_size = env.get_state().shape[0]
    action_size = len(env.vms)
    agent = EnergyAwareScheduler(state_size, action_size)

    start_time = time.time()
    experiment_metrics = {
        'episodes': [],
        'checkpoints': {},
        'final_metrics': None,
        'total_time': None
    }

    for e in range(episodes):
        state = env.get_state()
        done = False
        episode_metrics = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, metrics, delta_Q = env.step(action, agent.model)
            agent.remember(state, action, reward, next_state, done, delta_Q)
            agent.replay(32)
            state = next_state
            episode_metrics.append(metrics.copy())

            if len(env.completed_tasks) in env.task_checkpoints:
                checkpoint = len(env.completed_tasks)
                if str(checkpoint) not in experiment_metrics['checkpoints']:
                    experiment_metrics['checkpoints'][str(checkpoint)] = metrics.copy()

        episode_summary = {
            'episode': e + 1,
            'final_metrics': metrics.copy(),
            'avg_metrics': {
                key: sum(step[key] for step in episode_metrics) / len(episode_metrics)
                for key in metrics.keys()
            }
        }
        experiment_metrics['episodes'].append(episode_summary)

        print(f"Episode {e + 1} completed: {metrics}")

    save_rewards_data(env.reward_history, output_dir, num_tasks)
    plot_smoothed_rewards(env.reward_history, output_dir, num_tasks)

    experiment_metrics['final_metrics'] = metrics.copy()
    experiment_metrics['total_time'] = time.time() - start_time

    plot_metrics(env, output_dir)

    return experiment_metrics, output_dir

def main():
    """Main function to run all experiments."""
    
    task_sizes = [50, 100, 200, 500, 1000]
    base_output_dir = 'results'
    all_final_metrics = {}

    for num_tasks in task_sizes:
        print(f"\nStarting experiment with {num_tasks} tasks...")

        experiment_metrics, output_dir = run_experiment(num_tasks, base_output_dir=base_output_dir)

        all_final_metrics = save_experiment_results(
            experiment_metrics, num_tasks, all_final_metrics, output_dir
        )

        clear_session()

        print(f"Completed experiment with {num_tasks} tasks")

    solutions = analyze_pareto_solutions(all_final_metrics, base_output_dir)

    print("\nPareto Optimal Solutions:")
    for solution in solutions:
        if solution.is_pareto_optimal:
            print(f"\nTask Size: {solution.task_size}")
            for metric, value in solution.metrics.items():
                print(f"{metric}: {value:.2f}")

    plot_final_comparisons(all_final_metrics, base_output_dir)

    df = compare_energy_consumption(base_output_dir)
    plot_energy_comparison(df, base_output_dir)

    print("\nAll experiments completed. Results organized in separate directories.")

if __name__ == "__main__":
    main()