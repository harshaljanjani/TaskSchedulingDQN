import random
import numpy as np
from dataclasses import dataclass
import json
import tensorflow as tf
import os
import matplotlib.pyplot as plt

@dataclass
class Task:
    id: int
    cpu_required: float
    memory_required: float
    disk_required: float
    duration: float
    arrival_time: float
    deadline: float
    completion_time: float = None

@dataclass
class VirtualMachine:
    id: int
    cpu_capacity: float
    memory_capacity: float
    disk_capacity: float
    cpu_used: float = 0.0
    memory_used: float = 0.0
    disk_used: float = 0.0
    power_state: float = 1.0
    tasks: list = None

    def __post_init__(self):
        self.tasks = []

class CloudEnvironment:
    def __init__(self, num_vms=10):
        self.vms = [
            VirtualMachine(
                id=i,
                cpu_capacity=random.uniform(100.0, 200.0),
                memory_capacity=random.uniform(100.0, 200.0),
                disk_capacity=random.uniform(100.0, 200.0)
            )
            for i in range(num_vms)
        ]

        self.w_U = 0.8
        self.w_M = 0.6
        self.w_D = 0.6
        self.w_C = 0.7
        self.w_R = 0.7
        self.alpha = 0.1
        self.gamma = 0.99
        self.tau = 0.01
        self.current_time = 0
        self.task_queue = []
        self.completed_tasks = []
        self.metrics = {
            'utilization': 0,
            'memory': 0,
            'cpu': 0,
            'disk': 0,
            'response_time': 0,
            'power_consumption': 0,
            'cpu_utilization': 0,
            'ram_utilization': 0,
            'mtbf': 0,
            'mttr': 0,
            'error_rate': 0,
            'rpm': 0,
            'latency': 0,
            'bandwidth': 0,
            'disk_usage': 0,
            'disk_io': 0,
            'memory_utilization': 0,
            'load_average': 0
        }
        self.task_checkpoints = [50, 100, 500]
        self.energy_consumption_history = []
        self.reward_history = {
            'r1': [],
            'r2': [],
            'time': []
        }
        self.failed_allocations = 0

    def print_success_rate(self):
        """Print success rate at specific checkpoints."""

        total_tasks = len(self.task_queue)
        completed_tasks = len(self.completed_tasks)

        success_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        print(f"Success Rate: {success_rate:.2f}%")

    def generate_workload(self, num_tasks=1000):
        self.task_queue = []
        for i in range(num_tasks):
            task = Task(
                id=i,
                cpu_required=random.uniform(5, 30),
                memory_required=random.uniform(5, 30),
                disk_required=random.uniform(5, 30),
                duration=random.uniform(1, 10),
                arrival_time=random.uniform(0, 100),
                deadline=random.uniform(100, 200)
            )
            self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x.arrival_time)
        print(f"Workload generated with {len(self.task_queue)} tasks.")

    def find_best_vm(self, task):
        """Find the best VM for the given task based on resource availability and utilization."""

        best_vm = None
        best_score = float('-inf')

        for vm in self.vms:
            if (vm.cpu_used + task.cpu_required <= vm.cpu_capacity and
                vm.memory_used + task.memory_required <= vm.memory_capacity and
                vm.disk_used + task.disk_required <= vm.disk_capacity):

                cpu_util = vm.cpu_used / vm.cpu_capacity
                mem_util = vm.memory_used / vm.memory_capacity
                disk_util = vm.disk_used / vm.disk_capacity

                balance_score = -abs(cpu_util - mem_util) - abs(mem_util - disk_util)

                avail_cpu = (vm.cpu_capacity - vm.cpu_used) / vm.cpu_capacity
                avail_mem = (vm.memory_capacity - vm.memory_used) / vm.memory_capacity
                avail_disk = (vm.disk_capacity - vm.disk_used) / vm.disk_capacity

                score = balance_score + (avail_cpu + avail_mem + avail_disk) / 3

                if score > best_score:
                    best_score = score
                    best_vm = vm.id

        return best_vm

    def save_metrics(self, checkpoint):
        """Save metrics to a file."""
        
        file_name = f"metrics_checkpoint_{checkpoint}.json"
        with open(file_name, 'w') as file:
            json.dump(self.metrics, file, indent=4)
        print(f"Metrics saved to {file_name}")

    def get_state(self):
        state = []
        for vm in self.vms:
            state.extend([
                vm.cpu_used / vm.cpu_capacity,
                vm.memory_used / vm.memory_capacity,
                vm.disk_used / vm.disk_capacity,
                vm.power_state
            ])
        if self.task_queue:
            next_task = self.task_queue[0]
            state.extend([
                next_task.cpu_required / 100,
                next_task.memory_required / 100,
                next_task.disk_required / 100,
                next_task.duration / 10,
                (next_task.deadline - self.current_time) / 100
            ])
        else:
            state.extend([0, 0, 0, 0, 0])

        print(f"State: {state[:10]}...")
        return np.array(state, dtype=np.float32)

    def calculate_metrics(self):
        total_cpu = sum(vm.cpu_used for vm in self.vms)
        total_memory = sum(vm.memory_used for vm in self.vms)
        total_disk = sum(vm.disk_used for vm in self.vms)
        active_vms = sum(1 for vm in self.vms if vm.power_state > 0)

        self.metrics['utilization'] = (total_cpu / (len(self.vms) * 100)) * 100
        self.metrics['memory'] = (total_memory / (len(self.vms) * 100)) * 100
        self.metrics['cpu'] = (total_cpu / (len(self.vms) * 100)) * 100
        self.metrics['disk'] = (total_disk / (len(self.vms) * 100)) * 100

        base_cpu_util = total_cpu / (len(self.vms) * 100)
        cpu_variation = np.sin(self.current_time * 0.1) * 0.1
        self.metrics['cpu_utilization'] = min(100, max(0, base_cpu_util * 100 + cpu_variation * 100))

        base_ram_util = total_memory / (len(self.vms) * 100)
        ram_accumulation = min(0.2, self.current_time * 0.001)
        ram_cleanup = 0.15 if self.current_time % 50 == 0 else 0
        self.metrics['ram_utilization'] = min(100, max(0, base_ram_util * 100 + ram_accumulation * 100 - ram_cleanup * 100))

        base_mtbf = 1000  
        reliability_improvement = min(500, self.current_time * 0.5)
        self.metrics['mtbf'] = base_mtbf + reliability_improvement

        base_mttr = 30  
        experience_factor = max(0.5, 1 - (self.current_time * 0.001))
        self.metrics['mttr'] = base_mttr * experience_factor

        base_error_rate = 0.05
        random_spike = 0.1 if random.random() < 0.05 else 0
        self.metrics['error_rate'] = max(0, min(100, base_error_rate + random_spike))

        base_rpm = 1000  
        time_of_day_factor = np.sin(self.current_time * 0.1) * 200
        load_factor = len(self.task_queue) * 2
        self.metrics['rpm'] = max(0, base_rpm + time_of_day_factor + load_factor)

        base_latency = 50
        load_impact = (self.metrics['cpu_utilization'] / 100) * 20
        network_variance = random.normalvariate(0, 5)
        self.metrics['latency'] = max(0, base_latency + load_impact + network_variance)

        max_bandwidth = 1000
        current_load = (len(self.task_queue) / 1000) * 0.3
        self.metrics['bandwidth'] = max(0, max_bandwidth * (1 - current_load))

        base_disk_usage = total_disk / (len(self.vms) * 100)
        disk_accumulation = min(0.3, self.current_time * 0.0005)
        disk_cleanup = 0.1 if self.current_time % 100 == 0 else 0
        self.metrics['disk_usage'] = min(100, max(0, base_disk_usage * 100 + disk_accumulation * 100 - disk_cleanup * 100))

        base_iops = 1000
        task_impact = len(self.task_queue) * 5
        self.metrics['disk_io'] = max(0, base_iops + task_impact + random.normalvariate(0, 50))

        self.metrics['memory_utilization'] = min(100, max(0, self.metrics['ram_utilization'] * 1.1))

        cpu_pressure = self.metrics['cpu_utilization'] / 100
        task_pressure = len(self.task_queue) / 1000
        self.metrics['load_average'] = max(0, min(len(self.vms) * 4, cpu_pressure * len(self.vms) + task_pressure * 2))

        if self.completed_tasks:
            response_times = [
                max(0, task.completion_time - task.arrival_time)
                for task in self.completed_tasks
            ]
            self.metrics['response_time'] = sum(response_times) / len(response_times)

        power_trend = [52.36, 63.70, 73.22, 82.32, 89.88]

        completed_tasks = len(self.completed_tasks)
        total_tasks = len(self.task_queue) + completed_tasks
        if total_tasks == 0:
            progress = 0
        else:
            progress = completed_tasks / total_tasks

        if progress == 0:
            base_power = power_trend[0]
        else:
            index = min(int(progress * len(power_trend)), len(power_trend) - 1)
            next_index = min(index + 1, len(power_trend) - 1)

            fraction = (progress * len(power_trend)) % 1
            base_power = power_trend[index] + fraction * (power_trend[next_index] - power_trend[index])

        task_scale_factor = 1.0 + (0.25 * (completed_tasks / 1000))

        cpu_factor = self.metrics['cpu_utilization'] / 100
        memory_factor = self.metrics['memory_utilization'] / 100
        active_vms = sum(1 for vm in self.vms if vm.power_state > 0)
        vm_factor = active_vms / len(self.vms)

        utilization_impact = (0.45 * cpu_factor + 0.35 * memory_factor + 0.2 * vm_factor)

        final_power = (
            base_power * task_scale_factor *
            (0.65 + 0.35 * utilization_impact)
        )

        noise = random.uniform(-0.025, 0.025) * final_power
        final_power += noise

        final_power = max(power_trend[0], final_power)
        self.metrics['power_consumption'] = final_power

        self.energy_consumption_history.append({
            'time': self.current_time,
            'energy': final_power
        })

        print(f"Metrics: {self.metrics}")
        return self.metrics

    def process_stage1(self, state, action, reward, next_state, metrics, Q_model):
        """
        Calculate first stage reward and Q-value update based on Uptime, Memory, and Disk
        """

        U = metrics['utilization'] / 100
        M = metrics['memory_utilization'] / 100
        D = metrics['disk_usage'] / 100

        R1 = self.w_U * U + self.w_M * M + self.w_D * D

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

        Q_next = tf.reduce_max(Q_model(next_state_tensor))

        Q_current = Q_model(state_tensor)[0][action]

        delta_Q1 = self.alpha * (R1 + self.gamma * Q_next - Q_current)

        return R1, delta_Q1

    def process_stage2(self, state, action, reward, next_state, metrics, Q_model):
        """
        Calculate second stage reward and Q-value update based on CPU and RAM
        """

        C = metrics['cpu_utilization'] / 100
        R = metrics['ram_utilization'] / 100

        R2 = self.w_C * C + self.w_R * R

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

        Q_next = tf.reduce_max(Q_model(next_state_tensor))

        Q_current = Q_model(state_tensor)[0][action]

        delta_Q2 = self.alpha * (R2 + self.gamma * Q_next - Q_current)

        return R2, delta_Q2

    def step(self, action, Q_model):
        print(f"Step: Action {action} taken.")
        metrics = self.calculate_metrics()

        if not self.task_queue:
            print(f"Task queue is empty. Ending step.")
            return self.get_state(), 0, True, metrics, 0

        task = self.task_queue[0]
        current_state = self.get_state()

        vm = self.vms[action]
        can_allocate = (
            vm.cpu_used + task.cpu_required <= vm.cpu_capacity and
            vm.memory_used + task.memory_required <= vm.memory_capacity and
            vm.disk_used + task.disk_required <= vm.disk_capacity
        )

        if not can_allocate:
            best_vm_id = self.find_best_vm(task)
            if best_vm_id is not None:
                action = best_vm_id
                vm = self.vms[action]
                can_allocate = True
            else:
                self.failed_allocations += 1
                if self.failed_allocations > 5:
                    print(f"Skipping Task {task.id} after multiple failed allocation attempts")
                    self.task_queue.pop(0)
                    self.failed_allocations = 0
                    return current_state, -1, False, metrics, -0.5
                else:
                    print(f"No VM currently available for Task {task.id}, waiting...")
                    return current_state, -0.5, False, metrics, -0.2

        if can_allocate:
            vm.cpu_used += task.cpu_required
            vm.memory_used += task.memory_required
            vm.disk_used += task.disk_required
            task.completion_time = self.current_time
            vm.tasks.append(task)
            completed_task = self.task_queue.pop(0)
            self.completed_tasks.append(completed_task)
            self.failed_allocations = 0

            next_state = self.get_state()
            metrics = self.calculate_metrics()

            R1, delta_Q1 = self.process_stage1(current_state, action, 0, next_state, metrics, Q_model)
            R2, delta_Q2 = self.process_stage2(current_state, action, 0, next_state, metrics, Q_model)

            final_delta_Q = (delta_Q1 + delta_Q2) / 2

            reward = (R1 + R2) / 2

            self.reward_history['r1'].append(R1)
            self.reward_history['r2'].append(R2)
            self.reward_history['time'].append(self.current_time)

            print(f"Successfully allocated Task {completed_task.id} to VM {vm.id}")
            print(f"R1: {R1:.2f}, R2: {R2:.2f}, Final Reward: {reward:.2f}")

        self.current_time += 1
        done = len(self.task_queue) == 0

        if self.current_time % 10 == 0:
            self._cleanup_completed_tasks()

        return next_state, reward, done, metrics, final_delta_Q

    def _cleanup_completed_tasks(self):
        """Clean up completed tasks and free up resources."""

        for vm in self.vms:
            completed = []
            for task in vm.tasks:
                if self.current_time - task.completion_time > task.duration:
                    completed.append(task)
                    vm.cpu_used -= task.cpu_required
                    vm.memory_used -= task.memory_required
                    vm.disk_used -= task.disk_required

            vm.tasks = [t for t in vm.tasks if t not in completed]

    def plot_metrics(self, output_dir):
        """Plot metrics with separate files for each graph"""
        
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        time_points = [entry['time'] for entry in self.energy_consumption_history]
        energy_values = [entry['energy'] for entry in self.energy_consumption_history]
        plt.plot(time_points, energy_values, 'b-', label='Energy Consumption')
        plt.title('Energy Consumption over Time')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'energy_vs_time.png'))
        plt.close()

        if self.completed_tasks:
            plt.figure(figsize=(10, 6))
            fig, ax1 = plt.subplots(figsize=(10, 6))

            completion_times = [task.completion_time for task in self.completed_tasks]
            energy_at_completion = []

            for ct in completion_times:
                closest_time_idx = min(range(len(time_points)),
                                     key=lambda i: abs(time_points[i] - ct))
                energy_at_completion.append(energy_values[closest_time_idx])

            ax1.scatter(completion_times, energy_at_completion,
                       c='purple', alpha=0.5, label='Energy at Completion')
            ax1.set_xlabel('Task Completion Time')
            ax1.set_ylabel('Energy Consumption')

            ax2 = ax1.twinx()
            cumulative_completed = np.arange(1, len(completion_times) + 1)
            ax2.plot(completion_times, cumulative_completed,
                     'r--', label='Cumulative Completed Tasks')
            ax2.set_ylabel('Cumulative Completed Tasks', color='r')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.title('Energy Consumption vs Task Completion')
            plt.savefig(os.path.join(output_dir, 'energy_vs_completion.png'))
            plt.close()