import tensorflow as tf

class MetricsProcessor:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma

    def process_stage1(self, state, action, reward, next_state, metrics, Q_model, w_U=0.8, w_M=0.6, w_D=0.6):
        """
        Calculate the first stage reward (R1) and Q-value update (ΔQ1).
        
        Parameters:
        - state: Current state of the environment.
        - action: Action taken in the current state.
        - reward: Immediate reward received.
        - next_state: State after taking the action.
        - metrics: Current metrics data from the environment.
        - Q_model: The Q-value estimation model.
        - w_U, w_M, w_D: Weights for utilization, memory, and disk.

        Returns:
        - R1: Reward for stage 1.
        - delta_Q1: Q-value update for stage 1.
        """

        U = metrics['utilization'] / 100
        M = metrics['memory_utilization'] / 100
        D = metrics['disk_usage'] / 100

        R1 = w_U * U + w_M * M + w_D * D

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

        Q_next = tf.reduce_max(Q_model(next_state_tensor))

        Q_current = Q_model(state_tensor)[0][action]

        delta_Q1 = self.alpha * (R1 + self.gamma * Q_next - Q_current)

        return R1, delta_Q1

    def process_stage2(self, state, action, reward, next_state, metrics, Q_model, w_C=0.7, w_R=0.7):
        """
        Calculate the second stage reward (R2) and Q-value update (ΔQ2).
        
        Parameters:
        - state: Current state of the environment.
        - action: Action taken in the current state.
        - reward: Immediate reward received.
        - next_state: State after taking the action.
        - metrics: Current metrics data from the environment.
        - Q_model: The Q-value estimation model.
        - w_C, w_R: Weights for CPU and RAM utilization.

        Returns:
        - R2: Reward for stage 2.
        - delta_Q2: Q-value update for stage 2.
        """

        C = metrics['cpu_utilization'] / 100
        R = metrics['ram_utilization'] / 100

        R2 = w_C * C + w_R * R

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

        Q_next = tf.reduce_max(Q_model(next_state_tensor))

        Q_current = Q_model(state_tensor)[0][action]

        delta_Q2 = self.alpha * (R2 + self.gamma * Q_next - Q_current)

        return R2, delta_Q2

    def print_success_rate(self, task_queue, completed_tasks):
        """
        Print the success rate of task allocation.
        
        Parameters:
        - task_queue: List of tasks that are yet to be processed.
        - completed_tasks: List of tasks that have been successfully completed.
        """
        
        total_tasks = len(task_queue) + len(completed_tasks)
        success_rate = (len(completed_tasks) / total_tasks) * 100 if total_tasks > 0 else 0
        print(f"Success Rate: {success_rate:.2f}%")