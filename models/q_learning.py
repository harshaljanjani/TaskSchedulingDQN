import numpy as np
import tensorflow as tf

class QLearning:
    """
    Implements Q-learning for optimizing task scheduling.
    This class encapsulates the Q-learning algorithm, including policy updates and model training.
    """
    
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds a neural network model to estimate Q-values for each state-action pair.
        """

        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(24, activation='relu')(inputs)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def predict(self, state):
        """
        Predict Q-values for a given state using the Q-learning model.

        Parameters:
        - state: Current state of the environment.

        Returns:
        - Predicted Q-values for each possible action.
        """

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.model(state_tensor)

    def act(self, state):
        """
        Choose an action based on the epsilon-greedy policy.

        Parameters:
        - state: Current state of the environment.

        Returns:
        - action: The action chosen based on exploration-exploitation trade-off.
        """

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.predict(state)
            return np.argmax(q_values[0])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using the Q-learning update rule.
        
        Parameters:
        - state: Current state of the environment.
        - action: Action taken at the current state.
        - reward: Reward received after taking the action.
        - next_state: State after taking the action.
        - done: Boolean indicating whether the episode is done.
        """

        q_values = self.predict(state)
        target_q_value = q_values[0][action]

        next_q_values = self.predict(next_state)
        next_max_q_value = np.max(next_q_values[0])

        target = reward + self.gamma * next_max_q_value * (1 - done)

        new_q_value = (1 - self.alpha) * target_q_value + self.alpha * target
        q_values[0][action] = new_q_value

        self.model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size, memory):
        """
        Perform experience replay to update the Q-values by training on a batch of experiences.

        Parameters:
        - batch_size: Number of experiences to sample from memory for replay.
        - memory: List of past experiences (state, action, reward, next_state, done).
        """
        
        if len(memory) < batch_size:
            return

        minibatch = np.random.choice(memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            self.update(state, action, reward, next_state, done)