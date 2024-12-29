import random
import numpy as np
from collections import deque
import tensorflow as tf

class EnergyAwareScheduler:
    """
    Implements an energy-aware task scheduler using a Q-learning approach.
    """

    def __init__(self, state_size, action_size, memory_size=2000, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=0.001, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.model = self._build_model()

    def _build_model(self):
        """
        Build a neural network model for Q-value estimation.
        """

        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(24, activation='relu')(inputs)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, delta_Q):
        """
        Store a step in the replay memory.
        """

        self.memory.append((state, action, reward, next_state, done, delta_Q))

    def predict(self, state):
        """
        Predict Q-values for a given state using the model.
        """

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.model(state_tensor)

    def act(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        """

        if random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Random action chosen: {action}")
            return action

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        act_values = self.model(state_tensor)
        action = tf.argmax(act_values, axis=1).numpy()[0]
        print(f"Action chosen by model: {action}")
        return action

    def replay(self, batch_size):
        """
        Train the model using a batch of experiences from the replay memory.
        """
        
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        deltas = np.array([i[5] for i in minibatch])

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        old_weights = self.model.get_weights()

        target = self.model(states)
        target_next = self.model(next_states)

        for i in range(len(actions)):
            target_val = rewards[i] + self.gamma * tf.reduce_max(target_next[i]) * (1 - dones[i])
            target = tf.tensor_scatter_nd_update(target, [[i, actions[i]]], [target_val + deltas[i]])

        self.model.fit(states, target, epochs=1, verbose=0)

        new_weights = self.model.get_weights()
        final_weights = [
            self.tau * new + (1 - self.tau) * old
            for new, old in zip(new_weights, old_weights)
        ]
        self.model.set_weights(final_weights)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay