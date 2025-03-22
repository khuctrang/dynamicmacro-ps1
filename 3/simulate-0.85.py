import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(2025)

# Number of periods to simulate
T = 50

# Choose the initial state randomly with uniform probability
initial_state = np.random.choice(len(state_vector))

# Simulate the Markov Chain
states = np.zeros(T, dtype=int)
states[0] = initial_state

for t in range(1, T):
    # Get the transition probabilities for the current state
    transition_probs = transition_matrix[states[t-1], :]
    # Choose next state based on transition probabilities
    states[t] = np.random.choice(len(state_vector), p=transition_probs)

# Convert state indices to actual values
simulated_values = state_vector[states]

# Plot the simulated Markov Chain
plt.figure(figsize=(10, 5))
plt.plot(range(T), simulated_values, marker='o', linestyle='-', color='b', markersize=5, label="Simulated Markov Chain")
plt.xlabel("Time Period")
plt.ylabel("State Value")
plt.title("Simulated Markov Chain (50 Periods)")
plt.legend()
plt.grid(True)
plt.show()