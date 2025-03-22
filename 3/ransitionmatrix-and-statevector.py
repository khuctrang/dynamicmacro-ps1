import numpy as np

def rouwenhorst(n, p, q):
    """ Recursively constructs the Rouwenhorst transition matrix for an n-state Markov Chain """
    if n == 2:
        return np.array([[p, 1 - p], [1 - q, q]])
    
    P_n_minus_1 = rouwenhorst(n - 1, p, q)
    P_n = np.zeros((n, n))
    
    # Construct P_n using the recursive formula
    P_n[:-1, :-1] += p * P_n_minus_1
    P_n[:-1, 1:] += (1 - p) * P_n_minus_1
    P_n[1:, :-1] += (1 - q) * P_n_minus_1
    P_n[1:, 1:] += q * P_n_minus_1
    
    # Normalize rows to sum to 1
    P_n /= P_n.sum(axis=1, keepdims=True)
    
    return P_n

# Define parameters
gamma1 = 0.85
sigma = 1
n = 7

# Compute standard deviation of stationary distribution
sigma_y = sigma / np.sqrt(1 - gamma1**2)

# Compute state vector (evenly spaced around 0)
state_vector = np.linspace(-sigma_y * np.sqrt(n-1), sigma_y * np.sqrt(n-1), n)

# Compute transition matrix using Rouwenhorst’s method
p = (1 + gamma1) / 2
q = p
transition_matrix = rouwenhorst(n, p, q)

# Print results
print("State Vector (S):\n", state_vector)
print("\nTransition Matrix (P):\n", transition_matrix)
