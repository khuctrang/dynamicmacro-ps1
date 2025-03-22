import numpy as np
import matplotlib.pyplot as plt

def rouwenhorst(n, p, q):
    """
    Constructs the transition matrix using Rouwenhorst's method.
    
    n: Number of states
    p, q: Persistence parameters
    """
    if n == 2:
        return np.array([[p, 1 - p], [1 - q, q]])

    Pn_1 = rouwenhorst(n - 1, p, q)  # Recursive call for (n-1)
    
    # Expanding the transition matrix
    P = np.zeros((n, n))
    P[:-1, :-1] += p * Pn_1
    P[:-1, 1:] += (1 - p) * Pn_1
    P[1:, :-1] += (1 - q) * Pn_1
    P[1:, 1:] += q * Pn_1
    
    return P / np.sum(P, axis=1, keepdims=True)
def compute_state_vector(n, gamma, sigma):
    """
    Computes the state vector for the Markov Chain approximation.
    
    n: Number of states
    gamma: Persistence parameter
    sigma: Standard deviation of the white noise
    """
    sigma_y = sigma / np.sqrt(1 - gamma**2)  # AR(1) stationary std deviation
    return np.linspace(-sigma_y * np.sqrt(n - 1), sigma_y * np.sqrt(n - 1), n)

# Parameters
n = 7  # Number of states
sigma = 1  # Standard deviation of white noise
gamma_values = [0.75, 0.85, 0.95, 0.99]  # Different γ1 values
T = 50  # Number of periods
np.random.seed(2025)  # Set seed for reproducibility

# Plot setup
plt.figure(figsize=(10, 6))

for gamma in gamma_values:
    p = q = (1 + gamma) / 2  # Persistence parameter
    P = rouwenhorst(n, p, q)  # Compute transition matrix
    states = compute_state_vector(n, gamma, sigma)  # Compute state values
    
    # Compute stationary distribution (left eigenvector of P corresponding to eigenvalue 1)
    eigvals, eigvecs = np.linalg.eig(P.T)
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stat_dist = stat_dist / np.sum(stat_dist)  # Normalize

    # Draw the initial state from the stationary distribution
    state_index = np.random.choice(n, p=stat_dist.flatten())  
    simulation = [states[state_index]]

    # Simulate the Markov Chain
    for _ in range(T - 1):
        state_index = np.random.choice(n, p=P[state_index])
        simulation.append(states[state_index])

    # Plot results
    plt.plot(range(T), simulation, label=f'γ1 = {gamma}')

# Finalize plot
plt.xlabel('Time Periods')
plt.ylabel('State Value')
plt.title('Markov Chain Simulations for Different γ1 Values')
plt.legend()
plt.grid(True)
plt.show()
