import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters (example values, to be calibrated with actual data)
alpha = 0.35  # Output elasticity of private capital
beta = 0.5    # Output elasticity of labor
gamma = 0.15  # Output elasticity of infrastructure
delta_K = 0.05  # Depreciation rate of private capital
delta_G = 0.03  # Depreciation rate of infrastructure
tau = 0.2     # Public investment share
beta_discount = 0.96  # Discount factor
T = 100       # Time periods

# Initial values
K = np.zeros(T)
G = np.zeros(T)
Y = np.zeros(T)
C = np.zeros(T)
A = np.random.normal(1, 0.1, T)  # Stochastic TFP

K[0] = 100  # Initial private capital
G[0] = 50   # Initial infrastructure capital

# Production function
def production(K, G, A):
    return A * (K**alpha) * (G**gamma)

# Value function iteration (simplified)
def value_function_iteration(params, K, G, A):
    alpha, beta, gamma, delta_K, delta_G, tau, beta_discount = params
    V = np.zeros_like(K)
    for t in range(T-1):
        Y[t] = production(K[t], G[t], A[t])
        G[t+1] = (1 - delta_G) * G[t] + tau * Y[t]
        K[t+1] = Y[t] - C[t] + (1 - delta_K) * K[t]
        C[t] = 0.7 * Y[t]  # Example consumption rule
        V[t] = np.log(C[t]) + beta_discount * V[t+1]
    return -np.sum(V)  # Negative for minimization

# Calibration (example)
params = [alpha, beta, gamma, delta_K, delta_G, tau, beta_discount]
result = minimize(value_function_iteration, params, args=(K, G, A), method='BFGS')
calibrated_params = result.x

# Simulation with calibrated parameters
alpha_calib, beta_calib, gamma_calib, delta_K_calib, delta_G_calib, tau_calib, beta_discount_calib = calibrated_params

for t in range(T-1):
    Y[t] = production(K[t], G[t], A[t])
    G[t+1] = (1 - delta_G_calib) * G[t] + tau_calib * Y[t]
    K[t+1] = Y[t] - C[t] + (1 - delta_K_calib) * K[t]
    C[t] = 0.7 * Y[t]  # Example consumption rule

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(Y, label='Output')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(K, label='Private Capital')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(G, label='Infrastructure Capital')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(C, label='Consumption')
plt.legend()

plt.show()