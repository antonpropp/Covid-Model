import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib

# Use TkAgg to open the plot in a separate window
matplotlib.use('TkAgg')

# Load the CSV file with the correct path
file_path = 'SIRD_Russia.csv'
df = pd.read_csv(file_path)

# Prepare data and time for optimization
t = df['Day'].values
data = df[['Susceptible', 'Infected', 'Recovered', 'Deaths']].values

# Initial values for the model based on dataset values
N = 145911426  # Total population, approximate
I0 = df['Infected'].iloc[0]
R0 = df['Recovered'].iloc[0]
D0 = df['Deaths'].iloc[0]
S0 = N - I0 - R0 - D0

# SIRD model equations
def deriv(y, t, N, beta, gamma, mu):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return dSdt, dIdt, dRdt, dDdt

# Function to numerically solve SIRD
def solve_sird(params, t, S0, I0, R0, D0, N):
    beta, gamma, mu = params
    y0 = S0, I0, R0, D0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, mu))
    return ret.T

# Error function for optimization
def error(params, t, data, S0, I0, R0, D0, N):
    S, I, R, D = solve_sird(params, t, S0, I0, R0, D0, N)
    mse_s = mean_squared_error(data[:, 0], S)
    mse_i = mean_squared_error(data[:, 1], I)
    mse_r = mean_squared_error(data[:, 2], R)
    mse_d = mean_squared_error(data[:, 3], D)
    return mse_s + mse_i + mse_r + mse_d

# Define parameter bounds for optimization
bounds = [(0.001, 1), (1/20, 1/5), (0.001, 0.1)]

# Optimize model parameters using differential evolution
result = differential_evolution(error, bounds, args=(t, data, S0, I0, R0, D0, N), maxiter=1000)
beta_opt, gamma_opt, mu_opt = result.x

# Output optimal parameters
print(f'Optimal values: β={beta_opt:.4f}, γ={gamma_opt:.4f}, μ={mu_opt:.4f}')

# Solve the model with optimal parameters
S, I, R, D = solve_sird([beta_opt, gamma_opt, mu_opt], t, S0, I0, R0, D0, N)

# Calculate metrics
mse = mean_squared_error(data, np.column_stack((S, I, R, D)))
mae = mean_absolute_error(data, np.column_stack((S, I, R, D)))
r2 = r2_score(data, np.column_stack((S, I, R, D)))

# Output metrics
print(f'MSE (Mean Squared Error): {mse:.4f}')
print(f'MAE (Mean Absolute Error): {mae:.4f}')
print(f'R² (Coefficient of Determination): {r2:.4f}')

# Plotting
plt.figure(figsize=(12, 8))

# Actual data plot
plt.plot(t, data[:, 0], 'b-', alpha=0.5, lw=2, label='Data S (Susceptible)')
plt.plot(t, data[:, 1], 'r-', alpha=0.5, lw=2, label='Data I (Infected)')
plt.plot(t, data[:, 2], 'g-', alpha=0.5, lw=2, label='Data R (Recovered)')
plt.plot(t, data[:, 3], 'k-', alpha=0.5, lw=2, label='Data D (Deaths)')

# SIRD model plot
plt.plot(t, S, 'b--', lw=2, label='Model S (Susceptible)')
plt.plot(t, I, 'r--', lw=2, label='Model I (Infected)')
plt.plot(t, R, 'g--', lw=2, label='Model R (Recovered)')
plt.plot(t, D, 'k--', lw=2, label='Model D (Deaths)')

# Plot settings
plt.xlabel('Day')
plt.ylabel('Count')
plt.title('Comparison of Data and SIRD Model with Optimization')
plt.legend()

# Open the plot in a separate window
plt.show(block=True)
