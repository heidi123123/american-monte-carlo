import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql

# Set initial parameters for the option
np.random.seed(42)

S0 = 100  # Initial stock price
K = 100  # Strike price
T = 1.0  # Maturity in years
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility of the underlying stock
n_steps = 100  # Number of time steps
n_paths = 5000  # Number of Monte Carlo paths
n_exercise_dates = 4  # Number of exercise dates (Bermudan feature)

dt = T / n_steps  # Time step size


# Generate asset price paths using Geometric Brownian Motion (GBM)
def generate_asset_paths(S0, r, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return paths


# Calculate payoff for a Bermudan call option
def bermudan_payoff(S, K):
    return np.maximum(S - K, 0)


# Perform Least-Squares Monte Carlo (LSMC) to estimate option price
def LSMC_option_price(paths, K, r, dt, n_exercise_dates):
    n_paths, n_steps = paths.shape
    n_steps -= 1  # Adjust for the initial price at time 0
    exercise_dates = np.linspace(0, n_steps, n_exercise_dates + 1, dtype=int)[1:]  # Exclude time 0
    cash_flows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_steps)

    # Backward iteration: from the last exercise date backwards
    for t in reversed(exercise_dates):
        in_the_money = bermudan_payoff(paths[:, t], K) > 0  # ITM = exercise yes or no
        X = paths[in_the_money, t]  # stock prices for the ITM paths at time t
        Y = cash_flows[in_the_money] * np.exp(-r * dt * (exercise_times[in_the_money] - t))  # discounted ITM-CFs at time t

        if len(X) > 0:
            # Polynomial basis functions up to x^3 for regression to estimate continuation values for X
            A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T  # basis functions
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]  # least squares coefficients fitting Y against X
            continuation_estimated = A @ coeffs
            exercise_value = bermudan_payoff(X, K)
            exercise = exercise_value > continuation_estimated  # exercise or hold
            idx = np.where(in_the_money)[0][exercise]  # path index with optimal early exercise
            cash_flows[idx] = exercise_value[exercise]
            exercise_times[idx] = t

    # Discount cash flows back to present
    option_price = np.mean(cash_flows * np.exp(-r * dt * exercise_times))
    return option_price


# Nested Monte Carlo (inefficient, just for comparison)
def nested_mc_option_price(paths, K, r, sigma, T, dt, n_paths_inner, n_exercise_dates):
    n_paths, n_steps = paths.shape
    n_steps -= 1  # Adjust for the initial price at time 0
    exercise_dates = np.linspace(0, n_steps, n_exercise_dates + 1, dtype=int)[1:]  # Exclude time 0
    cash_flows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_steps)

    for t in reversed(exercise_dates):
        for i in range(n_paths):
            S = paths[i, t]
            exercise_value = bermudan_payoff(S, K)
            if exercise_value > 0:
                # Inner simulation for continuation value
                inner_paths = generate_asset_paths(S, r, sigma, dt * (n_steps - t), n_steps - t, n_paths_inner)
                inner_payoffs = bermudan_payoff(inner_paths[:, -1], K)
                continuation_value = np.mean(inner_payoffs) * np.exp(-r * dt * (n_steps - t))
                # Decide whether to exercise or continue
                if exercise_value > continuation_value:
                    cash_flows[i] = exercise_value
                    exercise_times[i] = t
    # Discount cash flows back to present
    option_price = np.mean(cash_flows * np.exp(-r * dt * exercise_times))
    return option_price


# Generate asset price paths
paths = generate_asset_paths(S0, r, sigma, T, n_steps, n_paths)

# Calculate option prices using LSMC and Nested Monte Carlo
lsmc_price = LSMC_option_price(paths, K, r, dt, n_exercise_dates)
nested_mc_price = nested_mc_option_price(paths, K, r, sigma, T, dt, n_paths_inner=100,
                                         n_exercise_dates=n_exercise_dates)

print(f"LSMC Price: {lsmc_price:.4f}")
print(f"Nested MC Price: {nested_mc_price:.4f}")
