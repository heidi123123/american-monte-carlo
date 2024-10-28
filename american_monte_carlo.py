import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
import QuantLib as ql
import matplotlib.cm as cm
import matplotlib.colors as mcolors


S0 = 100  # Initial stock price
K = 100  # Strike price
T = 1.0  # Maturity in years
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility of the underlying stock
n_time_steps = 19  # Number of time steps for grid (resolution of simulation)
n_paths = 10  # Number of Monte Carlo paths
dt = T / n_time_steps  # Time step size for simulation

exercise_type = "American"
n_exercise_dates = 4  # Number of exercise dates (Bermudan feature)

plot = True


# Generate QuantLib object for comparison
def get_quantlib_option(S0, K, r, T, sigma, n_steps, exercise_type="European", n_exercise_dates=1):
    # QuantLib Setup
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Exercise and engine setup functions
    def european_exercise_and_engine():
        exercise = ql.EuropeanExercise(today + int(T * 365))
        engine = ql.AnalyticEuropeanEngine(process)
        return exercise, engine

    def bermudan_exercise_and_engine():
        exercise_dates = [today + int(t * 365 * T / n_steps)
                          for t in np.linspace(1, n_steps, n_exercise_dates, dtype=int)]
        exercise = ql.BermudanExercise(exercise_dates)
        engine = ql.BinomialVanillaEngine(process, "crr", n_steps)
        return exercise, engine

    def american_exercise_and_engine():
        exercise = ql.AmericanExercise(today, today + int(T * 365))
        engine = ql.BinomialVanillaEngine(process, "crr", n_steps)
        return exercise, engine

    # Map exercise styles to their respective exercise and engine functions
    exercise_map = {
        "European": european_exercise_and_engine,
        "Bermudan": bermudan_exercise_and_engine,
        "American": american_exercise_and_engine
    }

    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))

    # Black-Scholes process
    process = ql.BlackScholesProcess(spot_handle, flat_ts, vol_ts)

    # Retrieve the selected exercise and engine
    exercise, engine = exercise_map.get(exercise_type, american_exercise_and_engine)()

    # Payoff
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)

    # Construct the option
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    return option


# Generate asset price paths using Geometric Brownian Motion (GBM)
def generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths):
    dt = T / n_time_steps
    paths = np.zeros((n_paths, n_time_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_time_steps + 1):
        Z = np.random.normal(0, 1, n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return paths


# Calculate payoff for a call option
def call_payoff(S, K):
    return np.maximum(S - K, 0)


def lsmc_option_price_with_visualization(paths, K, r, dt, exercise_type="European", n_exercise_dates=1):
    n_paths, n_time_steps = paths.shape
    n_time_steps -= 1  # Adjust for initial price at time 0
    cash_flows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_time_steps)

    # Set exercise dates based on exercise type
    if exercise_type == "European":
        exercise_dates = [n_time_steps]  # Only at expiration
    elif exercise_type == "Bermudan":
        exercise_dates = np.linspace(0, n_time_steps, n_exercise_dates + 1, dtype=int)[1:]  # Exclude time 0
    elif exercise_type == "American":
        exercise_dates = np.arange(1, n_time_steps + 1)  # Every time step from 1 to n_time_steps

    # For storing option values and continuation values for visualization
    option_values_triangle = []
    continuation_values_triangle = []

    # Backward iteration through time steps
    for t in reversed(range(n_time_steps + 1)):
        if t == n_time_steps:  # At expiration, option value is intrinsic value
            cash_flows = call_payoff(paths[:, t], K)
            exercise_times = np.full(n_paths, t)
            option_values_triangle.append((t, paths[:, t], cash_flows))
            continuation_values_triangle.append((t, paths[:, t], cash_flows))
        elif t in exercise_dates:
            # In-the-money paths where exercise might be considered
            in_the_money = call_payoff(paths[:, t], K) > 0
            X = paths[in_the_money, t]
            Y = cash_flows[in_the_money] * np.exp(-r * dt * (exercise_times[in_the_money] - t))

            if len(X) > 0:
                # Regression to estimate continuation value
                A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T
                coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
                continuation_estimated = A @ coeffs

                # Determine option values based on exercise vs. continuation
                exercise_value = call_payoff(X, K)
                exercise = exercise_value > continuation_estimated
                idx = np.where(in_the_money)[0][exercise]
                cash_flows[idx] = exercise_value[exercise]
                exercise_times[idx] = t

                # Store for visualization
                option_values_triangle.append((t, paths[:, t], cash_flows))
                continuation_values = np.zeros(n_paths)
                continuation_values[in_the_money] = continuation_estimated
                continuation_values_triangle.append((t, paths[:, t], continuation_values))

    # Discount cash flows back to present
    option_price = np.mean(cash_flows * np.exp(-r * dt * exercise_times))
    option_values_triangle.reverse()  # For plotting from t=0 onward
    continuation_values_triangle.reverse()
    return option_price, option_values_triangle, continuation_values_triangle


# Plot the grid for LSMC process with both option and continuation values in separate subplots
def plot_lsmc_grid(option_values_triangle, continuation_values_triangle, paths, dt, n_time_steps,
                   key_S_lines=None, plot_asset_paths=True, plot_values=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    cmap = cm.viridis

    # Calculate the global min and max values across all option and continuation values
    all_option_values = np.concatenate([values for _, _, values in option_values_triangle])
    all_continuation_values = np.concatenate([values for _, _, values in continuation_values_triangle])
    vmin = min(all_option_values.min(), all_continuation_values.min())
    vmax = max(all_option_values.max(), all_continuation_values.max())

    # Calculate unique time points based on n_time_steps
    time_steps = [t * dt for t in range(n_time_steps + 1)]

    # Option Values Plot with asset paths and grid lines
    ax = axes[0]
    if plot_asset_paths:
        # Plot asset paths as faint lines
        for path in paths:
            ax.plot(time_steps, path, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    # Scatter option values with labels
    for t, stock_prices, option_values in option_values_triangle:
        T_step = t * dt
        ax.scatter([T_step] * len(stock_prices), stock_prices, c=option_values, cmap=cmap, s=30, marker='o', vmin=vmin, vmax=vmax)
        if plot_values:
            # Add labels for each point
            for i, (s, v) in enumerate(zip(stock_prices, option_values)):
                ax.text(T_step, s, f"{v:.2f}", ha='center', va='center', fontsize=6, color="black",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
    ax.set_title("Option Values")
    ax.set_xlabel("Time to Maturity (T)")
    ax.set_ylabel("Stock Price (S)")

    # Add vertical lines for each time step
    for t in time_steps:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)

    # Optionally add horizontal lines for specified key stock prices
    if key_S_lines:
        for s in key_S_lines:
            ax.axhline(s, color='gray', linestyle='-', linewidth=0.8)

    # Continuation Values Plot with asset paths and grid lines
    ax = axes[1]
    if plot_asset_paths:
        # Plot asset paths as faint lines
        for path in paths:
            ax.plot(time_steps, path, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    # Scatter continuation values with labels
    for t, stock_prices, continuation_values in continuation_values_triangle:
        T_step = t * dt
        ax.scatter([T_step] * len(stock_prices), stock_prices, c=continuation_values, cmap=cmap, s=30, marker='x', vmin=vmin, vmax=vmax)
        if plot_values:
            # Add labels for each point
            for i, (s, v) in enumerate(zip(stock_prices, continuation_values)):
                ax.text(T_step, s, f"{v:.2f}", ha='center', va='center', fontsize=6, color="black",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
    ax.set_title("Continuation Values")
    ax.set_xlabel("Time to Maturity (T)")

    # Add vertical lines for each time step
    for t in time_steps:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)

    # Optionally add horizontal lines for specified key stock prices
    if key_S_lines:
        for s in key_S_lines:
            ax.axhline(s, color='gray', linestyle='-', linewidth=0.8)

    # Add colorbar with the correct value range
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Required for matplotlib colorbar
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Value")

    plt.suptitle("LSMC Backward Iteration")
    plt.show()


# Generate MC asset price paths
paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
# Run LSMC with backward iteration visualization on triangular grid
lsmc_price, option_values_triangle, continuation_values_triangle = lsmc_option_price_with_visualization(
    paths, K, r, dt, exercise_type=exercise_type, n_exercise_dates=n_exercise_dates)
# Plot the grid to visualize the LSMC process
if plot:
    plot_lsmc_grid(option_values_triangle, continuation_values_triangle, paths, dt, n_time_steps, key_S_lines=[S0, K])

# Compare final option prices from LSMC with QuantLib
print(f"{exercise_type} Option Price (LSMC): {lsmc_price:.4f}")
ql_option = get_quantlib_option(S0, K, r, T, sigma, n_time_steps,
                                exercise_type=exercise_type, n_exercise_dates=n_exercise_dates)
print(f"{exercise_type} Option Price (QuantLib): {ql_option.NPV():.4f}")
