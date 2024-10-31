import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql
import matplotlib.cm as cm
import matplotlib.colors as mcolors


np.random.seed(42)


# Set up the QuantLib engine based on exercise type
def setup_exercise_and_engine(S0, K, r, T, sigma, n_steps, exercise_type="European", n_exercise_dates=1):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    process = ql.BlackScholesProcess(spot_handle, flat_ts, vol_ts)

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

    return exercise_map[exercise_type]()


# Generate QuantLib option for comparison
def get_quantlib_option(S0, K, r, T, sigma, n_steps, exercise_type="European", n_exercise_dates=1):
    exercise, engine = setup_exercise_and_engine(S0, K, r, T, sigma, n_steps, exercise_type, n_exercise_dates)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option


# Generate Monte Carlo asset price paths using GBM
def generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths):
    dt = T / n_time_steps
    paths = np.zeros((n_paths, n_time_steps))
    paths[:, 0] = S0
    for t in range(1, n_time_steps):
        Z = np.random.normal(0, 1, n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return paths


# Calculate intrinsic value for a call option
def intrinsic_value(S, K):
    return np.maximum(S - K, 0)


# Perform Least Squares Monte Carlo (LSMC) with visualization data
def lsmc_option_pricing(paths, K, r, dt, exercise_type="European", n_exercise_dates=1):
    n_paths, n_time_steps = paths.shape
    cash_flows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_time_steps - 1)

    # Set exercise dates based on exercise type
    exercise_dates = get_exercise_dates(exercise_type, n_time_steps, n_exercise_dates)
    option_values, continuation_values = [], []

    for t in reversed(range(n_time_steps)):
        if t == n_time_steps - 1:
            cash_flows = intrinsic_value(paths[:, t], K)
            exercise_times = np.full(n_paths, t)
            store_option_values(t, paths[:, t], cash_flows, option_values, continuation_values)
        elif t in exercise_dates:
            update_cash_flows(paths, t, K, r, dt, cash_flows, exercise_times, option_values, continuation_values)

    # Discount cash flows back to present
    option_price = np.mean(cash_flows * np.exp(-r * dt * exercise_times))
    option_values.reverse()
    continuation_values.reverse()
    return option_price, option_values, continuation_values


# Define exercise dates based on option type
def get_exercise_dates(exercise_type, n_time_steps, n_exercise_dates):
    if exercise_type == "European":
        return [n_time_steps - 1]
    elif exercise_type == "Bermudan":
        return np.linspace(0, n_time_steps - 1, n_exercise_dates + 1, dtype=int)[1:]
    elif exercise_type == "American":
        return np.arange(1, n_time_steps)


# Update cash flows based on regression of continuation values
def update_cash_flows(paths, t, K, r, dt, cash_flows, exercise_times, option_values, continuation_values):
    in_the_money = intrinsic_value(paths[:, t], K) > 0
    X, Y = paths[in_the_money, t], cash_flows[in_the_money] * np.exp(-r * dt * (exercise_times[in_the_money] - t))
    if len(X) > 0:
        continuation_estimated = regression_estimate(X, Y, basis_type, degree)
        apply_exercise(X, K, continuation_estimated, cash_flows, exercise_times, t, in_the_money)
        store_option_values(t, paths[:, t], cash_flows, option_values, continuation_values, continuation_estimated)


# Generate basis polynomials based on the selected basis type
def get_basis_polynomials(X, basis_type, degree):
    if basis_type == "Power":
        # Standard power basis: 1, X, X^2, ..., X^degree
        return np.column_stack([X ** i for i in range(degree + 1)])
    elif basis_type == "Chebyshev":
        # Chebyshev polynomials of the first kind: T0(X), T1(X), T2(X), ...
        return np.column_stack([np.polynomial.chebyshev.chebval(X, [0] * i + [1]) for i in range(degree + 1)])
    elif basis_type == "Legendre":
        # Legendre polynomials: P0(X), P1(X), P2(X), ...
        return np.column_stack([np.polynomial.legendre.legval(X, [0] * i + [1]) for i in range(degree + 1)])
    else:
        raise ValueError(f"Unknown basis type '{basis_type}'. Use 'Power', 'Chebyshev', or 'Legendre'.")


# Regression estimate of continuation value using specified basis functions
def regression_estimate(X, Y, basis_type="Power", degree=3):
    A = get_basis_polynomials(X, basis_type, degree)
    coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
    return A @ coeffs


# Apply exercise if intrinsic value > continuation value
def apply_exercise(X, K, continuation_estimated, cash_flows, exercise_times, t, in_the_money):
    exercise_value = intrinsic_value(X, K)
    exercise = exercise_value > continuation_estimated
    idx = np.where(in_the_money)[0][exercise]
    cash_flows[idx], exercise_times[idx] = exercise_value[exercise], t


# Store option and continuation values (optionally) during LSMC backward iteration
def store_option_values(t, stock_prices, cash_flows, option_values, continuation_values, continuation_estimated=None):
    option_values.append((t, stock_prices[:len(cash_flows)], cash_flows[:len(cash_flows)]))
    if continuation_estimated is not None:
        continuation_values.append((t, stock_prices[:len(continuation_estimated)], continuation_estimated))
    else:
        continuation_values.append((t, stock_prices[:len(cash_flows)], cash_flows[:len(cash_flows)]))


# Plot LSMC process with option and continuation values
def plot_lsmc_grid(option_values, continuation_values, paths, dt,
                   key_S_lines=None, plot_asset_paths=True, plot_values=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    cmap = cm.viridis
    vmin, vmax = get_color_range(option_values, continuation_values)

    # Plot option and continuation values with auxiliary grid and asset paths
    plot_value_scatter(option_values, paths, dt, axes[0], "Option Values", vmin, vmax,
                       key_S_lines, plot_asset_paths, plot_values)
    plot_value_scatter(continuation_values, paths, dt, axes[1], "Continuation Values", vmin, vmax,
                       key_S_lines, plot_asset_paths, plot_values)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Value")
    plt.suptitle("LSMC Backward Iteration: Option Values vs Continuation Values")
    plt.show()


# Helper function to determine color range across both option and continuation values
def get_color_range(option_values, continuation_values):
    all_option_values = np.concatenate([values for _, _, values in option_values])
    all_continuation_values = np.concatenate([values for _, _, values in continuation_values])
    return min(all_option_values.min(), all_continuation_values.min()), max(all_option_values.max(),
                                                                            all_continuation_values.max())


# Perform Least Squares Monte Carlo (LSMC) with visualization data
def lsmc_option_pricing(paths, K, r, dt, exercise_type="European", n_exercise_dates=1):
    n_paths, n_time_steps = paths.shape
    cash_flows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_time_steps - 1)

    # Set exercise dates based on exercise type
    exercise_dates = get_exercise_dates(exercise_type, n_time_steps, n_exercise_dates)
    option_values, continuation_values = [], []

    for t in reversed(range(n_time_steps)):
        if t == n_time_steps - 1:
            cash_flows = intrinsic_value(paths[:, t], K)
            exercise_times = np.full(n_paths, t)
        elif t in exercise_dates:
            update_cash_flows(paths, t, K, r, dt, cash_flows, exercise_times, option_values, continuation_values)

        # Store values at each timestep for visualization
        store_option_values(t, paths[:, t], cash_flows, option_values, continuation_values)

    # Discount cash flows back to present
    option_price = np.mean(cash_flows * np.exp(-r * dt * exercise_times))
    option_values.reverse()
    continuation_values.reverse()
    return option_price, option_values, continuation_values


# Plot value scatter plots with labels and gridlines
def plot_value_scatter(values, paths, dt, ax, title, vmin, vmax, key_S_lines, plot_asset_paths, plot_values):
    cmap = cm.viridis
    time_steps = [t * dt for t in range(len(paths[0]))]

    if plot_asset_paths:
        for path in paths:
            ax.plot(time_steps, path, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Iterate over time steps and plot option values
    for t, stock_prices, option_vals in values:
        T_step = t * dt
        # Expand stock_prices and option_vals to match the full size
        x_values = np.full(len(stock_prices), T_step)  # Same time step for all stock prices

        # Scatter plot with expanded values
        sc = ax.scatter(x_values, stock_prices, c=option_vals, cmap=cmap, s=30, marker='o', vmin=vmin, vmax=vmax)

        if plot_values:
            for s, v in zip(stock_prices, option_vals):
                ax.text(T_step, s, f"{v:.2f}", ha='center', va='center', fontsize=6, color="black",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

    ax.set_title(title)
    ax.set_xlabel("Time to Maturity (T)")
    if key_S_lines:
        for s in key_S_lines:
            ax.axhline(s, color='gray', linestyle='-', linewidth=0.8)
    for t in time_steps:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)


# Main function to run LSMC and plot results
def main():
    paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
    lsmc_price, option_values, continuation_values = lsmc_option_pricing(paths, K, r, dt, exercise_type,
                                                                         n_exercise_dates)
    if plot:
        plot_lsmc_grid(option_values, continuation_values, paths, dt, key_S_lines=[S0, K], plot_values=True)

    # Compare LSMC with QuantLib
    quantlib_option = get_quantlib_option(S0, K, r, T, sigma, n_time_steps, exercise_type, n_exercise_dates)
    print(f"{exercise_type} Option Price (LSMC): {lsmc_price:.4f}")
    print(f"{exercise_type} Option Price (QuantLib): {quantlib_option.NPV():.4f}")


if __name__ == "__main__":
    S0 = 100  # Initial stock price
    K = 100  # Strike price
    T = 1.0  # Maturity in years
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility of the underlying stock
    n_time_steps = 4  # Number of time steps for grid (resolution of simulation)
    n_paths = 5  # Number of Monte Carlo paths
    dt = T / n_time_steps  # Time step size for simulation

    exercise_type = "American"
    n_exercise_dates = 4  # Number of exercise dates (Bermudan feature)
    plot = True

    basis_type = "Chebyshev"
    degree = 4
    main()
