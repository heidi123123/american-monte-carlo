import numpy as np
import matplotlib.pyplot as plt
import QuantLib as Ql
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Set up the QuantLib engine based on exercise type
def setup_exercise_and_engine(S0, r, T, sigma, n_steps, exercise_type="European", barrier_level=None):
    calendar = Ql.NullCalendar()
    day_count = Ql.Actual365Fixed()
    today = Ql.Date().todaysDate()
    Ql.Settings.instance().evaluationDate = today

    spot_handle = Ql.QuoteHandle(Ql.SimpleQuote(S0))
    flat_ts = Ql.YieldTermStructureHandle(Ql.FlatForward(today, r, day_count))
    vol_ts = Ql.BlackVolTermStructureHandle(Ql.BlackConstantVol(today, calendar, sigma, day_count))
    process = Ql.BlackScholesProcess(spot_handle, flat_ts, vol_ts)

    # Exercise and engine setup functions
    def european_exercise_and_engine():
        exercise = Ql.EuropeanExercise(today + int(T * 365))
        engine = Ql.AnalyticEuropeanEngine(process)
        return exercise, engine

    def american_exercise_and_engine():
        exercise = Ql.AmericanExercise(today, today + int(T * 365))
        engine = Ql.BinomialVanillaEngine(process, "crr", n_steps)
        return exercise, engine

    # Barrier option handling
    if barrier_level is not None:
        # Use the specified exercise type
        if exercise_type == "European":
            exercise = Ql.EuropeanExercise(today + int(T * 365))
            engine = Ql.AnalyticBarrierEngine(process)
        elif exercise_type == "American":
            exercise = Ql.AmericanExercise(today, today + int(T * 365))
            engine = Ql.BinomialBarrierEngine(process, "crr", n_steps)
        else:
            raise NotImplementedError("Barrier options with this exercise type are not implemented.")
        return exercise, engine

    exercise_map = {
        "European": european_exercise_and_engine,
        "American": american_exercise_and_engine
    }

    return exercise_map[exercise_type]()


# Generate QuantLib option for comparison
def get_quantlib_option(S0, K, r, T, sigma, n_steps, option_type="Call", exercise_type="European", barrier_level=None):
    exercise, engine = setup_exercise_and_engine(S0, r, T, sigma, n_steps, exercise_type, barrier_level)
    option_type = Ql.Option.Put if option_type == "Put" else Ql.Option.Call
    payoff = Ql.PlainVanillaPayoff(option_type, K)
    if barrier_level is not None:
        barrier_type = Ql.Barrier.DownIn
        option = Ql.BarrierOption(barrier_type, barrier_level, 0.0, payoff, exercise)
    else:
        option = Ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option


# Generate Monte Carlo asset price paths using GBM
def generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths):
    dt = T / n_time_steps
    Z = np.random.normal(size=(n_paths, n_time_steps))
    growth_factor = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    increments = np.exp(growth_factor)

    # Initialize paths array with shape (n_paths, n_time_steps + 1)
    paths = np.zeros((n_paths, n_time_steps + 1))
    paths[:, 0] = S0  # Set initial asset price at time zero

    # Calculate asset prices for each time step
    paths[:, 1:] = S0 * np.cumprod(increments, axis=1)
    return paths


# Calculate intrinsic value
def intrinsic_value(S, K, option_type="Call"):
    return np.maximum(K - S, 0) if option_type == "Put" else np.maximum(S - K, 0)


# Define exercise dates based on option type
def get_early_exercise_dates(exercise_type, n_time_steps):
    if exercise_type == "European":
        return []
    elif exercise_type == "American":
        return np.arange(1, n_time_steps)


# Update cashflows based on regression of continuation values
def update_cashflows(paths, t, K, r, dt, cashflows, exercise_times, option_values, continuation_values, option_type,
                     barrier_hit_t, basis_type, degree):
    in_the_money = intrinsic_value(paths[:, t], K, option_type) > 0
    valid_paths = barrier_hit_t & in_the_money
    valid_paths_indices = np.where(valid_paths)[0]
    X = paths[valid_paths, t]
    Y = cashflows[valid_paths] * np.exp(-r * dt * (exercise_times[valid_paths] - t))

    if len(X) > 0:
        continuation_estimated = regression_estimate(X, Y, basis_type, degree)
        exercise_value = intrinsic_value(X, K, option_type)
        apply_exercise(cashflows, exercise_times, valid_paths_indices, exercise_value, continuation_estimated, t)
        store_option_values(t, paths[:, t], cashflows, option_values, continuation_values,
                            continuation_estimated, valid_paths_indices)
    else:
        # No in-the-money or knocked-in paths --> store zeros
        store_option_values(t, paths[:, t], cashflows, option_values, continuation_values)


# Generate basis polynomials based on the selected basis type
def get_basis_polynomials(X, basis_type, degree):
    basis_func_map = {"Power": lambda X, i: X ** i,
                      "Chebyshev": lambda X, i: np.polynomial.chebyshev.chebval(X, [0] * i + [1]),
                      "Legendre": lambda X, i: np.polynomial.legendre.legval(X, [0] * i + [1])}

    if basis_type not in basis_func_map:
        raise ValueError(f"Unknown basis type '{basis_type}'. Use 'Power', 'Chebyshev', or 'Legendre'.")

    return np.column_stack([basis_func_map[basis_type](X, i) for i in range(degree + 1)])


# Regression estimate of continuation value using specified basis functions
def regression_estimate(X, Y, basis_type="Power", degree=3):
    A = get_basis_polynomials(X, basis_type, degree)
    coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
    return A @ coeffs


# Apply exercise if intrinsic value > continuation value
def apply_exercise(cashflows, exercise_times, in_the_money_idx, exercise_value, continuation_estimated, t):
    exercise = exercise_value > continuation_estimated
    selected_idx = in_the_money_idx[exercise]
    cashflows[selected_idx], exercise_times[selected_idx] = exercise_value[exercise], t


# Store option and continuation values during LSMC backward iteration
def store_option_values(t, stock_prices, cashflows, option_values, continuation_values, continuation_estimated=None):
    option_values.append((t, stock_prices.copy(), cashflows.copy()))
    if continuation_estimated is not None:
        continuation_values.append((t, stock_prices.copy(), continuation_estimated.copy()))
    else:
        continuation_values.append((t, stock_prices.copy(), np.zeros_like(stock_prices)))


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


# Function to update barrier_hit flags
def check_barrier_hit(paths, barrier_level, barrier_hit, t):
    new_breaches = paths[:, t] <= barrier_level
    barrier_hit |= new_breaches
    return barrier_hit


# Estimate continuation values, applying regression onto asset paths
def estimate_continuation_values(paths, t, r, dt, cashflows, exercise_times, basis_type, degree):
    X = paths[:, t]
    Y = cashflows * np.exp(-r * dt * (exercise_times - t))

    continuation_estimated = np.zeros(paths.shape[0])
    if len(X) > 0:
        estimated_values = regression_estimate(X, Y, basis_type, degree)
        estimated_values = np.maximum(estimated_values, 0)  # floor continuation values at zero
        continuation_estimated = estimated_values
    return continuation_estimated


# Perform Least Squares Monte Carlo (LSMC) with visualization data
def lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level=None,
                        exercise_type="European", basis_type="Chebyshev", degree=3):
    n_paths, n_time_steps_plus_one = paths.shape
    n_time_steps = n_time_steps_plus_one - 1
    cashflows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_time_steps)

    # Precompute barrier hit matrix
    if barrier_level is not None:
        barrier_hit = np.maximum.accumulate(paths <= barrier_level, axis=1)
    else:
        barrier_hit = np.ones_like(paths, dtype=bool)

    # Set early exercise dates
    early_exercise_dates = get_early_exercise_dates(exercise_type, n_time_steps)
    option_values, continuation_values = [], []

    for t in reversed(range(n_time_steps + 1)):
        barrier_hit_t = barrier_hit[:, t]

        # Initialize continuation_estimated for this time step
        continuation_estimated = np.zeros(n_paths)

        if t == n_time_steps:  # At maturity
            cashflows[barrier_hit_t] = intrinsic_value(paths[barrier_hit_t, t], K, option_type)
            exercise_times[barrier_hit_t] = t
        else:
            # Estimate continuation values at every time step
            continuation_estimated = estimate_continuation_values(paths, t, r, dt, cashflows, exercise_times,
                                                                  basis_type, degree)

            # Apply exercise decision only at exercise dates
            if t in early_exercise_dates:
                in_the_money = intrinsic_value(paths[:, t], K, option_type) > 0
                valid_paths = barrier_hit_t & in_the_money
                valid_paths_indices = np.where(valid_paths)[0]
                X = paths[valid_paths, t]
                exercise_value = intrinsic_value(X, K, option_type)
                estimated_continuation = continuation_estimated[valid_paths_indices]
                apply_exercise(cashflows, exercise_times, valid_paths_indices, exercise_value,
                               estimated_continuation, t)

        # Store option and continuation values for plotting
        store_option_values(t, paths[:, t], cashflows, option_values, continuation_values,
                            continuation_estimated)

    # Reverse the stored values for correct time ordering
    option_values.reverse()
    continuation_values.reverse()

    # Calculate the discounted option price
    option_price = np.mean(cashflows * np.exp(-r * dt * exercise_times))
    return option_price, option_values, continuation_values


# Plot annotations of option / continuation values
def plot_annotated_option_values(stock_prices, option_vals, T_step, time_steps, ax):
    for s, v in zip(stock_prices, option_vals):
        try:
            # Dynamically calculate QuantLib option price at each step
            Ql_option = get_quantlib_option(S0=s, K=K, r=r, T=T - T_step, sigma=sigma, n_steps=len(time_steps),
                                            option_type=option_type, exercise_type=exercise_type,
                                            barrier_level=barrier_level)
            Ql_price = Ql_option.NPV()
        except RuntimeError:
            Ql_option = get_quantlib_option(S0=s, K=K, r=r, T=T - T_step, sigma=sigma, n_steps=len(time_steps),
                                            option_type=option_type, exercise_type=exercise_type)
            Ql_price = Ql_option.NPV()

        # Annotate LSMC and QuantLib prices on the plot
        ax.annotate(f"{v:.2f}", (T_step, s), ha='right', va='bottom', fontsize=6, color="black", rotation=30)
        ax.annotate(f"{Ql_price:.2f}", (T_step, s), ha='right', va='top', fontsize=6, color="red", rotation=30)

    # Legend via text boxes
    ax.text(0.02, 0.98, "LSMC Prices", transform=ax.transAxes,
            fontsize=7, color="black", verticalalignment='top')
    ax.text(0.02, 0.96, "QuantLib Prices", transform=ax.transAxes,
            fontsize=7, color="red", verticalalignment='top')


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
        if len(stock_prices) == len(option_vals):
            x_values = np.full(len(stock_prices), T_step)
            ax.scatter(x_values, stock_prices, c=option_vals, cmap=cmap, s=30, marker='o', vmin=vmin, vmax=vmax)
            if plot_values:
                plot_annotated_option_values(stock_prices, option_vals, T_step, time_steps, ax)

    ax.set_title(title)
    ax.set_xlabel("Time to Maturity (T)")
    if key_S_lines:
        for s in key_S_lines:
            ax.axhline(s, color='gray', linestyle='-', linewidth=0.8)
    for t in time_steps:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.5)


# Crop option_values and continuation_values to the first n_plotted_paths
def crop_data(option_values, continuation_values, paths, n_plotted_paths=10):
    cropped_option_values = [(t, stock_prices[:n_plotted_paths], cashflows[:n_plotted_paths])
                             for t, stock_prices, cashflows in option_values]

    cropped_continuation_values = [(t, stock_prices[:n_plotted_paths], continuation[:n_plotted_paths])
                                   for t, stock_prices, continuation in continuation_values]

    cropped_paths = paths[:n_plotted_paths]
    return cropped_option_values, cropped_continuation_values, cropped_paths


# Main function to run LSMC and plot results
def main():
    paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)

    lsmc_price, option_values, continuation_values = lsmc_option_pricing(
        paths, K, r, dt, option_type, barrier_level, exercise_type, basis_type, degree
    )

    option_values, continuation_values, paths_cropped = crop_data(
        option_values, continuation_values, paths, min(n_plotted_paths, n_paths)
    )
    plot_lsmc_grid(option_values, continuation_values, paths_cropped, dt, key_S_lines=[S0, K], plot_values=plot_values)

    # Compare LSMC with QuantLib
    quantlib_barrier_option = get_quantlib_option(
        S0, K, r, T, sigma, n_time_steps, option_type, exercise_type, barrier_level
    )
    quantlib_option = get_quantlib_option(
        S0, K, r, T, sigma, n_time_steps, option_type, exercise_type
    )
    barrier_pct = barrier_level / S0 * 100 if barrier_level else None
    option_description = f"{exercise_type} {option_type}"
    print(f"{option_description} Option Price with {barrier_pct}% Barrier (LSMC): {lsmc_price:.4f}")
    print(f"{option_description} Option Price with {barrier_pct}% Barrier (QuantLib): {quantlib_barrier_option.NPV():.4f}")
    print(f"{option_description} Option Price without Barrier (QuantLib): {quantlib_option.NPV():.4f}")


if __name__ == "__main__":
    np.random.seed(42)
    S0 = 100  # Initial stock price
    K = 100  # Strike price
    T = 1.0  # Maturity in years
    r = 0.01  # Risk-free rate
    sigma = 0.2  # Volatility of the underlying stock
    n_time_steps = 4  # Number of time steps for grid (NOT including S_0)
    n_paths = 10000  # Number of Monte Carlo paths
    dt = T / n_time_steps  # Time step size for simulation

    option_type = "Put"
    exercise_type = "European"
    n_plotted_paths = 6
    barrier_level = 0.8 * S0
    basis_type = "Chebyshev"
    degree = 4

    plot_values = True

    main()
