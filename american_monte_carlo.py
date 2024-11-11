import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Set up the QuantLib exercise and engine based on exercise type and barrier level
def setup_exercise_and_engine(S0, r, T, sigma, n_steps, exercise_type="European", barrier_level=None):
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

    def american_exercise_and_engine():
        exercise = ql.AmericanExercise(today, today + int(T * 365))
        engine = ql.BinomialVanillaEngine(process, "crr", n_steps)
        return exercise, engine

    # Barrier option handling
    if barrier_level is not None:
        if exercise_type == "European":
            exercise = ql.EuropeanExercise(today + int(T * 365))
            engine = ql.AnalyticBarrierEngine(process)
        elif exercise_type == "American":
            exercise = ql.AmericanExercise(today, today + int(T * 365))
            engine = ql.BinomialBarrierEngine(process, "crr", n_steps)
        else:
            raise NotImplementedError("Barrier options with this exercise type are not implemented.")
        return exercise, engine

    exercise_map = {
        "European": european_exercise_and_engine,
        "American": american_exercise_and_engine,
    }

    return exercise_map[exercise_type]()


# Generate QuantLib option for comparison
def get_quantlib_option(S0, K, r, T, sigma, n_steps, option_type="Call", exercise_type="European", barrier_level=None):
    exercise, engine = setup_exercise_and_engine(S0, r, T, sigma, n_steps, exercise_type, barrier_level)
    option_type_ql = ql.Option.Put if option_type == "Put" else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type_ql, K)
    if barrier_level is not None:
        barrier_type = ql.Barrier.DownIn
        option = ql.BarrierOption(barrier_type, barrier_level, 0.0, payoff, exercise)
    else:
        option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option


# Generate Monte Carlo asset price paths using GBM
def generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths):
    dt = T / n_time_steps
    Z = np.random.normal(size=(n_paths, n_time_steps))
    growth_factor = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    increments = np.exp(growth_factor)

    paths = np.zeros((n_paths, n_time_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(increments, axis=1)
    return paths


# Calculate intrinsic value
def intrinsic_value(S, K, option_type="Call"):
    return np.maximum(K - S, 0) if option_type == "Put" else np.maximum(S - K, 0)


# Define exercise dates based on option type
def get_early_exercise_times(exercise_type, n_time_steps):
    if exercise_type == "European":
        return []
    elif exercise_type == "American":
        return np.arange(1, n_time_steps)


# Apply exercise if intrinsic value > continuation value
def apply_exercise(cashflows, exercise_times, in_the_money_idx, exercise_value, continuation_estimated, t):
    exercise = exercise_value > continuation_estimated
    selected_idx = in_the_money_idx[exercise]
    cashflows[selected_idx] = exercise_value[exercise]
    exercise_times[selected_idx] = t


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


# Estimate continuation values, applying regression onto asset paths
def estimate_continuation_values(paths, t, r, dt, cashflows, exercise_times, basis_type, degree):
    X = paths[:, t]
    Y = cashflows * np.exp(-r * dt * (exercise_times - t))

    if len(X) > 0:
        estimated_values = regression_estimate(X, Y, basis_type, degree)
        continuation_estimated = np.maximum(estimated_values, 0)
    else:
        continuation_estimated = np.zeros(paths.shape[0])
    return continuation_estimated


# Perform the backward iteration of American Monte Carlo procedure
def perform_backward_iteration(K, r, dt, n_time_steps, barrier_hit, cashflows, paths, option_type, exercise_times,
                               early_exercise_times, continuation_values, basis_type, degree):
    for t in reversed(range(n_time_steps + 1)):
        barrier_hit_t = barrier_hit[:, t]

        # Initialize continuation_estimated for this time step
        continuation_estimated = np.zeros(paths.shape[0])

        if t == n_time_steps:  # At maturity
            cashflows[barrier_hit_t] = intrinsic_value(paths[barrier_hit_t, t], K, option_type)
            exercise_times[barrier_hit_t] = t
        else:
            continuation_estimated = estimate_continuation_values(paths, t, r, dt, cashflows, exercise_times,
                                                                  basis_type, degree)

            if t in early_exercise_times:
                in_the_money = intrinsic_value(paths[:, t], K, option_type) > 0
                valid_paths = barrier_hit_t & in_the_money
                valid_paths_indices = np.where(valid_paths)[0]
                X = paths[valid_paths, t]
                exercise_value = intrinsic_value(X, K, option_type)
                estimated_continuation = continuation_estimated[valid_paths_indices]
                apply_exercise(cashflows, exercise_times, valid_paths_indices, exercise_value,
                               estimated_continuation, t)

        continuation_values.append((t, paths[:, t].copy(), continuation_estimated.copy()))

    # Reverse the stored values for correct time ordering
    continuation_values.reverse()


# Perform Least Squares Monte Carlo (LSMC) with visualization data
def lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level=None,
                        exercise_type="European", basis_type="Chebyshev", degree=4):
    n_paths, n_time_steps_plus_one = paths.shape
    n_time_steps = n_time_steps_plus_one - 1
    cashflows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_time_steps)

    # Precompute barrier hit matrix
    if barrier_level is not None:
        barrier_hit = np.maximum.accumulate(paths <= barrier_level, axis=1)
    else:
        barrier_hit = np.ones_like(paths, dtype=bool)

    early_exercise_times = get_early_exercise_times(exercise_type, n_time_steps)
    continuation_values = []

    perform_backward_iteration(K, r, dt, n_time_steps, barrier_hit, cashflows, paths, option_type, exercise_times,
                               early_exercise_times, continuation_values, basis_type, degree)

    # Calculate the discounted option price
    option_price = np.mean(cashflows * np.exp(-r * dt * exercise_times))
    return option_price, continuation_values


# Crop continuation_values to the first n_plotted_paths for plotting
def crop_data(continuation_values, paths, n_plotted_paths=10):
    cropped_continuation_values = [(t, stock_prices[:n_plotted_paths], continuation[:n_plotted_paths])
                                   for t, stock_prices, continuation in continuation_values]

    cropped_paths = paths[:n_plotted_paths]
    return cropped_continuation_values, cropped_paths


# Compute differences between continuation values and QuantLib prices
def compute_differences(continuation_values, dt, difference_type, K, r, T, sigma, n_time_steps, option_type,
                        exercise_type, barrier_level):
    differences = []
    for t, stock_prices, continuation_estimated in continuation_values:
        T_step = t * dt
        diffs = []
        for s, cont_value in zip(stock_prices, continuation_estimated):
            try:
                ql_option = get_quantlib_option(
                    S0=s, K=K, r=r, T=T - T_step, sigma=sigma,
                    n_steps=n_time_steps, option_type=option_type,
                    exercise_type=exercise_type, barrier_level=barrier_level
                )
                ql_price = ql_option.NPV()
            except RuntimeError:  # occurs when the barrier was knocked
                ql_option = get_quantlib_option(
                    S0=s, K=K, r=r, T=T - T_step, sigma=sigma,
                    n_steps=n_time_steps, option_type=option_type,
                    exercise_type=exercise_type
                )
                ql_price = ql_option.NPV()
            # Compute difference according to difference_type
            if difference_type == 'absolute':
                diff = abs(cont_value - ql_price)
            elif difference_type == 'difference':
                diff = cont_value - ql_price
            elif difference_type == 'relative':
                if abs(ql_price - cont_value) < 0.0001:
                    diff = 0
                elif ql_price != 0:
                    diff = (cont_value - ql_price) / ql_price
                else:  # handle division by zero
                    diff = (cont_value - 0.0001) / 0.0001
            else:
                raise ValueError(f"Invalid difference_type '{difference_type}'. Must be 'absolute', 'difference', or 'relative'.")
            diffs.append(diff)
        differences.append((t, stock_prices, np.array(diffs)))
    return differences


# Add text box with parameters to plot
def add_description_text_box(ax, S0, K, barrier_level):
    textstr = f"$S_0$ = {S0}\n$K$ = {K}\nBarrier = {barrier_level}"
    ax.text(0.05, 0.97, textstr, transform=ax.transAxes, fontsize=10, va='top', bbox=dict(facecolor='white'))


# Plot differences between LSMC and QuantLib prices
def plot_differences(differences, paths, dt, ax, title, vmin, vmax, key_S_lines, plot_asset_paths,
                     difference_type, S0, K, barrier_level, norm=None):
    if norm is None:
        if difference_type == "relative":
            norm = mcolors.SymLogNorm(linthresh=1e-2, linscale=1, vmin=vmin, vmax=vmax, base=10)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.Spectral
    time_steps = [t * dt for t in range(len(paths[0]))]

    if plot_asset_paths:
        for path in paths:
            ax.plot(time_steps, path, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    for t, stock_prices, diff_values in differences:
        T_step = t * dt
        if len(stock_prices) == len(diff_values):
            x_values = np.full(len(stock_prices), T_step)
            ax.scatter(x_values, stock_prices, c=diff_values, cmap=cmap, s=30, marker="o", norm=norm)

    ax.set_title(title)
    ax.set_xlabel("Time to Maturity (T)")
    if key_S_lines:
        for s_line in key_S_lines:
            ax.axhline(s_line, color="gray", linestyle="--", linewidth=0.8)
    for t_line in time_steps:
        ax.axvline(t_line, color="gray", linestyle="--", linewidth=0.5)

    add_description_text_box(ax, S0, K, barrier_level)


# Plot continuation values as a scatter plot
def plot_continuation_values(continuation_values, paths, dt, ax, title, vmin, vmax, key_S_lines, plot_asset_paths):
    cmap = cm.Spectral
    time_steps = [t * dt for t in range(len(paths[0]))]

    if plot_asset_paths:
        for path in paths:
            ax.plot(
                time_steps,
                path,
                color="gray",
                linestyle="-",
                linewidth=0.5,
                alpha=0.3,
            )

    for t, stock_prices, cont_values in continuation_values:
        T_step = t * dt
        if len(stock_prices) == len(cont_values):
            x_values = np.full(len(stock_prices), T_step)
            ax.scatter(
                x_values,
                stock_prices,
                c=cont_values,
                cmap=cmap,
                s=30,
                marker="o",
                vmin=vmin,
                vmax=vmax,
            )

    ax.set_title(title)
    ax.set_xlabel("Time to Maturity (T)")
    if key_S_lines:
        for s_line in key_S_lines:
            ax.axhline(s_line, color="gray", linestyle="--", linewidth=0.8)
    for t_line in time_steps:
        ax.axvline(t_line, color="gray", linestyle="--", linewidth=0.5)


# Plot the LSMC process with continuation values and differences
def plot_lsmc_results(continuation_values, paths, dt, K, r, T, sigma, n_time_steps, option_type,
                      exercise_type, barrier_level, S0, key_S_lines=None, plot_asset_paths=False,
                      difference_type="difference", vmin_diff=None, vmax_diff=None):
    # Compute differences
    differences = compute_differences(continuation_values, dt, difference_type, K, r, T, sigma, n_time_steps,
                                      option_type, exercise_type, barrier_level)

    # Determine color range for continuation values
    all_cont_values = np.concatenate([values for _, _, values in continuation_values])
    vmin_cont, vmax_cont = all_cont_values.min(), all_cont_values.max()

    # Determine vmin_diff and vmax_diff dynamically if not provided
    if vmin_diff is None or vmax_diff is None:
        all_diff_values = np.concatenate([values[~np.isnan(values)] for _, _, values in differences])
        if vmin_diff is None:
            vmin_diff = all_diff_values.min()
        if vmax_diff is None:
            vmax_diff = all_diff_values.max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    cmap = cm.Spectral

    # Create norm for differences with specified or dynamic vmin and vmax
    if difference_type == "relative":
        norm_diff = mcolors.SymLogNorm(linthresh=1e-2, linscale=1, vmin=vmin_diff, vmax=vmax_diff, base=10)
    else:
        norm_diff = mcolors.Normalize(vmin=vmin_diff, vmax=vmax_diff)

    plot_title = f"{difference_type.title()} Differences to QuantLib" \
        if difference_type != "difference" else "Differences to QuantLib"

    plot_differences(differences, paths, dt, axes[0], plot_title, vmin_diff, vmax_diff,
                     key_S_lines, plot_asset_paths, difference_type, S0, K, barrier_level, norm=norm_diff)

    plot_continuation_values(continuation_values, paths, dt, axes[1], "Continuation Values", vmin_cont, vmax_cont,
                             key_S_lines, plot_asset_paths)

    # Add color bar for differences
    sm_diff = cm.ScalarMappable(cmap=cmap, norm=norm_diff)
    sm_diff.set_array([])
    fig.colorbar(sm_diff, ax=axes[0], label=f"Differences to QuantLib")

    # Add color bar for continuation values
    norm_cont = mcolors.Normalize(vmin=vmin_cont, vmax=vmax_cont)
    sm_cont = cm.ScalarMappable(cmap=cmap, norm=norm_cont)
    sm_cont.set_array([])
    fig.colorbar(sm_cont, ax=axes[1], label="Continuation Value")

    plt.suptitle(f"LSMC Backward Iteration")
    plt.show()


# Main function to run LSMC and plot results
def main(params):
    # Unpack the parameters dictionary
    S0 = params['S0']
    K = params['K']
    T = params['T']
    r = params['r']
    sigma = params['sigma']
    n_time_steps = params['n_time_steps']
    n_paths = params['n_paths']
    dt = params['dt']
    option_type = params['option_type']
    exercise_type = params['exercise_type']
    n_plotted_paths = params['n_plotted_paths']
    barrier_level = params['barrier_level']
    basis_type = params['basis_type']
    degree = params['degree']
    difference_type = params['difference_type']
    vmin_diff = params['vmin_diff']
    vmax_diff = params['vmax_diff']

    # Generate asset paths
    paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)

    # Perform LSMC pricing
    lsmc_price, continuation_values = lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level,
                                                          exercise_type, basis_type, degree)

    # Crop data for plotting
    cont_values_cropped, paths_cropped = crop_data(continuation_values, paths, min(n_plotted_paths, n_paths))
    key_S_lines = [S0, K, barrier_level] if barrier_level else [S0, K]

    # Plot results
    plot_lsmc_results(
        cont_values_cropped, paths_cropped, dt, K, r, T, sigma, n_time_steps, option_type, exercise_type,
        barrier_level, S0, key_S_lines=key_S_lines, plot_asset_paths=False,
        difference_type=difference_type, vmin_diff=vmin_diff, vmax_diff=vmax_diff
    )

    # Compare LSMC with QuantLib
    quantlib_barrier_option = get_quantlib_option(S0, K, r, T, sigma, n_time_steps, option_type, exercise_type,
                                                  barrier_level)
    quantlib_option = get_quantlib_option(S0, K, r, T, sigma, n_time_steps, option_type, exercise_type)
    option_description = f"{exercise_type} {option_type}"
    barrier_text = f"with {barrier_level}" if barrier_level else "without"
    print(f"{option_description} Option Price {barrier_text} Barrier (LSMC): {lsmc_price:.4f}")
    print(f"{option_description} Option Price {barrier_text} Barrier (QuantLib): {quantlib_barrier_option.NPV():.4f}")
    if barrier_level:
        print(f"{option_description} Option Price without Barrier (QuantLib): {quantlib_option.NPV():.4f}")


if __name__ == "__main__":
    params = {
        "S0": 95,               # Initial stock price
        "K": 100,               # Strike price
        "T": 1.0,               # Maturity in years
        "r": 0.01,              # Risk-free rate
        "sigma": 0.2,           # Volatility of the underlying stock
        "n_time_steps": 100,    # Number of time steps (excluding S0)
        "n_paths": 1000,        # Number of Monte Carlo paths
        "dt": 1.0 / 100,        # Time step size
        "option_type": "Put",   # Option type
        "exercise_type": "European",  # Exercise type
        "n_plotted_paths": 1000,
        "barrier_level": 80,    # Barrier level
        "basis_type": "Chebyshev",
        "degree": 4,
        "difference_type": "difference",
        "vmin_diff": None,
        "vmax_diff": None
    }

    np.random.seed(42)
    main(params)
