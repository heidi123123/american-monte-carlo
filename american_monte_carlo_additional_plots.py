import matplotlib.pyplot as plt
import numpy as np
from american_monte_carlo import generate_asset_paths, get_quantlib_option, lsmc_option_pricing


def plot_asset_paths(paths, T, n_time_steps, n_paths_to_plot=100):
    # Plotting a subset of the generated paths
    plt.figure(figsize=(12, 6))
    time_steps = np.linspace(0, T, n_time_steps)  # Generate time steps for the x-axis

    # Plot the first n_paths_to_plot paths as an example
    for i in range(n_paths_to_plot):
        plt.plot(time_steps, paths[i], lw=1, alpha=0.7)

    plt.title("Simulated Asset Price Paths")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Asset Price")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_convergence_with_paths(S0, K, r, T, sigma, n_time_steps, option_type, exercise_type, barrier_level, path_range,
                                basis_type='Chebyshev', degree=4):
    lsmc_prices = []
    n_paths_list = path_range
    benchmark_option = get_quantlib_option(
        S0, K, r, T, sigma, n_time_steps, option_type, exercise_type, barrier_level
    )
    benchmark_price = benchmark_option.NPV()

    dt = T / n_time_steps
    for n_paths in n_paths_list:
        paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
        lsmc_price, _, _ = lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level, exercise_type,
                                               basis_type, degree)
        lsmc_prices.append(lsmc_price)

    plt.figure(figsize=(12, 8))
    plt.plot(n_paths_list, lsmc_prices, color='royalblue', marker='o', linestyle='-', linewidth=2, markersize=6,
             label='LSMC Estimated Price')

    # Benchmark line
    plt.axhline(benchmark_price, color='red', linestyle='--', linewidth=2, label='Benchmark Price (QuantLib)')

    # Plot styling
    plt.xlabel('Number of Paths', fontsize=12)
    plt.ylabel(f'{option_type} Option Price', fontsize=12)
    plt.title(f'Convergence of LSMC {option_type} Option Price with Number of Paths', fontsize=14, fontweight='bold')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_convergence_with_time_steps(S0, K, r, T, sigma, n_paths, option_type, exercise_type, barrier_level,
                                     time_step_range, basis_type='Chebyshev', degree=4):
    lsmc_prices = []

    # Calculate the benchmark option price using QuantLib using high resolution time grid
    high_res_steps = max(time_step_range) * 10  # 10x resolution
    benchmark_option = get_quantlib_option(S0, K, r, T, sigma, high_res_steps, option_type,
                                           exercise_type, barrier_level)
    benchmark_price = benchmark_option.NPV()

    for n_time_steps in time_step_range:
        dt = T / n_time_steps
        paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
        lsmc_price, _, _ = lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level, exercise_type,
                                               basis_type, degree)
        lsmc_prices.append(lsmc_price)

    plt.figure(figsize=(12, 8))
    plt.plot(time_step_range, lsmc_prices, color='royalblue', marker='o', linestyle='-', linewidth=2, markersize=6,
             label='LSMC Estimated Price')

    # Benchmark line
    plt.axhline(benchmark_price, color='red', linestyle='--', linewidth=2, label='Benchmark Price (QuantLib)')

    # Plot styling
    plt.xlabel('Number of Time Steps', fontsize=12)
    plt.ylabel(f'{option_type} Option Price', fontsize=12)
    plt.title(f'Convergence of LSMC {option_type} Option Price with Number of Time Steps', fontsize=14, fontweight='bold')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_error_heatmap(S0, K, r, T, sigma, time_step_range, path_range, option_type, exercise_type, barrier_level,
                       basis_type='Chebyshev', degree=4):
    # Initialize a matrix to hold the absolute error for each (time step, path) pair
    error_matrix = np.zeros((len(path_range), len(time_step_range)))

    # Calculate the benchmark option price using QuantLib using high resolution time grid
    high_res_steps = max(time_step_range) * 10  # Increase resolution for accuracy
    benchmark_option = get_quantlib_option(S0, K, r, T, sigma, high_res_steps, option_type, exercise_type, barrier_level)
    benchmark_price = benchmark_option.NPV()

    # Fill the error matrix with absolute errors
    for i, n_paths in enumerate(path_range):
        for j, n_time_steps in enumerate(time_step_range):
            dt = T / n_time_steps
            paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
            lsmc_price, _, _ = lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level, exercise_type,
                                                   basis_type=basis_type, degree=degree)
            # Store the absolute error in the matrix
            error_matrix[i, j] = abs(lsmc_price - benchmark_price)

    # Find the indices of the minimum error
    min_error_index = np.unravel_index(np.argmin(error_matrix, axis=None), error_matrix.shape)
    min_error_value = error_matrix[min_error_index]
    min_n_paths = path_range[min_error_index[0]]
    min_n_time_steps = time_step_range[min_error_index[1]]

    # Plot the heatmap of absolute errors
    plt.figure(figsize=(10, 8))
    c = plt.pcolormesh(time_step_range, path_range, error_matrix, shading='auto', cmap='viridis')
    plt.colorbar(c, label='Absolute Error')

    # Highlight the minimum combination of time steps and paths
    plt.scatter(min_n_time_steps, min_n_paths, color='red', s=200, edgecolor='black', marker='*',
                label=f'Minimum Absolute Error\nTimeSteps={min_n_time_steps}, Paths={min_n_paths}')

    # Set ticks to match the tested points and add grid lines
    plt.xticks(time_step_range, rotation=45)
    plt.yticks(path_range)
    plt.grid(visible=True, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Labels and title
    plt.xlabel("Number of Time Steps")
    plt.ylabel("Number of Paths (Log Scale)")
    plt.title(f"Absolute Error in {option_type} Option Price with {exercise_type} Exercise")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_basis_degree(S0, K, r, T, sigma, n_time_steps, n_paths, option_type, exercise_type, barrier_level,
                               max_degree):
    degrees = range(1, max_degree + 1)
    benchmark_option = get_quantlib_option(S0, K, r, T, sigma, n_time_steps, option_type, exercise_type, barrier_level)
    benchmark_price = benchmark_option.NPV()

    paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)

    plt.figure(figsize=(12, 8))
    color_map = {'Chebyshev': "royalblue", 'Power': "forestgreen", 'Legendre': "darkorange"}
    marker_map = {'Chebyshev': "o", 'Power': "s", 'Legendre': "^"}

    for basis_type in ["Chebyshev", "Power", "Legendre"]:
        lsmc_prices = []
        for degree in degrees:
            lsmc_price, _, _ = lsmc_option_pricing(paths, K, r, T / n_time_steps, option_type, barrier_level,
                                                   exercise_type, basis_type=basis_type, degree=degree)
            lsmc_prices.append(lsmc_price)

        plt.plot(degrees, lsmc_prices, label=f"{basis_type} Basis",
                 color=color_map[basis_type], marker=marker_map[basis_type],
                 linewidth=2, markersize=6)

    # Benchmark line
    plt.axhline(benchmark_price, color='red', linestyle='--', linewidth=2, label='Benchmark Price (QuantLib)')

    # Plot styling
    plt.xlabel("Degree of Polynomial Basis", fontsize=12)
    plt.ylabel(f"{option_type} Option Price", fontsize=12)
    plt.title(f"LSMC {option_type} Option Price vs. Degree of Polynomial Basis", fontsize=14, fontweight='bold')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    S0 = 100  # Initial stock price
    K = 100  # Strike price
    T = 1.0  # Maturity in years
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility of the underlying stock
    n_time_steps = 100  # Number of time steps for grid (resolution of simulation)
    n_paths = 1000  # Number of Monte Carlo paths
    dt = T / n_time_steps  # Time step size for simulation

    option_type = "Put"
    exercise_type = "American"
    n_plotted_paths = 6
    barrier_level = 0.8 * S0
    basis_type = "Chebyshev"
    degree = 4

    plot_error_vs_basis_degree(S0=S0, K=K, r=r, T=T, sigma=sigma, n_time_steps=n_time_steps, n_paths=n_paths,
                               option_type=option_type, exercise_type=exercise_type, barrier_level=barrier_level,
                               max_degree=10)

    path_range = [500, 1000, 3000, 5000, 7000, 10000]
    plot_convergence_with_paths(S0=S0, K=K, r=r, T=T, sigma=sigma, n_time_steps=n_time_steps,
                                option_type=option_type, exercise_type=exercise_type, barrier_level=barrier_level,
                                path_range=path_range, basis_type=basis_type, degree=degree)

    time_step_range = [5, 10, 50, 100, 150, 250]
    plot_convergence_with_time_steps(S0=S0, K=K, r=r, T=T, sigma=sigma, n_paths=n_paths,
                                     option_type=option_type, exercise_type=exercise_type, barrier_level=barrier_level,
                                     time_step_range=time_step_range, basis_type=basis_type, degree=degree)

    plot_error_heatmap(S0=S0, K=K, r=r, T=T, sigma=sigma, time_step_range=time_step_range, path_range=path_range,
                       option_type=option_type, exercise_type=exercise_type, barrier_level=barrier_level,
                       basis_type=basis_type, degree=degree)
