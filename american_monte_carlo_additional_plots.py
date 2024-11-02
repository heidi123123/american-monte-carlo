import numpy as np
import matplotlib as plt
from american_monte_carlo import *


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
    plt.show()


def plot_convergence_with_paths(S0, K, r, T, sigma, n_time_steps, option_type, exercise_type, path_range):
    lsmc_prices = []
    n_paths_list = path_range
    benchmark_option = get_quantlib_option(
        S0, K, r, T, sigma, n_time_steps, option_type, exercise_type
    )
    benchmark_price = benchmark_option.NPV()

    dt = T / n_time_steps
    for n_paths in n_paths_list:
        paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
        lsmc_price, _, _ = lsmc_option_pricing(
            paths, K, r, dt, option_type, exercise_type=exercise_type, basis_type='Chebyshev', degree=4
        )
        lsmc_prices.append(lsmc_price)

    plt.figure(figsize=(10, 6))
    plt.plot(n_paths_list, lsmc_prices, label='LSMC Estimated Price', marker='o')
    plt.axhline(benchmark_price, color='red', linestyle='--', label='Benchmark Price (QuantLib)')
    plt.xlabel('Number of Paths')
    plt.ylabel('Option Price')
    plt.title('Convergence of LSMC Option Price with Number of Paths')
    plt.legend()
    plt.show()


def plot_error_vs_basis_degree(S0, K, r, T, sigma, n_time_steps, n_paths, option_type, exercise_type, max_degree):
    degrees = range(1, max_degree + 1)
    benchmark_option = get_quantlib_option(
        S0, K, r, T, sigma, n_time_steps, option_type, exercise_type)
    benchmark_price = benchmark_option.NPV()

    paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)

    plt.figure(figsize=(12, 8))
    color_map = {'Chebyshev': "royalblue", 'Power': "forestgreen", 'Legendre': "darkorange"}
    markers = {'Chebyshev': "o", 'Power': "s", 'Legendre': "^"}

    for basis_type in ["Chebyshev", "Power", "Legendre"]:
        lsmc_prices = []
        for degree in degrees:
            lsmc_price, _, _ = lsmc_option_pricing(
                paths, K, r, T / n_time_steps, option_type,
                exercise_type=exercise_type, basis_type=basis_type, degree=degree
            )
            lsmc_prices.append(lsmc_price)

        plt.plot(degrees, lsmc_prices, label=f"{basis_type} Basis",
                 color=color_map[basis_type], marker=markers[basis_type],
                 linewidth=2, markersize=6)

    # Benchmark line
    plt.axhline(benchmark_price, color='red', linestyle='--', linewidth=2, label='Benchmark Price')

    # Plot styling
    plt.xlabel("Degree of Polynomial Basis", fontsize=12)
    plt.ylabel("Option Price", fontsize=12)
    plt.title("LSMC Option Price vs. Degree of Polynomial Basis", fontsize=14, fontweight='bold')
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
    exercise_type = "European"
    n_plotted_paths = 6
    barrier_level = None
    basis_type = "Chebyshev"
    degree = 4

    plot_error_vs_basis_degree(
        S0=S0, K=K, r=r, T=T, sigma=sigma, n_time_steps=n_time_steps, n_paths=n_paths, option_type=option_type,
        exercise_type=exercise_type, max_degree=10
    )

    path_range = [100, 500, 1000, 5000, 10000]
    plot_convergence_with_paths(
        S0=S0, K=K, r=r, T=T, sigma=sigma, n_time_steps=n_time_steps, option_type=option_type,
        exercise_type=exercise_type, path_range=path_range
    )
