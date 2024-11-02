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


path_range = [100, 500, 1000, 5000, 10000]
plot_convergence_with_paths(
    S0=100, K=100, r=0.05, T=1.0, sigma=0.2, n_time_steps=50, option_type='Put',
    exercise_type='American', path_range=path_range
)
