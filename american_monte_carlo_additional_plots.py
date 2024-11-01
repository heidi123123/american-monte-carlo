import numpy as np
import matplotlib as plt


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
