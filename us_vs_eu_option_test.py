import matplotlib.pyplot as plt
import numpy as np
from american_monte_carlo import get_quantlib_option

# Fixed parameters
r = 0.03
sigma = 0.25
n_steps = 1000


def calculate_option_prices(param_vals, varying_param, S0, K, T, dividend_yield):
    """
    Calculate American and European Call/Put option prices for varying parameters.

    param_vals: List of values for the varying parameter.
    varying_param: Parameter being varied ('S0', 'K', 'T', 'dividend_yield').
    S0, K, T: Fixed parameters for the non-varying parameters.
    """
    us_call_prices, eu_call_prices = [], []
    us_put_prices, eu_put_prices = [], []

    for val in param_vals:
        # Adjust the varying parameter
        if varying_param == "S0":
            S0_val = val
            K_val, T_val, div_val = K, T, dividend_yield
        elif varying_param == "K":
            K_val = val
            S0_val, T_val, div_val = S0, T, dividend_yield
        elif varying_param == "T":
            T_val = val
            S0_val, K_val, div_val = S0, K, dividend_yield
        elif varying_param == "dividend_yield":
            div_val = val
            S0_val, K_val, T_val = S0, K, T
        else:
            raise ValueError("Invalid parameter to vary.")

        # American and European Call
        us_call = get_quantlib_option(S0_val, K_val, r, T_val, sigma, n_steps, option_type="Call",
                                      exercise_type="American", dividend_yield=div_val)
        eu_call = get_quantlib_option(S0_val, K_val, r, T_val, sigma, n_steps, option_type="Call",
                                      exercise_type="European", dividend_yield=div_val)
        us_call_prices.append(us_call.NPV())
        eu_call_prices.append(eu_call.NPV())

        # American and European Put
        us_put = get_quantlib_option(S0_val, K_val, r, T_val, sigma, n_steps, option_type="Put",
                                     exercise_type="American", dividend_yield=div_val)
        eu_put = get_quantlib_option(S0_val, K_val, r, T_val, sigma, n_steps, option_type="Put",
                                     exercise_type="European", dividend_yield=div_val)
        us_put_prices.append(us_put.NPV())
        eu_put_prices.append(eu_put.NPV())

    return us_call_prices, eu_call_prices, us_put_prices, eu_put_prices


# Plot results for a given parameter variation
def plot_option_prices(ax, x_vals, us_prices, eu_prices, title, xlabel, ylabel):
    ax.plot(x_vals, us_prices[0], label="American Call", linestyle="-")
    ax.plot(x_vals, eu_prices[0], label="European Call", linestyle="--")
    ax.plot(x_vals, us_prices[1], label="American Put", linestyle="-")
    ax.plot(x_vals, eu_prices[1], label="European Put", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()


def vary_S0(ax, K, T, dividend_yield, S0_vals=np.linspace(60, 140, 80)):
    us_call, eu_call, us_put, eu_put = calculate_option_prices(S0_vals, "S0", S0=None, K=K, T=T,
                                                               dividend_yield=dividend_yield)
    plot_option_prices(
        ax,
        S0_vals,
        (us_call, us_put),
        (eu_call, eu_put),
        title=f"Option Prices vs Spot Price\nK={K}, T={T}, DivYield={dividend_yield}",
        xlabel="Spot Price (S0)",
        ylabel="Option Price"
    )


def vary_K(ax, S0, T, dividend_yield, K_vals=np.linspace(60, 140, 80)):
    us_call, eu_call, us_put, eu_put = calculate_option_prices(K_vals, "K", S0=S0, K=None, T=T,
                                                               dividend_yield=dividend_yield)
    plot_option_prices(
        ax,
        K_vals,
        (us_call, us_put),
        (eu_call, eu_put),
        title=f"Option Prices vs Strike Price\nS0={S0}, T={T}, DivYield={dividend_yield}",
        xlabel="Strike Price (K)",
        ylabel="Option Price"
    )


def vary_T(ax, S0, K, dividend_yield, T_vals=np.linspace(0.01, 2, 50)):
    us_call, eu_call, us_put, eu_put = calculate_option_prices(T_vals, "T", S0=S0, K=K, T=None,
                                                               dividend_yield=dividend_yield)
    plot_option_prices(
        ax,
        T_vals,
        (us_call, us_put),
        (eu_call, eu_put),
        title=f"Option Prices vs Time to Maturity\nS0={S0}, K={K}, DivYield={dividend_yield}",
        xlabel="Time to Maturity (T)",
        ylabel="Option Price"
    )


def vary_div_yield(ax, S0, K, T, dividend_yield_vals=np.linspace(0, 0.05, 50)):
    us_call, eu_call, us_put, eu_put = calculate_option_prices(dividend_yield_vals, "dividend_yield", S0=S0, K=K, T=T,
                                                               dividend_yield=None)
    plot_option_prices(
        ax,
        dividend_yield_vals,
        (us_call, us_put),
        (eu_call, eu_put),
        title=f"Option Prices vs Dividend Yield\nS0={S0}, K={K}, T={T}",
        xlabel="Dividend Yield",
        ylabel="Option Price"
    )


if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1
    dividend_yield = 0.05

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot the four option prices on the subplots
    vary_S0(axs[0, 0], K, T, dividend_yield)
    vary_K(axs[0, 1], S0, T, dividend_yield)
    vary_div_yield(axs[0, 2], S0, K, T)
    vary_T(axs[1, 0], S0, K, dividend_yield)
    vary_T(axs[1, 1], 110, K, dividend_yield)
    vary_T(axs[1, 2], S0, 110, dividend_yield)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
