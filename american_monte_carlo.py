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


# Calculate payoff for a call option
def call_payoff(S, K):
    return np.maximum(S - K, 0)


# Generalized LSMC for European, Bermudan, and American options
def LSMC_option_price(paths, K, r, dt, option_type="European", n_exercise_dates=1):
    n_paths, n_steps = paths.shape
    n_steps -= 1  # Adjust for the initial price at time 0

    # Set exercise dates based on option type
    if option_type == "European":
        exercise_dates = [n_steps]  # Only at maturity
    elif option_type == "Bermudan":
        exercise_dates = np.linspace(0, n_steps, n_exercise_dates + 1, dtype=int)[1:]  # Exclude time 0
    elif option_type == "American":
        exercise_dates = np.arange(1, n_steps + 1)  # Every time step from 1 to n_steps

    cash_flows = np.zeros(n_paths)
    exercise_times = np.full(n_paths, n_steps)

    # Backward iteration: from the last exercise date backwards
    for t in reversed(exercise_dates):
        in_the_money = call_payoff(paths[:, t], K) > 0  # ITM = exercise yes or no
        X = paths[in_the_money, t]  # stock prices for the ITM paths at time t
        Y = cash_flows[in_the_money] * np.exp(
            -r * dt * (exercise_times[in_the_money] - t))  # discounted ITM-CFs at time t

        if len(X) > 0:
            # Polynomial basis functions up to x^3 for regression to estimate continuation values for X
            A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T  # basis functions
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]  # least squares coefficients fitting Y against X
            continuation_estimated = A @ coeffs
            exercise_value = call_payoff(X, K)
            exercise = exercise_value > continuation_estimated  # exercise or hold
            idx = np.where(in_the_money)[0][exercise]  # path index with optimal early exercise
            cash_flows[idx] = exercise_value[exercise]
            exercise_times[idx] = t

    # Discount cash flows back to present
    option_price = np.mean(cash_flows * np.exp(-r * dt * exercise_times))
    return option_price


# Generate QuantLib object for comparison
def get_quantlib_option(S0, K, r, T, sigma, n_steps, exercise_style="European", n_exercise_dates=1):
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
    exercise, engine = exercise_map.get(exercise_style, american_exercise_and_engine)()

    # Payoff
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)

    # Construct the option
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    return option


# LSMC option prices
paths = generate_asset_paths(S0, r, sigma, T, n_steps, n_paths)
lsmc_european_price = LSMC_option_price(paths, K, r, dt, option_type="European")
lsmc_bermudan_price = LSMC_option_price(paths, K, r, dt, option_type="Bermudan", n_exercise_dates=4)
lsmc_american_price = LSMC_option_price(paths, K, r, dt, option_type="American")
print(f"European Option Price (LSMC): {lsmc_european_price:.4f}")
print(f"Bermudan Option Price (LSMC): {lsmc_bermudan_price:.4f}")
print(f"American Option Price (LSMC): {lsmc_american_price:.4f}")

# QuantLib option prices
ql_option_european = get_quantlib_option(S0, K, r, T, sigma, n_steps, exercise_style="European")
ql_option_bermudan = get_quantlib_option(S0, K, r, T, sigma, n_steps, exercise_style="Bermudan", n_exercise_dates=4)
ql_option_american = get_quantlib_option(S0, K, r, T, sigma, n_steps, exercise_style="American")
print(f"European Option Price (QuantLib): {ql_option_european.NPV():.4f}")
print(f"Bermudan Option Price (QuantLib): {ql_option_bermudan.NPV():.4f}")
print(f"American Option Price (QuantLib): {ql_option_american.NPV():.4f}")
