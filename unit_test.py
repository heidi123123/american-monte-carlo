import numpy as np
import pytest
from american_monte_carlo import lsmc_option_pricing, get_quantlib_option, generate_asset_paths, intrinsic_value


def run_lsmc_quantlib_test(S0, K, T, r, sigma, n_time_steps, n_paths, option_type, exercise_type, barrier_level):
    np.random.seed(42)
    dt = T / n_time_steps
    basis_type, degree = "Chebyshev", 4

    paths = generate_asset_paths(S0, r, sigma, T, n_time_steps, n_paths)
    lsmc_price, _, _, _ = lsmc_option_pricing(paths, K, r, dt, option_type, barrier_level, exercise_type, basis_type, degree)
    lsmc_price = round(lsmc_price, 4)

    quantlib_option = get_quantlib_option(S0, K, r, T, sigma, n_time_steps, option_type, exercise_type, barrier_level)
    quantlib_price = round(quantlib_option.NPV(), 4)

    print(f"\nTesting configuration: {exercise_type} {option_type}, Barrier Level: {barrier_level}")
    print(f"LSMC Price: {lsmc_price}, QuantLib Price: {quantlib_price}")

    assert abs(lsmc_price - quantlib_price) < 0.2, (
        f"LSMC price does not match QuantLib price within tolerance for "
        f"{exercise_type} {option_type} with barrier level {barrier_level}.\n"
        f"LSMC Price: {lsmc_price}, QuantLib Price: {quantlib_price}"
    )


# Define the parameterized test cases
@pytest.mark.parametrize("option_type, exercise_type, barrier_pct", [
    ("Put", "European", None),
    ("Call", "European", None),
    ("Put", "American", None),
    ("Call", "American", None),
    ("Put", "European", 80),
    ("Call", "European", 80),
    ("Put", "American", 80),
    ("Call", "American", 80),
    ("Put", "European", 60),
    ("Call", "European", 60),
    ("Put", "American", 60),
    ("Call", "American", 60),
])
def test_lsmc_quantlib_comparison(option_type, exercise_type, barrier_pct):
    # Common parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.01, 0.2
    n_time_steps, n_paths = 100, 10000
    barrier_level = S0 * barrier_pct / 100 if barrier_pct else None

    # Run the test for the current parameter set
    run_lsmc_quantlib_test(S0, K, T, r, sigma, n_time_steps, n_paths, option_type, exercise_type, barrier_level)


# Separate intrinsic value test
def test_intrinsic_value():
    S = np.array([90, 100, 110])
    K = 100

    put_intrinsic = intrinsic_value(S, K, "Put")
    call_intrinsic = intrinsic_value(S, K, "Call")

    np.testing.assert_array_almost_equal(put_intrinsic, [10, 0, 0], err_msg="Put intrinsic values incorrect.")
    np.testing.assert_array_almost_equal(call_intrinsic, [0, 0, 10], err_msg="Call intrinsic values incorrect.")


if __name__ == "__main__":
    pytest.main([__file__])
