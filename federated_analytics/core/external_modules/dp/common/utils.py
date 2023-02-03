from numbers import Real, Integral

"""
Common functions for DP. Some codes refer to diffprivlib: https://github.com/IBM/differential-privacy-library
"""


def check_bounds(lower, upper):
    if not isinstance(lower, Real) or not isinstance(upper, Real):
        raise TypeError("Bounds must be numeric")
    if lower > upper:
        raise ValueError("Lower bound must not be greater than upper bound")
    return lower, upper


def check_numeric_value(value):
    if not isinstance(value, Real):
        raise TypeError("Value to be randomised must be a number")
    return True


def check_integer_value(value):
    if not isinstance(value, Integral):
        raise TypeError("Value to be randomised must be an integer")
    return True


def check_epsilon_delta(epsilon, delta, allow_zero=False):
    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")
    if delta < 0 or float(delta) > 1.0:
        raise ValueError("Delta must be in [0, 1]")
    if not allow_zero and epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")


def check_params(epsilon, delta, sensitivity):
    check_epsilon_delta(epsilon, delta, allow_zero=False)
    if not isinstance(sensitivity, Real):
        raise TypeError("Sensitivity must be numeric")
    if sensitivity < 0:
        raise ValueError("Sensitivity must be non-negative")
