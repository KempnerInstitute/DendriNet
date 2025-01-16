from math import sqrt, exp, erf
from scipy.stats import norm


def compute_expectation_truncated_log_normal(
    mean, std, upper_quantile,
):
    z_score = mean + std * norm.ppf(1 - upper_quantile).item()

    numerator = exp(mean + 0.5 * std**2) * (
        1 - erf((z_score - mean - std**2) / (std * sqrt(2))))
    denominator = 1 - erf((z_score - mean) / (std * sqrt(2)))

    expectation = numerator / denominator
    return expectation







