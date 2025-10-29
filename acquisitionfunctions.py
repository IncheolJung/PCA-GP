from numpy import sqrt
from scipy.stats import norm


def expected_improvement(mu, var, f_best, xi=1e-6):
    """Expected Improvement (EI) for maximization."""
    std = sqrt(var)
    z = (mu - f_best - xi) / (std + 1e-12)
    return (mu - f_best - xi) * norm.cdf(z) + std * norm.pdf(z)

def probability_of_improvement(mu, var, f_best, xi=1e-6):
    """Probability of Improvement (PI)."""
    std = sqrt(var)
    z = (mu - f_best - xi) / (std + 1e-12)
    return norm.cdf(z)

def upper_confidence_bound(mu, var, kappa=2.0):
    """Upper Confidence Bound (UCB)."""
    std = sqrt(var)
    return mu + kappa * std

def lower_confidence_bound(mu, var, kappa=2.0):
    """Lower Confidence Bound (LCB), for minimization."""
    std = sqrt(var)
    return mu - kappa * std

def max_variance(mu, var):
    """Variance-only heuristic (pure exploration)."""
    return var