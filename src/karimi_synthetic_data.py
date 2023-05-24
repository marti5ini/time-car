"""
A semi-synthetic SCM based on the structural equations defined in
Karimi, A.H., Von K¨ugelgen, J., Sch¨olkopf, B., Valera, I.: Algorithmic recourse
under imperfect causal knowledge: a probabilistic approach. Advances in neural
information processing systems 33, 265–277 (2020)
"""
import numpy as np
from utils import sigmoid

rf = 1
seed = 3


def f_g(n_samples):
    np.random.seed(seed)
    return np.random.binomial(1, 0.5, size=n_samples)


def f_a(n_samples):
    np.random.seed(seed + 1)
    return -35 + np.random.gamma(10, scale=3.5, size=n_samples)


def f_e(n_samples, G, A):
    np.random.seed(seed + 2)
    return -0.5 + sigmoid(-1 + 0.5 * G + sigmoid(0.1 * A) + np.random.normal(0, rf * 0.25, size=n_samples))


def f_j(n_samples, G, A, E):
    np.random.seed(seed + 3)
    return -0.5 + sigmoid(0.5 * G + A + 2 * E + np.random.normal(0, rf * 3, size=n_samples))


def f_l(n_samples, A, G):
    np.random.seed(seed + 4)
    return 1 + -0.01 * (A - 5) * (5 - A) + G + np.random.normal(0, rf * 4, size=n_samples)


def f_d(n_samples, G, A, L):
    np.random.seed(seed + 5)
    return -1 + 0.1 * A + 2 * G + L + np.random.normal(0, rf * 9, size=n_samples)


def f_i(n_samples, G, A, E, J):
    np.random.seed(seed + 6)
    return -4 + 0.1 * (A + 35) + 2 * G + G * E + 4 * J + np.random.normal(0, rf * 4, size=n_samples)


def f_s(n_samples, I):
    np.random.seed(seed + 7)
    return -4 + 1.5 * (I > 0) * I + np.random.normal(0, rf * 25, size=n_samples)


def f_y(n_samples, I, S, L, D):
    np.random.seed(seed + 8)
    return sigmoid(0.3 * (I + S + I * S - L - D), scale=30)


def get_karimi_structural_equations():
    structural_equations = {
        # Gender
        'G': f_g,
        # Age
        'A': f_a,
        # Education
        'E': f_e,
        # Job
        'J': f_j,
        # Loan amount
        'L': f_l,
        # Loan duration
        'D': f_d,
        # Income
        'I': f_i,
        # Savings
        'S': f_s,
        # Outcome
        'Y': f_y,
    }
    return structural_equations
