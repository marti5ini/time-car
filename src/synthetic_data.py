"""
A semi-synthetic SCM based on the German Credit dataset
(https://www.kaggle.com/datasets/uciml/german-credit)
"""

import numpy as np
from utils import sigmoid, compute_std, linear


# random factor - controls the randomness of the model
rf = 0.3
# seed: Seed for random number generation
seed = 3
# activation function to be used
activation = linear

c_G = 1
c_A = 1
c_E = 1
c_J = 1
c_L = 1
c_D = 1
c_I = 1
c_S = 1
c_Y = 1

proper_std = {}


def f_G(n_samples):
    np.random.seed(seed)
    proper_std["G"] = 1
    return (np.random.binomial(1, 0.5, size=n_samples) - 0.5) * 2


def f_A(n_samples):
    np.random.seed(seed + 1)
    proper_std["A"] = 1
    return (-35 + np.random.gamma(10, scale=3.5, size=n_samples)) / 10


def f_E(n_samples, G, A):
    np.random.seed(seed + 2)

    k_G = 1
    k_A = 1
    g_N = 1

    arr = [k_G * c_G, k_A * c_A, g_N * rf]
    l2norm = np.linalg.norm(arr)
    proper_std["E"] = compute_std(rf * g_N, l2norm)
    return activation(k_G * c_G * G + k_A * c_A * A + np.random.normal(0, rf * g_N, size=n_samples), scale=l2norm)


def f_J(n_samples, G, A, E):
    np.random.seed(seed + 3)

    k_G = 1
    k_A = 2
    k_E = 4
    g_N = 2

    arr = [k_G * c_G, k_A * c_A, k_E * c_E, g_N * rf]
    l2norm = np.linalg.norm(arr)
    proper_std["J"] = compute_std(rf * g_N, l2norm)
    return activation(k_G * c_G * G + k_A * c_A * A + k_E * c_E * E + np.random.normal(0, rf * g_N, size=n_samples),
                      scale=l2norm)


def f_L(n_samples, A, G):
    np.random.seed(seed + 4)

    k_A = 1
    k_G = 0.5
    g_N = 3

    arr = [k_A * c_A, k_G * c_G, rf * g_N]
    l2norm = np.linalg.norm(arr)
    proper_std["L"] = compute_std(rf * g_N, l2norm)
    return activation(k_A * c_A * A + k_G * c_G * G + np.random.normal(0, rf * g_N, size=n_samples), scale=l2norm)


def f_D(n_samples, G, A, L):
    np.random.seed(seed + 5)

    k_G = 1
    k_A = -0.5
    k_L = 2
    g_N = 2

    arr = [k_G * c_G, k_A * c_A, k_L * c_L, g_N * rf]
    l2norm = np.linalg.norm(arr)
    proper_std["D"] = compute_std(rf * g_N, l2norm)
    return activation(k_A * c_A * A + k_G * c_G * G + k_L * c_L * L + np.random.normal(0, rf * g_N, size=n_samples),
                      scale=l2norm)


def f_I(n_samples, G, A, E, J):
    np.random.seed(seed + 6)

    k_G = 0.5
    k_J = 5
    k_E = 4
    k_A = 1
    g_N = 4

    arr = [k_G * c_G, k_J * c_J, k_E * c_E, k_A * c_A, rf * g_N]
    l2norm = np.linalg.norm(arr)
    proper_std["I"] = compute_std(rf * g_N, l2norm)
    return activation(
        k_A * c_A * A + k_G * c_G * G + k_J * c_J * J + k_E * c_E * E + np.random.normal(0, rf * g_N, size=n_samples),
        scale=l2norm)


def f_S(n_samples, I):
    np.random.seed(seed + 7)

    k_I = 5
    g_N = 2

    arr = [k_I * c_I, g_N * rf]
    l2norm = np.linalg.norm(arr)
    proper_std["S"] = compute_std(rf * g_N, l2norm)
    return activation(k_I * c_I * I + np.random.normal(0, rf * g_N, size=n_samples), scale=l2norm)


def f_Y(n_samples, I, S, L, D):
    np.random.seed(seed + 8)

    k_I = 2
    k_S = 3
    k_L = -1
    k_D = -1

    arr = [k_I * c_I, k_S * c_S, k_L * c_L, k_D * c_D]
    l2norm = np.linalg.norm(arr)
    proper_std["Y"] = 0.3
    return sigmoid((k_I * c_I * I + k_S * c_S * S + k_L * c_L * L + k_D * c_D * D), scale=l2norm)


def get_structural_equations():
    structural_equations = {
        # Gender
        'G': f_G,
        # Age
        'A': f_A,
        # Education
        'E': f_E,
        # Job
        'J': f_J,
        # Loan amount
        'L': f_L,
        # Loan duration
        'D': f_D,
        # Income
        'I': f_I,
        # Savings
        'S': f_S,
        # Outcome
        'Y': f_Y,
    }
    return structural_equations
