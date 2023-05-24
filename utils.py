import numpy as np
import math


def sigmoid(x, scale=1, factor=1):
    return factor * np.exp(-np.logaddexp(0, -x / scale))


def log_sigmoid(x, scale=1, factor=1):
    return factor * np.sign(x) * np.log(1 + abs(x / scale)) / (1 + np.log(1 + abs(x / scale)))


def k_sigmoid(x, k=1, scale=1, factor=1):
    return factor * (np.sign(x) * abs(x / scale) ** k) / (1 + abs(x / scale))


def linear(x, scale=1):
    return x / scale


def compute_std(x, norm):
    return math.sqrt(x / norm)