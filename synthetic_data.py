import numpy as np
from utils import sigmoid, compute_std, linear
from causalgraphicalmodels import StructuralCausalModel


class GermanDataset:
    """
    A semi-synthetic SCM based on the German Credit dataset
    (https://www.kaggle.com/datasets/uciml/german-credit)
    """

    def __init__(self, rf=0.3, seed=3, activation_function=linear, n_samples=1000):
        """
        Initializes the GermanDataset class.

        :param rf: Random factor - controls the randomness of the model (default: 0.3)
        :param seed: Seed for random number generation (default: 3)
        :param activation_function: Activation function to be used (default: linear)
        :param n_samples: Number of the samples in the datasets (default: 1000)
        """
        self.rf = rf
        self.seed = seed
        self.activation = activation_function
        self.columns = ['G', 'A', 'E', 'J', 'L', 'D', 'I', 'S', 'Y']
        self.n_samples = n_samples
        for col in self.columns:
            col = 'c_' + col
            setattr(self, col, 1)
        self.proper_std = {}
        self.scm = None

    def f_G(self):
        np.random.seed(self.seed)
        self.proper_std["G"] = 1
        return (np.random.binomial(1, 0.5, size=self.n_samples) - 0.5) * 2

    def f_A(self):
        np.random.seed(self.seed + 1)
        self.proper_std["A"] = 1
        return (-35 + np.random.gamma(10, scale=3.5, size=self.n_samples)) / 10

    def f_E(self, G, A):
        np.random.seed(self.seed + 2)
        k_G = 1
        k_A = 1
        g_N = 1
        arr = [k_G * self.c_G, k_A * self.c_A, g_N * self.rf]
        l2norm = np.linalg.norm(arr)
        self.proper_std["E"] = compute_std(self.rf * g_N, l2norm)
        return self.activation(k_G * self.c_G * G + k_A * self.c_A * A
                               + np.random.normal(0, self.rf * g_N, size=self.n_samples), scale=l2norm)

    def f_J(self, G, A, E):
        np.random.seed(self.seed + 3)
        k_G = 1
        k_A = 2
        k_E = 4
        g_N = 2

        arr = [k_G * self.c_G, k_A * self.c_A, k_E * self.c_E, g_N * self.rf]
        l2norm = np.linalg.norm(arr)
        self.proper_std["J"] = compute_std(self.rf * g_N, l2norm)
        return self.activation(k_G * self.c_G * G + k_A * self.c_A * A + k_E * self.c_E * E +
                               np.random.normal(0, self.rf * g_N, size=self.n_samples), scale=l2norm)

    def f_L(self, A, G):
        np.random.seed(self.seed + 4)

        k_A = 1
        k_G = 0.5
        g_N = 3

        arr = [k_A * self.c_A, k_G * self.c_G, self.rf * g_N]
        l2norm = np.linalg.norm(arr)
        self.proper_std["L"] = compute_std(self.rf * g_N, l2norm)
        return self.activation(k_A * self.c_A * A + k_G * self.c_G * G +
                               np.random.normal(0, self.rf * g_N, size=self.n_samples), scale=l2norm)

    def f_D(self, G, A, L):
        np.random.seed(self.seed + 5)

        k_G = 1
        k_A = -0.5
        k_L = 2
        g_N = 2

        arr = [k_G * self.c_G, k_A * self.c_A, k_L * self.c_L, g_N * self.rf]
        l2norm = np.linalg.norm(arr)
        self.proper_std["D"] = compute_std(self.rf * g_N, l2norm)
        return self.activation(k_A * self.c_A * A + k_G * self.c_G * G + k_L * self.c_L *
                               L + np.random.normal(0, self.rf * g_N, size=self.n_samples),
                               scale=l2norm)

    def f_I(self, G, A, E, J):
        np.random.seed(self.seed + 6)
        k_G = 0.5
        k_J = 5
        k_E = 4
        k_A = 1
        g_N = 4

        arr = [k_G * self.c_G, k_J * self.c_J, k_E * self.c_E, k_A * self.c_A, self.rf * g_N]
        l2norm = np.linalg.norm(arr)
        self.proper_std["I"] = compute_std(self.rf * g_N, l2norm)
        return self.activation(
            k_A * self.c_A * A + k_G * self.c_G * G + k_J * self.c_J * J + k_E * self.c_E * E +
            np.random.normal(0, self.rf * g_N, size=self.n_samples),
            scale=l2norm)

    def f_S(self, I):
        np.random.seed(self.seed + 7)
        k_I = 5
        g_N = 2
        arr = [k_I * self.c_I, g_N * self.rf]
        l2norm = np.linalg.norm(arr)
        self.proper_std["S"] = compute_std(self.rf * g_N, l2norm)
        return self.activation(k_I * self.c_I * I + np.random.normal(0, self.rf * g_N, size=self.n_samples),
                               scale=l2norm)

    def f_Y(self, I, S, L, D):
        np.random.seed(self.seed + 8)

        k_I = 2
        k_S = 3
        k_L = -1
        k_D = -1

        arr = [k_I * self.c_I, k_S * self.c_S, k_L * self.c_L, k_D * self.c_D]
        l2norm = np.linalg.norm(arr)
        self.proper_std["Y"] = 0.3
        return sigmoid((k_I * self.c_I * I + k_S * self.c_S * S + k_L * self.c_L * L + k_D * self.c_D * D),
                       scale=l2norm)

    def data(self):
        structural_equations = {
            # Gender
            'G': self.f_G,
            # Age
            'A': self.f_A,
            # Education
            'E': self.f_E,
            # Job
            'J': self.f_J,
            # Loan amount
            'L': self.f_L,
            # Loan duration
            'D': self.f_D,
            # Income
            'I': self.f_I,
            # Savings
            'S': self.f_S,
            # Outcome
            'Y': self.f_Y,
        }
        self.scm = StructuralCausalModel(structural_equations)
        return self.scm.sample(n_samples=self.n_samples).astype(float)

    def get_ground_truth(self):
        return self.scm.cgm.draw()
