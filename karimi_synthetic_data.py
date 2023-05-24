import numpy as np
from utils import sigmoid
from causalgraphicalmodels import StructuralCausalModel


class KarimiGermanDataset:
    """
    A semi-synthetic SCM based on the structural equations defined in
    Karimi, A.H., Von K¨ugelgen, J., Sch¨olkopf, B., Valera, I.: Algorithmic recourse
    under imperfect causal knowledge: a probabilistic approach. Advances in neural
    information processing systems 33, 265–277 (2020)
    """

    def __init__(self, rf=1, seed=3, n_samples=1000):
        """
        Initializes the GermanDataset class.

        :param rf: Random factor - controls the randomness of the model (default: 0.3)
        :param self.seed: self.seed for random number generation (default: 3)
        :param activation_function: Activation function to be used (default: linear)
        :param n_samples: Number of the samples in the datasets (default: 1000)
        """
        self.rf = rf
        self.seed = seed
        self.columns = ['G', 'A', 'E', 'J', 'L', 'D', 'I', 'S', 'Y']
        self.n_samples = n_samples
        self.proper_std = {}
        self.scm = None

    def f_g(self):
        np.random.seed(self.seed)
        return np.random.binomial(1, 0.5, size=self.n_samples)

    def f_a(self):
        np.random.seed(self.seed + 1)
        return -35 + np.random.gamma(10, scale=3.5, size=self.n_samples)

    def f_e(self, G, A):
        np.random.seed(self.seed + 2)
        return -0.5 + sigmoid(
            -1 + 0.5 * G + sigmoid(0.1 * A) + np.random.normal(0, self.rf * 0.25, size=self.n_samples))

    def f_j(self, G, A, E):
        np.random.seed(self.seed + 3)
        return -0.5 + sigmoid(0.5 * G + A + 2 * E + np.random.normal(0, self.rf * 3, size=self.n_samples))

    def f_l(self, A, G):
        np.random.seed(self.seed + 4)
        return 1 + -0.01 * (A - 5) * (5 - A) + G + np.random.normal(0, self.rf * 4, size=self.n_samples)

    def f_d(self, G, A, L):
        np.random.seed(self.seed + 5)
        return -1 + 0.1 * A + 2 * G + L + np.random.normal(0, self.rf * 9, size=self.n_samples)

    def f_i(self, G, A, E, J):
        np.random.seed(self.seed + 6)
        return -4 + 0.1 * (A + 35) + 2 * G + G * E + 4 * J + np.random.normal(0, self.rf * 4, size=self.n_samples)

    def f_s(self, I):
        np.random.seed(self.seed + 7)
        return -4 + 1.5 * (I > 0) * I + np.random.normal(0, self.rf * 25, size=self.n_samples)

    def f_y(self, I, S, L, D):
        np.random.seed(self.seed + 8)
        return sigmoid(0.3 * (I + S + I * S - L - D), scale=30)


    def data(self):

        structural_equations = {
            # Gender
            'G': self.f_g,
            # Age
            'A': self.f_a,
            # Education
            'E': self.f_e,
            # Job
            'J': self.f_j,
            # Loan amount
            'L': self.f_l,
            # Loan duration
            'D': self.f_d,
            # Income
            'I': self.f_i,
            # Savings
            'S': self.f_s,
            # Outcome
            'Y': self.f_y,
        }
        self.scm = StructuralCausalModel(structural_equations)
        return self.scm.sample(n_samples=self.n_samples).astype(float)

    def get_ground_truth(self):
        return self.scm.cgm.draw()