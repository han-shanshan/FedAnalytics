import numpy as np
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Laplace(BaseDPMechanism):
    """
    The classical Laplace mechanism in differential privacy.
    """

    def __init__(self, epsilon, delta=0.0, sensitivity=1):
        check_params(epsilon, delta, sensitivity)
        self.scale = float(sensitivity) / (float(epsilon) - np.log(1 - float(delta)))
        self.sensitivity = sensitivity

    def compute_noise(self, size):
        return np.random.laplace(loc=0.0, scale=self.scale, size=size)