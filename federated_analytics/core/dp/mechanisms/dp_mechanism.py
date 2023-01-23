from federated_analytics.core.dp.mechanisms import Laplace, Gaussian

"""dp mechanisms, e.g., Gaussian, Laplace """

class DPMechanism:
    def __init__(self, mechanism_type, epsilon, delta, sensitivity=1):
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(
                epsilon=epsilon, delta=delta, sensitivity=sensitivity
            )
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon, delta=delta, sensitivity=sensitivity)
        else:
            raise NotImplementedError("DP mechanism not implemented!")

    def add_noise(self, real_val):
        return self._compute_new_value_after_adding_noise(real_val)

    def _compute_new_value_after_adding_noise(self, grad):
        noise = self.dp.compute_noise(grad.shape)
        return noise + grad

    def add_a_noise_to_local_data(self, local_data):
        new_data = []
        for i in range(len(local_data)):
            list = []
            for x in local_data[i]:
                y = self._compute_new_value_after_adding_noise(x)
                list.append(y)
            new_data.append(tuple(list))
        return new_data




