
import numpy as np
import matplotlib.pyplot as plt

# Adaptive Symmetric Noising Scheme
class BinsASRN:
    def __init__(self, waiting_period=0, learning_period=10000):
        self.steps = 0
        self.waiting_period = waiting_period
        self.learning_period = learning_period
        self.learning_experience = []
        self.n_bins = 10
        self.bins = []
        self.bins_noise = []
        self.training_done = False

    # Return a noised reward, noise is adaptive to the actual error. Smaller error causes bigger noise.
    def noise(self, estimated_Q, true_Q, reward):

        err = np.abs(estimated_Q - true_Q)

        if self.steps < self.waiting_period:
            noised_reward = reward
        elif self.steps < self.waiting_period + self.learning_period:
            self.learning_experience.append([err, reward])
            noised_reward = reward
        elif self.steps == self.waiting_period + self.learning_period:
            learning_experience = np.asarray(self.learning_experience)
            exp = learning_experience[learning_experience[:, 0].argsort()]
            step_size = int(np.floor(self.learning_period/self.n_bins))
            self.bins = exp[0:int(self.learning_period)-1:step_size, 0]
            bins_stds = np.zeros(self.n_bins)
            self.bins_noise = np.zeros(self.n_bins)
            for i in range(0, int(self.learning_period)-1, step_size):
                data = exp[i:i+step_size, 1]
                bins_stds[int(i / step_size)] = np.std(data)
            wanted_err_mean = np.max(bins_stds)
            self.bins_noise = np.sqrt(wanted_err_mean*wanted_err_mean - bins_stds*bins_stds)
            self.training_done = True
            noised_reward = reward
        else:
            assert self.training_done
            selected_bin = np.where(self.bins <= err)[0]
            if len(selected_bin) > 0:
                selected_bin = selected_bin[-1]
            else:
                selected_bin = 0
            noise = self.bins_noise[selected_bin]
            noised_reward = np.random.normal(reward, noise)

        self.steps += 1
        return noised_reward
