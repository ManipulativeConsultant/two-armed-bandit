
import numpy as np

np.random.seed(0)

##enums:
RIGHT = True
LEFT = False

class QLearning:

    def __init__(self, left_initial_mean=0., right_initial_mean=1., learning_rate=0.01, gamma=0.95,epsilon = 1.0, epsilon_decay=0.999):
        self.learning_rate = learning_rate
        self.left_mean = left_initial_mean
        self.right_mean = right_initial_mean
        self.epsilon_decay = epsilon_decay# exploration probability decay
        self.epsilon = epsilon #0. #1.0# exploration probability
        self.gamma = gamma

    def update(self, right=RIGHT, reward=0.):
        if right:
            before_update = self.right_mean
            self.right_mean = (1-self.learning_rate)*self.right_mean + self.learning_rate*(reward + self.gamma*self.right_mean)
            after_update = self.right_mean
        else:
            before_update = self.left_mean
            self.left_mean = (1-self.learning_rate)*self.left_mean + self.learning_rate*(reward + self.gamma*self.left_mean)
            after_update = self.left_mean
        loss = np.abs(after_update - before_update)
        return loss

    def choose(self):
        rand_num_01 = np.random.rand()
        self.epsilon *= self.epsilon_decay
        if rand_num_01 < self.epsilon: ## randon decision for exploration:
            right = np.random.randint(0, 2)
        else: ## decide according to learned policy:
            right = self.left_mean < self.right_mean

        if right:
            return RIGHT, self.right_mean
        else:
            return LEFT, self.left_mean

