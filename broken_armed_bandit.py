

import numpy as np

np.random.seed(0)

##enums:
RIGHT = True
LEFT = False

class BrokenArmedBandit:
    def __init__(self, left_arm_mean = 0., right_arm_mean = 1., left_arm_std = 0., right_arm_std = 1.):
        self.left_arm_mean = left_arm_mean
        self.right_arm_mean = right_arm_mean
        self.left_arm_std = left_arm_std
        self.right_arm_std = right_arm_std

    def pull(self, right=RIGHT):
        if right:
            res = np.random.normal(self.right_arm_mean, self.right_arm_std)
        else:
            res = np.random.normal(self.left_arm_mean, self.left_arm_std)
        return res


