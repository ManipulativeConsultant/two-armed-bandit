import numpy as np
import torch
import torch.nn as nn
import torch.optim

class Flatten(nn.Module):
    def forward(selfself, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m,
                       nn.init.orthogonal_,
                       lambda x: nn.init.constant_(x, 0),
                       nn.init.calculate_gain('relu'))

class DeepAsrn:
    def __init__(self, n_networks=5, action_space=3, num_input_images=4):
        self.n_networks = n_networks
        self.action_space = action_space
        self.num_input_images = num_input_images
        self.networks = list()
        self.optimizers = list()
        for i in range(self.n_networks):
            network = self.init_network()
            self.networks.append(network)
            optimizer = torch.optim.RMSprop(network.parameters(), lr=0.01, momentum=0.9)
            self.optimizers.append(optimizer)
        self.experience_size = 1000
        self.experience = np.zeros(self.experience_size)
        self.place_in_experience = 0
        self.desired_noise = 0
        self.representative_noise_ratio = 0.5
        self.representative_noise_place = int(self.experience_size * self.representative_noise_ratio)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def noise_reward(self, input_images, actions, in_reward):
        noised_rewards = []
        n_processes = in_reward.size()[0]
        for k in range(n_processes):
            action_num = np.asarray(actions)[k][0]
            reward_num = np.asarray(in_reward)[k][0]
            out_rewards = np.zeros(self.n_networks)
            for i in range(self.n_networks):
                #use network:
                input_image = torch.unsqueeze(input_images[k], 0)
                rewards_per_actions = self.networks[i](input_images)
                out_rewards[i] = rewards_per_actions[0][action_num]

                #update network:
                pred = torch.tensor([[rewards_per_actions[0][action_num]]], requires_grad=True)
                loss = self.loss_fn(pred, torch.unsqueeze(in_reward[k], 0))
                loss = torch.tensor(loss, requires_grad=True)
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
            real_std = False # True for neglecting true reward
            if real_std:
                std = np.std(out_rewards)
            else:
                std = np.sqrt(((out_rewards - reward_num)**2).mean())

            #update desired noise:
            self.experience[self.place_in_experience] = std
            self.place_in_experience += 1
            if self.place_in_experience >= self.experience_size:
                self.place_in_experience=0
            self.desired_noise = np.partition(self.experience, self.representative_noise_place)[self.representative_noise_place]

            #noise reward:
            wanted_var_add = self.desired_noise**2 - std**2
            wanted_var_add = max(0, wanted_var_add)
            noise = np.sqrt(wanted_var_add)
            noised_reward = np.random.normal(reward_num, noise)
            noised_rewards.append(noised_reward)
        noised_rewards = torch.Tensor([[noised_rewards[x]] for x in range(n_processes)])
        return noised_rewards

    def init_network(self):
        network = nn.Sequential(
            init_(nn.Conv2d(self.num_input_images, 32, 8, stride=4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            Flatten(),
            init_(nn.Linear(32 * 7 * 7, self.action_space))
        )
        return network



