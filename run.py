

import numpy as np
import matplotlib.pyplot as plt
from broken_armed_bandit import BrokenArmedBandit
from q_learning import QLearning
from bins_asrn import BinsASRN

np.random.seed(2)

##enums:
RIGHT = True
LEFT = False


def demonstrate_manipulative_consultant_stats_and_loss():
    #stats:
    np.random.seed(0)
    game_length = 10000
    n_players = 100
    filter_size = 300
    left_arm_mean = 0.
    right_arm_mean = 1.
    left_arm_std = 0.
    right_arm_std = 10.
    learning_rate = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.999

    all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=False, learning_rate = learning_rate, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, debug = False)
    all_rewards_asrn, all_goods_asrn, all_losses_asrn = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=True,  learning_rate = learning_rate, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, debug = False)

    chose_right_mu_smooth, mu_plus_sigma, mu_minus_sigma = add_sigma(all_goods, 0., 1., filter_size)
    chose_right_mu_asrn_smooth, mu_plus_sigma_asrn, mu_minus_sigma_asrn = add_sigma(all_goods_asrn, 0., 1., filter_size)

    plt.figure(4)
    plt.plot(chose_right_mu_smooth, '-r')
    plt.plot(chose_right_mu_asrn_smooth, '-g')
    # plt.fill_between(np.arange(len(chose_right_mu_smooth)), mu_plus_sigma, mu_minus_sigma, facecolor='red', alpha=0.5)
    # plt.fill_between(np.arange(len(chose_right_mu_asrn_smooth)), mu_plus_sigma_asrn, mu_minus_sigma_asrn, facecolor='green', alpha=0.5)
    plt.ylabel("%Chose right", fontsize=17)
    plt.xlabel("Episode", fontsize=17)
    plt.legend(['without ASRN','with ASRN'], fontsize=15)
    plt.savefig(r"fig2.png")

    # loss figure:
    chose_right = all_goods.sum(0)
    chose_right_asrn = all_goods_asrn.sum(0)

    last_good = np.where(chose_right==0)[0]
    if len(last_good) == 0:
        last_good = len(chose_right)
    else:
        last_good = last_good[0]

    all_losses_left = all_losses.copy()[:,:,0]
    all_losses_right = all_losses.copy()[:,:,0]
    all_goods = all_goods[:,:,0].astype(int)
    all_losses_left[all_goods > 0] = 0.
    all_losses_right[all_goods == 0] = 0.

    loss_left = np.divide(all_losses_left.sum(0), (n_players - chose_right.ravel()))
    loss_right = np.divide(all_losses_right.sum(0)[:last_good], chose_right[:last_good].ravel())

    loss_left = np.correlate(loss_left.ravel(), np.ones(filter_size) / float(filter_size))
    loss_right = np.correlate(loss_right.ravel(), np.ones(filter_size) / float(filter_size))

    last_good_asrn = np.where(chose_right_asrn == 0)[0] - 1
    if len(last_good_asrn) == 0:
        last_good_asrn = len(chose_right)
    else:
        last_good_asrn = last_good_asrn[0]

    all_losses_left_asrn = all_losses_asrn.copy()[:,:,0]
    all_losses_right_asrn = all_losses_asrn.copy()[:,:,0]
    all_goods_asrn = all_goods_asrn[:,:,0].astype(int)
    all_losses_left_asrn[all_goods_asrn>0] = 0.
    all_losses_right_asrn[all_goods_asrn==0] = 0.
    loss_left_asrn = np.divide(all_losses_left_asrn.sum(0),(n_players-chose_right_asrn.ravel()))
    loss_right_asrn = np.divide(all_losses_right_asrn.sum(0)[:last_good_asrn],chose_right_asrn[:last_good_asrn].ravel())
    loss_left_asrn = np.correlate(loss_left_asrn.ravel(), np.ones(filter_size) / float(filter_size))
    loss_right_asrn = np.correlate(loss_right_asrn.ravel(), np.ones(filter_size) / float(filter_size))

    plt.figure(5)
    plt.plot(loss_right, '-g')
    plt.plot(loss_left, '-r')
    plt.plot(loss_right_asrn, '-+g')
    plt.plot(loss_left_asrn, '-+r')
    plt.plot(0, 2.0, 'y.', markersize = 0.001)
    plt.ylabel("Loss", fontsize=17)
    plt.xlabel("Episode", fontsize=17)
    plt.legend(['Chose right', 'Chose left', 'Chose right ASRN', 'Chose left ASRN'], loc='upper center', fontsize=14, ncol=2)#,bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
    plt.savefig(r"fig3.png")
    plt.show()
    a=1


def add_sigma(data, min, max, filter_size):
    mu = data.mean(0)
    sigma = data.std(0)
    mu_plus_sigma = (mu + sigma).ravel()
    mu_minus_sigma = (mu - sigma).ravel()

    mu = np.correlate(mu.ravel(), np.ones(filter_size) / float(filter_size))
    mu_plus_sigma = np.correlate(mu_plus_sigma.ravel(), np.ones(filter_size) / float(filter_size))
    mu_minus_sigma = np.correlate(mu_minus_sigma.ravel(), np.ones(filter_size) / float(filter_size))

    mu_plus_sigma = np.minimum(mu_plus_sigma, max)
    mu_minus_sigma = np.maximum(mu_minus_sigma, min)

    return mu, mu_plus_sigma, mu_minus_sigma


def demonstrate_manipulative_consultant_problem():
    np.random.seed(0)
    game_length = 2000
    n_players = 10
    filter_size = 300

    left_arm_mean = 0.
    right_arm_mean = 1.
    left_arm_std = 0.
    right_arm_std = 8.
    learning_rate = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.99

    all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=False, learning_rate = learning_rate, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, debug = False)
    all_rewards_asrn, all_goods_asrn, all_losses_asrn = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=True,  learning_rate = learning_rate, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, debug = False)

    all_rewards = [np.correlate(all_rewards[i].ravel(), np.ones(filter_size) / float(filter_size)) for i in range(n_players)]
    all_rewards = np.asarray(all_rewards)
    plt.figure(2)
    plt.plot(all_rewards[:10, :].T)
    plt.title("rewards without ASRN", fontsize=20)
    plt.xlabel("reward", fontsize=20)
    plt.ylabel("episode", fontsize=20)
    plt.savefig(r"fig1a.png")

    all_rewards2 = [np.correlate(all_rewards_asrn[i].ravel(), np.ones(filter_size) / float(filter_size)) for i in range(n_players)]
    all_rewards2 = np.asarray(all_rewards2)
    plt.figure(3)
    plt.plot(all_rewards2[:10, :].T)
    plt.title("rewards with ASRN", fontsize=20)
    plt.xlabel("reward", fontsize=20)
    plt.ylabel("episode", fontsize=20)
    plt.savefig(r"fig1b.png")
    plt.show()

def demonstrate_boring_areas_trap():
    game_length = 400
    n_players = 1

    left_arm_mean = 0.
    right_arm_mean = 1.
    left_arm_std = 0.5
    right_arm_std = 7.
    epsilon = 0.0 # no exploration to see the pure problem.
    plt.figure(1)
    use_asrn = False
    all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = 0.1, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999, debug = True)
    plt.title("Q table values without ASRN")
    plt.savefig(r"fig4.png")
    plt.figure(100)

    use_asrn = True
    epsilon = 1.0 # ASRN needs exploration
    all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = 0.1, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999, debug = True)
    plt.title("Q table values with ASRN")
    plt.savefig(r"fig4b.png")


def demonstrate_low_alpha():
    game_length = 1000000
    n_players = 1

    left_arm_mean = 0.
    right_arm_mean = 0.001
    left_arm_std = 0.1
    right_arm_std = 13.
    learning_rate = 0.000001 # low alpha

    seed = 2
    np.random.seed(seed)
    plt.figure(100)
    use_asrn = False
    epsilon = 0.0 # ASRN needs exploration
    all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = learning_rate, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999, debug = True)
    plt.title("Q table values without ASRN")
    plt.savefig(r"fig4b.png")

    np.random.seed(seed)
    plt.figure(1)
    use_asrn = True
    epsilon = 1.0 # no exploration to see the pure problem.
    all_rewards, all_goods, all_losses = run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = learning_rate, gamma=0.9, epsilon=epsilon, epsilon_decay=0.999, debug = True)
    plt.title("Q table values with ASRN")
    plt.savefig(r"fig4a.png")

    plt.show()
    a=1

def run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn, learning_rate = 0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, debug = False, random_init=False):
    all_rewards = []
    all_goods = []
    all_losses = []
    trained_agent_q_values = [left_arm_mean / (1 - gamma), right_arm_mean / (1 - gamma)]
    mx = np.max(trained_agent_q_values)
    mn = np.min(trained_agent_q_values)
    avg = 0
    std = mx-mn
    for j in range(n_players):
        two_armed_bandit = BrokenArmedBandit(left_arm_mean=left_arm_mean, right_arm_mean=right_arm_mean,
                                           left_arm_std=left_arm_std, right_arm_std=right_arm_std)

        if random_init:
            left_initial_mean =  np.random.normal(avg, std)
            right_initial_mean = np.random.normal(avg, std)
            if left_initial_mean < right_initial_mean:
                left_initial_mean = -1
                right_initial_mean = 1
            else:
                left_initial_mean = 1
                right_initial_mean = -1
        else:
            ## giving the real mean as initialization(!)
            left_initial_mean = trained_agent_q_values[0]
            right_initial_mean = trained_agent_q_values[1]

        q_learning = QLearning(left_initial_mean, right_initial_mean, learning_rate, gamma, epsilon, epsilon_decay)
        rewards = np.zeros((game_length, 1))
        goods = np.zeros((game_length, 1))
        losses = np.zeros((game_length, 1))
        debug_data = []

        if use_asrn:
            asrn = BinsASRN(0, learning_period=game_length/10)
        for i in range(game_length):
            right, reward_estimation = q_learning.choose()
            good = q_learning.right_mean > q_learning.left_mean
            goods[i] = good
            if debug:
                debug_data.append([right, q_learning.right_mean, q_learning.left_mean])
            reward = two_armed_bandit.pull(right)
            rewards[i] = reward

            if use_asrn:
                if right:
                    updated_right_mean = (1 - q_learning.learning_rate) * q_learning.right_mean + q_learning.learning_rate * (reward + q_learning.gamma * q_learning.right_mean)
                    reward = asrn.noise(q_learning.right_mean, updated_right_mean, reward)
                else:
                    updated_left_mean = (1 - q_learning.learning_rate) * q_learning.left_mean + q_learning.learning_rate * (reward + q_learning.gamma * q_learning.left_mean)
                    reward = asrn.noise(q_learning.left_mean, updated_left_mean, reward)

            loss = q_learning.update(right, reward)
            losses[i] = loss

        all_rewards.append(rewards)
        all_goods.append(goods)
        all_losses.append(losses)
        if debug:
            debug_data = np.asarray(debug_data)[:, 1:]
            plt.plot(debug_data[:, 0], '-g')
            plt.plot(debug_data[:, 1], '-r')
            plt.legend(['Q r', 'Q l'])
            plt.show()

    return np.asarray(all_rewards), np.asarray(all_goods), np.asarray(all_losses)

if __name__ == '__main__':
    demonstrate_manipulative_consultant_problem()
    demonstrate_manipulative_consultant_stats_and_loss()
    demonstrate_boring_areas_trap()
    print("Checking multiple variance combinations will take time. Please run at weekend.")
    import check_gaussians
    check_gaussians.build_heat_map()
    plt.show()
    # demonstrate_low_alpha()
