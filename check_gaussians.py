

import numpy as np
import matplotlib.pyplot as plt
import run
import datetime


def build_heat_map():
    game_length = 10000
    n_players = 50
    left_range = [0., 10.]
    right_range = [0., 20.]
    sz = [50, 50]

    left_arm_mean = 0.0
    right_arm_mean = 1.0
    use_asrn = False
    total_rewards = np.zeros((sz[0],sz[1]))
    final_rights = np.zeros((sz[0],sz[1]))
    learning_rate = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.999
    i = 0
    if True:
        for left_arm_std in np.arange(left_range[0],left_range[1], (left_range[1]-left_range[0])/float(sz[0])):
            j=0
            for right_arm_std in np.arange(right_range[0],right_range[1], (right_range[1]-right_range[0])/float(sz[1])):
                print(str(i) + "," + str(j))
                all_rewards, all_goods, all_losses = run.run_games(game_length, left_arm_mean, left_arm_std, n_players, right_arm_mean, right_arm_std, use_asrn=use_asrn, learning_rate = learning_rate, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, debug = False)
                total_rewards[i, j] = all_rewards.sum()
                final_rights[i, j] = all_goods[:, -1, 0].sum()
                j += 1
            i += 1
    fig, ax1 = plt.subplots()
    plt.imshow(np.flipud(final_rights))
    plt.colorbar()
    ax1.set_xticklabels(range(int(right_range[0])-4, int(right_range[1]), 4))
    ax1.set_yticklabels(range(int(left_range[1])+2, int(left_range[0]), -2))
    plt.xlabel("Right arm", fontsize=17)
    plt.ylabel("Left arm", fontsize=17)

    date = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
    plt.savefig(r"fig5.jpg")
    np.save(r"D:\projects\NIPS2019\data\rightVsLeftVariancesFinalRights" + date, final_rights)
    # np.save(r"D:\projects\NIPS2019\data\rightVsLeftVariancesTotal_rewards"  + date, total_rewards)
    # plt.show()
    return final_rights, total_rewards, left_range, right_range

if __name__ == '__main__':
    load = True#False#
    debug = False#True#
    if not load:
        final_rights, total_rewards, left_range, right_range = build_heat_map()
    else:
        final_rights = np.load(r"D:\projects\NIPS2019\data\rightVsLeftVariancesFinalRights2019-04-14_17-01-53.150191.npy")
        left_range = [0., 10.]
        right_range = [0., 10.]

    if debug:
        fig, ax1 = plt.subplots()
        # plt.imshow(final_rights)
        plt.imshow(np.flipud(final_rights))
        plt.colorbar()
        ax1.set_xticklabels(range(int(right_range[0]) - 4, int(right_range[1]), 4))
        ax1.set_yticklabels(range(int(left_range[1]) + 2, int(left_range[0]), -2))
        plt.title("Influence of variance on convergence", fontsize=20)
        plt.xlabel("Right arm", fontsize=20)
        plt.ylabel("Left arm", fontsize=20)
        # plt.show()
        date = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
        plt.savefig(r"fig5.png")
        # np.save(r"D:\projects\NIPS2019\data\FinalRights" + str(left_range[0]) + "_" + str(left_range[1]) + "_" + + str(right_range[0]) + "_" + + str(right_range[0])  , final_rights)
        # np.save(r"D:\projects\NIPS2019\data\TotalRewards" + str(left_range[0]) + "_" + str(left_range[1]) + "_" + + str(right_range[0]) + "_" + + str(right_range[0])  , total_rewards)

    final_rights_ = final_rights.copy()
    final_rights_[final_rights>26] = 0
    final_rights_[final_rights<24] = 0
    y, x = np.where(final_rights_>0)
    a, b = np.polyfit(x, y, 1)
    x_ = np.arange(final_rights.shape[0])
    y_ = a * x_ + b
    if debug:
        # plt.plot(x_, y_, '-r')
        plt.show()

    x = (x - right_range[0]) / float(final_rights.shape[1]) * (right_range[1]-right_range[0])
    y = (y - left_range[0]) / float(final_rights.shape[0]) * (left_range[1] - left_range[0])
    a, b = np.polyfit(x, y, 1)
    print(str(a) + " , " + str(b))

