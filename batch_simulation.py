# import sys
import numpy as np
import matplotlib.pyplot as plt

from simulation import simulation

## save numpy array as npy files
#from numpy import asarray
#from numpy import save

# The RL algorithms:
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
# ... and the methods for exploration:
SOFTMAX = "softmax"
EPSILON_GREEDY = "epsilon_greedy"

NUM_EPISODES = 1000
STREAK_TO_END = 120 # number of "success" (i.e. how quick the maze is solved) needed to assess the good performance of a process

# PARAMETERS
size_maze = 10 # 20, 30, 50, 100, ... 
size_batch = 5

points_for_plotting = 50

# A[] <- A[] + a[]    with A of size NUM_EPISODES and len(a) < NUM_EPISODES
def sum_arrays(A, a):
    assert((len(A) == NUM_EPISODES) and (len(a) < NUM_EPISODES))
    for i in range(len(a)):
        A[i] += a[i]


"""
This file is used to run the simulation 
"""

CHOICE = 1 # 1, 2 or 3


## TO COMPARE SARSA and Q_LEARNING:
def compare_RL_algos():
    # sr_record_S = sr_record_Q = tr_record_S = tr_record_Q = rt_record_S = rt_record_Q = np.zeros(NUM_EPISODES)
    
    sr_record_S = np.zeros(NUM_EPISODES)
    sr_record_Q = np.zeros(NUM_EPISODES)
    tr_record_S = np.zeros(NUM_EPISODES)
    tr_record_Q = np.zeros(NUM_EPISODES)
    rt_record_S = np.zeros(NUM_EPISODES)
    rt_record_Q = np.zeros(NUM_EPISODES)
    
    min_nb_episodes = np.inf 
    for seed in range(size_batch):
        n_episode_S, sr_S, tr_S, rt_S = simulation(SARSA, EPSILON_GREEDY, 0.5, 0.9, size_maze, seed)
        n_episode_Q, sr_Q, tr_Q, rt_Q = simulation(Q_LEARNING, EPSILON_GREEDY, 0.5, 0.9, size_maze, seed)
        sum_arrays(sr_record_S, np.array(sr_S))
        sum_arrays(sr_record_Q, np.array(sr_Q))
        sum_arrays(tr_record_S, np.array(tr_S))
        sum_arrays(tr_record_Q, np.array(tr_Q))
        sum_arrays(rt_record_S, np.array(rt_S))
        sum_arrays(rt_record_Q, rt_Q)
        if (n_episode_S < min_nb_episodes):
            min_nb_episodes = n_episode_S
        if (n_episode_Q < min_nb_episodes):
            min_nb_episodes = n_episode_Q
    sr_record_S /= size_batch; sr_record_Q /= size_batch
    tr_record_S /= size_batch; tr_record_Q /= size_batch 
    rt_record_S /= size_batch; rt_record_S /= size_batch
    sr_record_S = sr_record_S[:min_nb_episodes]; sr_record_Q = sr_record_Q[:min_nb_episodes]
    tr_record_S = tr_record_S[:min_nb_episodes]; tr_record_Q = tr_record_Q[:min_nb_episodes]
    rt_record_S = rt_record_S[:min_nb_episodes]; rt_record_Q = rt_record_Q[:min_nb_episodes]
    
    # Plot the figures:
    plt.figure(1)
    plt.plot(range(0,min_nb_episodes,10),sr_record_S[0::10], label=SARSA)
    plt.plot(range(0,min_nb_episodes,10),sr_record_Q[0::10], label=Q_LEARNING)
    plt.title("Success rate in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    #plt.savefig("Simulations/CompareRL_Algos/Figures/successRate_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORE_METHOD, eps_decay, size_maze))
    
    plt.figure(2)
    plt.plot(range(0,min_nb_episodes,10),tr_record_S[0::10], label=SARSA)
    plt.plot(range(0,min_nb_episodes,10),tr_record_Q[0::10], label=Q_LEARNING)
    plt.title("Total reward in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    #plt.savefig("Simulations/CompareRL_Algos/Figures/totalReward_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORE_METHOD, eps_decay, size_maze))
    
    plt.figure(3)
    plt.plot(range(0,min_nb_episodes,10),rt_record_S[0::10], label=SARSA)
    plt.plot(range(0,min_nb_episodes,10),rt_record_Q[0::10], label=Q_LEARNING)
    plt.title("Resolution time in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    #plt.savefig("Simulations/CompareRL_Algos/Figures/resolutionTime_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORE_METHOD, eps_decay, size_maze))
    plt.show()
    
    return sr_record_S, sr_record_Q, tr_record_S, tr_record_Q, rt_record_S, rt_record_Q




## TO ANALYSE THE TRADE-OFF BETWEEN EXPLORATION AND EXPLOITATION
def trade_off():
    print("function trade_off to implement !")



## TO ANALYZE THE DIFFERENCE BETWEEN EPSILON_GREEDY AND SOFTMAX EXPLORATIONS
def exploration_method():
    print("function exploration_method to implement !")



if __name__ == "__main__":
    if CHOICE == 1:
        sr_S, sr_Q, tr_S, tr_Q, rt_S, rt_Q = compare_RL_algos()
    elif CHOICE == 2:
        trade_off()
    elif CHOICE == 3:
        exploration_method()
    else:
        print("CHOICE muste be 1, 2 or 3 !")
