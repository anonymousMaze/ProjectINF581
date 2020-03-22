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


# A[] <- A[] + a[]    with A of size NUM_EPISODES and len(a) < NUM_EPISODES
def sum_arrays(A, a):
    assert((len(A) == NUM_EPISODES) and (len(a) < NUM_EPISODES))
    for i in range(len(a)):
        A[i] += a[i]





# - - - - - - - - - - - - - - I N S T R U C T I O N S - - - - - - - - - - - - - 
        
"""
This file is used is executing simulation() function over a batch to get more relevant results to discuss.
"""
# CHOICE = 1 ---> TO COMPARE SARSA AND Q_LEARNING
# CHOICE = 2 ---> TO ANALYSE THE TRADE-OFF BETWEEN EXPLORATION AND EXPLOITATION
# CHOICE = 3 ---> TO COMPARE THE EXPLORATION METHODS : EPSILON_GREEDY VS SOFTMAX 

CHOICE = 3 # 1, 2 or 3


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 















def compare_RL_algos(EXPLORATION_METHOD = EPSILON_GREEDY, eps_start = 0.5, eps_decay = 0.9):
    # sr_record_S = sr_record_Q = tr_record_S = tr_record_Q = rt_record_S = rt_record_Q = np.zeros(NUM_EPISODES)
    
    sr_record_S = np.zeros(NUM_EPISODES)
    sr_record_Q = np.zeros(NUM_EPISODES)
    
    tr_record_S = np.zeros(NUM_EPISODES)
    tr_record_Q = np.zeros(NUM_EPISODES)
    
    rt_record_S = np.zeros(NUM_EPISODES)
    rt_record_Q = np.zeros(NUM_EPISODES)
    
    min_nb_episodes = np.inf 
    
    perm = np.random.permutation(1000) # shuffle to
    for k in range(size_batch): #            select a seed
        seed = perm[k] #                        randomly
        
        n_episode_S, sr_S, tr_S, rt_S = simulation(SARSA, EXPLORATION_METHOD, eps_start, eps_decay, size_maze, seed)
        n_episode_Q, sr_Q, tr_Q, rt_Q = simulation(Q_LEARNING, EXPLORATION_METHOD, eps_start, eps_decay, size_maze, seed)
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
    sr_record_S /= size_batch; sr_record_S = sr_record_S[:min_nb_episodes]
    sr_record_Q /= size_batch; sr_record_Q = sr_record_Q[:min_nb_episodes]
    
    tr_record_S /= size_batch; tr_record_S = tr_record_S[:min_nb_episodes]
    tr_record_Q /= size_batch; tr_record_Q = tr_record_Q[:min_nb_episodes]
    
    rt_record_S /= size_batch; rt_record_S = rt_record_S[:min_nb_episodes]
    rt_record_Q /= size_batch; rt_record_Q = rt_record_Q[:min_nb_episodes]
    
    # Plot the figures:
    plt.figure(1)
    plt.plot(range(0,min_nb_episodes,10),sr_record_S[0::10], label=SARSA)
    plt.plot(range(0,min_nb_episodes,10),sr_record_Q[0::10], label=Q_LEARNING)
    plt.title("Success rate in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Figures/CompareRL_Algos/successRate_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORATION_METHOD, eps_decay, size_maze))
    
    plt.figure(2)
    plt.plot(range(0,min_nb_episodes,10),tr_record_S[0::10], label=SARSA)
    plt.plot(range(0,min_nb_episodes,10),tr_record_Q[0::10], label=Q_LEARNING)
    plt.title("Total reward in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("Figures/CompareRL_Algos/totalReward_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORATION_METHOD, eps_decay, size_maze))
    
    plt.figure(3)
    plt.plot(range(0,min_nb_episodes,10),rt_record_S[0::10], label=SARSA)
    plt.plot(range(0,min_nb_episodes,10),rt_record_Q[0::10], label=Q_LEARNING)
    plt.title("Resolution time in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    plt.savefig("Figures/CompareRL_Algos/resolutionTime_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORATION_METHOD, eps_decay, size_maze))
    plt.show()
    
    # return sr_record_S, sr_record_Q, tr_record_S, tr_record_Q, rt_record_S, rt_record_Q





















"""
We will use 6 labels to compare the two algorithms.

The first one enhances the exploitation (very greedy ε = 0) and the following algorithms are more and more exploring paths randomly. 
"""
label1 = "ε = 0"
label2 = "ε = 0.01"
label3 = "ε = 0.1"
label4 = "ε = 0.5 & ε_decay=0.9"
label5 = "ε = 0.5 & ε_decay=0.99"
label6 = "ε = 0.5 & ε_decay=0.999"


def trade_off(ALGO = SARSA):
    """
    ---------------------------------------------------------------------------
      
    """

    sr_record_1 = np.zeros(NUM_EPISODES)
    sr_record_2 = np.zeros(NUM_EPISODES)
    sr_record_3 = np.zeros(NUM_EPISODES)
    sr_record_4 = np.zeros(NUM_EPISODES)
    sr_record_5 = np.zeros(NUM_EPISODES)
    sr_record_6 = np.zeros(NUM_EPISODES)
    
    tr_record_1 = np.zeros(NUM_EPISODES)
    tr_record_2 = np.zeros(NUM_EPISODES)
    tr_record_3 = np.zeros(NUM_EPISODES)
    tr_record_4 = np.zeros(NUM_EPISODES)
    tr_record_5 = np.zeros(NUM_EPISODES)
    tr_record_6 = np.zeros(NUM_EPISODES)
    
    rt_record_1 = np.zeros(NUM_EPISODES)
    rt_record_2 = np.zeros(NUM_EPISODES)
    rt_record_3 = np.zeros(NUM_EPISODES)
    rt_record_4 = np.zeros(NUM_EPISODES)
    rt_record_5 = np.zeros(NUM_EPISODES)
    rt_record_6 = np.zeros(NUM_EPISODES)

    min_nb_episodes = np.inf
    
    perm = np.random.permutation(1000) # shuffle to
    for k in range(size_batch): #            select a seed
        seed = perm[k] #                       randomly

        n_episode_1, sr_1, tr_1, rt_1 = simulation(ALGO, EPSILON_GREEDY, 0, 1, size_maze, seed) # ε = 0 --> greedy (label=1)
        n_episode_2, sr_2, tr_2, rt_2 = simulation(ALGO, EPSILON_GREEDY, 0.01, 1, size_maze, seed) # ε = 0.01 (label=2)
        n_episode_3, sr_3, tr_3, rt_3 = simulation(ALGO, EPSILON_GREEDY, 0.1, 1, size_maze, seed) # ε = 0.1 (label=3) 
        n_episode_4, sr_4, tr_4, rt_4 = simulation(ALGO, EPSILON_GREEDY, 0.5, 0.9, size_maze, seed) # ε = 0.5 & ε_decay=0.9 (label=4)
        n_episode_5, sr_5, tr_5, rt_5 = simulation(ALGO, EPSILON_GREEDY, 0.5, 0.99, size_maze, seed) # ε = 0.5 & ε_decay=0.99 (label=5)
        n_episode_6, sr_6, tr_6, rt_6 = simulation(ALGO, EPSILON_GREEDY, 0.5, 0.999, size_maze, seed) # ε = 0.5 & ε_decay=0.999 (label=6)
    
        sum_arrays(sr_record_1, np.array(sr_1))
        sum_arrays(sr_record_2, np.array(sr_2))
        sum_arrays(sr_record_3, np.array(sr_3))
        sum_arrays(sr_record_4, np.array(sr_4))
        sum_arrays(sr_record_5, np.array(sr_5))
        sum_arrays(sr_record_6, np.array(sr_6))
        
        sum_arrays(tr_record_1, np.array(tr_1))
        sum_arrays(tr_record_2, np.array(tr_2))
        sum_arrays(tr_record_3, np.array(tr_3))
        sum_arrays(tr_record_4, np.array(tr_4))
        sum_arrays(tr_record_5, np.array(tr_5))
        sum_arrays(tr_record_6, np.array(tr_6))
        
        sum_arrays(rt_record_1, np.array(rt_1))
        sum_arrays(rt_record_2, np.array(rt_2))
        sum_arrays(rt_record_3, np.array(rt_3))
        sum_arrays(rt_record_4, np.array(rt_4))
        sum_arrays(rt_record_5, np.array(rt_5))
        sum_arrays(rt_record_6, np.array(rt_6))
        
        n_episodes = min([n_episode_1, n_episode_2, n_episode_3, n_episode_4, n_episode_5, n_episode_6])
    
        if (n_episodes < min_nb_episodes):
           min_nb_episodes = n_episodes
    
    sr_record_1 /= size_batch; sr_record_1 = sr_record_1[:min_nb_episodes]
    sr_record_2 /= size_batch; sr_record_2 = sr_record_2[:min_nb_episodes]
    sr_record_3 /= size_batch; sr_record_3 = sr_record_3[:min_nb_episodes]
    sr_record_4 /= size_batch; sr_record_4 = sr_record_4[:min_nb_episodes]
    sr_record_5 /= size_batch; sr_record_5 = sr_record_5[:min_nb_episodes]
    sr_record_6 /= size_batch; sr_record_6 = sr_record_6[:min_nb_episodes]
    
    tr_record_1 /= size_batch; tr_record_1 = tr_record_1[:min_nb_episodes]
    tr_record_2 /= size_batch; tr_record_2 = tr_record_2[:min_nb_episodes]
    tr_record_3 /= size_batch; tr_record_3 = tr_record_3[:min_nb_episodes]
    tr_record_4 /= size_batch; tr_record_4 = tr_record_4[:min_nb_episodes]
    tr_record_5 /= size_batch; tr_record_5 = tr_record_5[:min_nb_episodes]
    tr_record_6 /= size_batch; tr_record_6 = tr_record_6[:min_nb_episodes]
    
    rt_record_1 /= size_batch; rt_record_1 = rt_record_1[:min_nb_episodes]
    rt_record_2 /= size_batch; rt_record_2 = rt_record_2[:min_nb_episodes]
    rt_record_3 /= size_batch; rt_record_3 = rt_record_3[:min_nb_episodes]
    rt_record_4 /= size_batch; rt_record_4 = rt_record_4[:min_nb_episodes]
    rt_record_5 /= size_batch; rt_record_5 = rt_record_5[:min_nb_episodes]
    rt_record_6 /= size_batch; rt_record_6 = rt_record_6[:min_nb_episodes]
    
    
    
    # Plot results
    # ... for SUCCESS RATE:
    plt.figure(1)
    plt.plot(range(0,min_nb_episodes,10),sr_record_1[0::10], label=label1)
    plt.plot(range(0,min_nb_episodes,10),sr_record_2[0::10], label=label2)
    plt.plot(range(0,min_nb_episodes,10),sr_record_3[0::10], label=label3)
    plt.plot(range(0,min_nb_episodes,10),sr_record_4[0::10], label=label4)
    plt.plot(range(0,min_nb_episodes,10),sr_record_5[0::10], label=label5)
    plt.plot(range(0,min_nb_episodes,10),sr_record_6[0::10], label=label6)
    plt.title("Influence of epsilon on the success rate with {} training. Size={}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Figures/Influence_Epsilon/{0}/influence_of_EpsilonONsuccessRate_with_ALGO_{0}_size{1}.png".format(ALGO, size_maze))
    
    
    # ... for TOTAL REWARD:
    plt.figure(2)
    plt.plot(range(0,min_nb_episodes,10),tr_record_1[0::10], label=label1)
    plt.plot(range(0,min_nb_episodes,10),tr_record_2[0::10], label=label2)
    plt.plot(range(0,min_nb_episodes,10),tr_record_3[0::10], label=label3)
    plt.plot(range(0,min_nb_episodes,10),tr_record_4[0::10], label=label4)
    plt.plot(range(0,min_nb_episodes,10),tr_record_5[0::10], label=label5)
    plt.plot(range(0,min_nb_episodes,10),tr_record_6[0::10], label=label6)
    plt.title("Influence of epsilon on the total reward with {} training. Size={}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("Figures/Influence_Epsilon/{0}/influence_of_EpsilonONtotalReward_with_ALGO_{0}_size{1}.png".format(ALGO, size_maze))
    
    
    # ... for RESOLUTION TIME
    plt.figure(3)
    plt.plot(range(0, min_nb_episodes, 10),rt_record_1[0::10], label=label1)
    plt.plot(range(0, min_nb_episodes, 10),rt_record_2[0::10], label=label2)
    plt.plot(range(0, min_nb_episodes, 10),rt_record_3[0::10], label=label3)
    plt.plot(range(0, min_nb_episodes, 10),rt_record_4[0::10], label=label4)
    plt.plot(range(0, min_nb_episodes, 10),rt_record_5[0::10], label=label5)
    plt.plot(range(0, min_nb_episodes, 10),rt_record_6[0::10], label=label6)
    plt.title("Influence of epsilon on the resolution time with {} training. Size={}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    plt.savefig("Figures/Influence_Epsilon/{0}/influence_of_EpsilonONresolutionTime_with_ALGO_{0}_size{1}.png".format(ALGO, size_maze))
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def exploration_method(ALGO = SARSA, eps_start = 0.5, eps_decay = 0.9):
    # sr_record_S = sr_record_Q = tr_record_S = tr_record_Q = rt_record_S = rt_record_Q = np.zeros(NUM_EPISODES)
    
    sr_record_Ep = np.zeros(NUM_EPISODES)
    sr_record_So = np.zeros(NUM_EPISODES)
    
    tr_record_Ep = np.zeros(NUM_EPISODES)
    tr_record_So = np.zeros(NUM_EPISODES)
    
    rt_record_Ep = np.zeros(NUM_EPISODES)
    rt_record_So = np.zeros(NUM_EPISODES)
    
    min_nb_episodes = np.inf 
    
    perm = np.random.permutation(1000) # shuffle to
    for k in range(size_batch): #            select a seed
        seed = perm[k] #                   randomly
        
        n_episode_Ep, sr_Ep, tr_Ep, rt_Ep = simulation(SARSA, EPSILON_GREEDY, eps_start, eps_decay, size_maze, seed)
        n_episode_So, sr_So, tr_So, rt_So = simulation(Q_LEARNING, SOFTMAX, eps_start, eps_decay, size_maze, seed)
        
        sum_arrays(sr_record_Ep, np.array(sr_Ep))
        sum_arrays(sr_record_So, np.array(sr_So))
        
        sum_arrays(tr_record_Ep, np.array(tr_Ep))
        sum_arrays(tr_record_So, np.array(tr_So))
        
        sum_arrays(rt_record_Ep, np.array(rt_Ep))
        sum_arrays(rt_record_So, np.array(rt_So))
        
        if (n_episode_Ep < min_nb_episodes):
            min_nb_episodes = n_episode_Ep
        if (n_episode_So < min_nb_episodes):
            min_nb_episodes = n_episode_So
    sr_record_Ep /= size_batch; sr_record_Ep = sr_record_Ep[:min_nb_episodes]
    sr_record_So /= size_batch; sr_record_So = sr_record_So[:min_nb_episodes]
    
    tr_record_Ep /= size_batch; tr_record_Ep = tr_record_Ep[:min_nb_episodes]
    tr_record_So /= size_batch; tr_record_So = tr_record_So[:min_nb_episodes]
    
    rt_record_Ep /= size_batch; rt_record_Ep = rt_record_Ep[:min_nb_episodes]
    rt_record_So /= size_batch; rt_record_So = rt_record_So[:min_nb_episodes]
    
    
    # Plot results
    # ... for SUCCESS RATE:
    plt.figure(1)
    plt.plot(range(0,min_nb_episodes,10),sr_record_Ep[0::10], color = "green", label=EPSILON_GREEDY)
    plt.plot(range(0,min_nb_episodes,10),sr_record_So[0::10], color = "red", label=SOFTMAX)
    plt.title("Success rate in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Figures/CompareExplorationMethods/successRate_with_ALGO_{0}_decay{1}_size{2}.png".format(ALGO, eps_decay, size_maze))
    
    # ... for TOTAL REWARD
    plt.figure(2)
    plt.plot(range(0,min_nb_episodes,10),tr_record_Ep[0::10], color = "green", label=EPSILON_GREEDY)
    plt.plot(range(0,min_nb_episodes,10),tr_record_So[0::10], color = "red", label=SOFTMAX)
    plt.title("Total reward in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("Figures/CompareExplorationMethods/totalReward_with_ALGO_{0}_decay{1}_size{2}.png".format(ALGO, eps_decay, size_maze))
    
    
    # ... for RESOLUTION TIME
    plt.figure(3)
    plt.plot(range(0,min_nb_episodes,10),rt_record_Ep[0::10], color = "green", label=EPSILON_GREEDY)
    plt.plot(range(0,min_nb_episodes,10),rt_record_So[0::10], color = "red", label=SOFTMAX)
    plt.title("Resolution time in function of the episodes with size_batch={0} and size_maze={1}".format(size_batch, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    plt.savefig("Figures/CompareExplorationMethods/resolutionTime_with_ALGO_{0}_decay{1}_size{2}.png".format(ALGO, eps_decay, size_maze))
    plt.show()
    
    #return sr_record_Ep, sr_record_So rt_record_Ep, rt_record_So


    
    
    
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



if __name__ == "__main__":
    if CHOICE == 1:
        compare_RL_algos()
    elif CHOICE == 2:
        trade_off()
    elif CHOICE == 3:
        exploration_method(ALGO = SARSA)
    else:
        print("CHOICE must be 1, 2 or 3!")
