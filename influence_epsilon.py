#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:17:37 2020

@author: Romain
"""

import sys
import numpy as np
import math
import random
from collections import deque
import gym
import gym_maze
import matplotlib.pyplot as plt

# save numpy array as npy file
from numpy import asarray
from numpy import save

gamma = 0.99
learning_rate = 0.1
verbose = True
RENDER_MAZE = False
PRINT = False 

# Define types of algorithms:
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
# ... and methods of exploration:
SOFTMAX = "softmax"
EPSILON_GREEDY = "epsilon_greedy"

NUM_EPISODES = 1000
STREAK_TO_END = 120 # number of "success" (i.e. how quick the maze is solved) needed to assess the good performance of a process


# A typical command to use this program from terminal is: python file.py size_maze RL_ALGO seed

"""
We will use 6 labels to compare the algoithms: 

    # ε = 0 --> greedy (label=1)
    # ε = 0.01 (label=2)
    # ε = 0.1 (label=3)
    # ε = 0.5 & ε_decay=0.9 (label=4)
    # ε = 0.5 & ε_decay=0.99 (label=5)
    # ε = 0.5 & ε_decay=0.999 (label=6)
    
The first one enhances the exploitation (very greedy) and the following algorithms are more and more exploring paths randomly. 

"""

label1 = "ε = 0"
label2 = "ε = 0.01"
label3 = "ε = 0.1"
label4 = "ε = 0.5 & ε_decay=0.9"
label5 = "ε = 0.5 & ε_decay=0.99"
label6 = "ε = 0.5 & ε_decay=0.999"


# Beware states are tuple of size 2. To facilitate, we will use (i,j)=s beforehand.

# Compute SARSA update
def sarsa_update(q,s,a,r,s_prime,a_prime,learning_rate):
    (i, j) = s
    (i_prime, j_prime) = s_prime 
    td = r + gamma * q[i_prime, j_prime, a_prime] - q[i, j, a]
    return q[i, j, a] + learning_rate * td

# Compute Q-Learning update
def q_learning_update(q,s,a,r,s_prime, learning_rate):
    (i, j) = s
    (i_prime, j_prime) = s_prime 
    td = r + gamma * np.max(q[i_prime, j_prime, :]) - q[i, j, a]
    return q[i, j, a] + learning_rate * td

# Draw a softmax sample but needs to improve !
def softmax(q, tau):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

# Act with softmax
def act_with_softmax(s, q, tau):
    (i,j) = s
    prob_a = softmax(q[i, j, :], tau)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q, epsilon, env):
    a = int(np.argmax(q[s]))
    if np.random.rand() < epsilon:
        a = env.action_space.sample()
    return a



# env.reset() and env.step(action)[0] both return a state with shape (2,) that needs to be converted into a tuple for our us
def getTupple(s): 
    i, j = int(s[0]), int(s[1])
    return (i,j)


















def simulation(size_maze, RL_ALGO, seed=13, eps_start=0.5, eps_decay=1, EXPLORE_METHOD=EPSILON_GREEDY):

    # Random seed
    random.seed(seed)
    
    # Initialize the "maze" environment with the given size
    env_name = "maze-random-" + str(size_maze) + "x" + str(size_maze) + "-plus-v0"
    env = gym.make(env_name)

    '''
    Defining the environment related constants
    '''
    
    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # = 4 : ["N", "S", "E", "W"]
    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.01
    DECAY_FACTOR = size_maze * size_maze / 10.0
    
    tau = init_tau = 1
    tau_inc = 0.01

    '''
    Defining the simulation related constants
    '''
    MAX_T = size_maze * size_maze * 100 # limit of steps in one episode after time out 
    SOLVED_T = size_maze * size_maze # number of step not to exceed to have a success in a give episode

    '''
    Creating a Q-Table for each state-action pair
        q_table is a tensor of shape (size_maze, size_maze, 4)
    '''
    q_table = np.zeros((size_maze, size_maze, NUM_ACTIONS)) 

    # Store parametres and sucess rate
    sr_record = []
    tr_record = [] # total reward
    resolution_time = []
    window = deque(maxlen = 120)

    # Instantiating the learning related parameters
    EPS_DECAY = (eps_decay >= 0) # True --> update of eps with eps_decay, False --> eps := max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((episode+1)/DECAY_FACTOR)))
    if EPS_DECAY:
        epsilon = 0.5
    else: 
        epsilon = 0.8

    num_streaks = 0

    # Render the maze
    if RENDER_MAZE:
        env.render()

    for episode in range(NUM_EPISODES):
        
        # Reset the environment and get initial state 
        s0_array = env.reset() # array([0., 0.]) (shape = (2,)) --> first state of the episode (top left corner)
        state = getTupple(s0_array)
        
        total_reward = 0

        # Select the first action
        if EXPLORE_METHOD == SOFTMAX:
            action = act_with_softmax(state, q_table, tau)
        elif EXPLORE_METHOD == EPSILON_GREEDY:
            action = act_with_epsilon_greedy(state, q_table, epsilon, env)
        else:
            raise ValueError("Wrong Explore Method:".format(EXPLORE_METHOD))

        for t in range(MAX_T):

            # Act
            obv, reward, done, info = env.step(action)

            # Observe the result
            state_prime = getTupple(obv) # format of a state
            total_reward += np.power(gamma,episode) * reward

            # Select an action
            if EXPLORE_METHOD == SOFTMAX:
                action_prime = act_with_softmax(state_prime, q_table, tau)
            elif EXPLORE_METHOD == EPSILON_GREEDY:
                action_prime = act_with_epsilon_greedy(state_prime, q_table, epsilon, env)
            
            (i, j) = state
            if RL_ALGO == SARSA:
                q_table[i, j, action] = sarsa_update(q_table,state,action,reward,state_prime,action_prime,learning_rate)
            elif RL_ALGO == Q_LEARNING:
                q_table[i, j, action] = q_learning_update(q_table,state,action,reward,state_prime,learning_rate)

            # Setting up for the next iteration
            state = state_prime
            action = action_prime
            
            # Render the maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                if PRINT:
                    print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                    window.append(1)
                else:
                    num_streaks = 0
                    window.append(0)
                break

            elif t >= MAX_T - 1:
                window.append(0)
                if PRINT: 
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, t, total_reward))


        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Computes sucess rate
        sr_record.append(window.count(1)/np.size(window))

        # Store resolution time
        resolution_time.append(t)
        
        # Store total_reward
        tr_record.append(total_reward)

        
        # Update parameters
        if (EPS_DECAY):
            epsilon = epsilon * eps_decay
        else: 
            epsilon = max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((episode+1)/DECAY_FACTOR)))
        tau = init_tau + episode * tau_inc
    return episode, sr_record, tr_record, resolution_time
    
    
    

if __name__ == "__main__":
    if ((len(sys.argv) <4) or ((int(sys.argv[1]) != 10) and (int(sys.argv[1]) != 20) and (int(sys.argv[1]) != 50))):
        print("The correct syntax is : \n")
        print("python influence_epsilon.py size_maze ALGO seed\n")
        print("where     size_maze is either 10 or 20 or 50")
        print("          ALGO denotes the RL algorithm used, either \"SARSA\" or \"Q_learning\" for Q-LEARNING")
        print("          seed initiates the random process deterministically")
        sys.exit(1)
        
    # we define the arguments/parameters as variables
    size_maze = int(sys.argv[1])
    ALGO = sys.argv[2] # either SARSA or Q_LEARNING
    SEED = int(sys.argv[3])
    
    EXPLORE_METHOD = EPSILON_GREEDY
    
    n_episode_1, sr_1, tr_1, resolution_time_1 = simulation(size_maze, ALGO, seed=SEED, eps_start=0) # ε = 0 --> greedy (label=1)
    n_episode_2, sr_2, tr_2, resolution_time_2 = simulation(size_maze, ALGO, seed=SEED, eps_start=0.01) # ε = 0.01 (label=2)
    n_episode_3, sr_3, tr_3, resolution_time_3 = simulation(size_maze, ALGO, seed=SEED, eps_start=0.1) # ε = 0.1 (label=3)
    n_episode_4, sr_4, tr_4, resolution_time_4 = simulation(size_maze, ALGO, seed=SEED, eps_start=0.5, eps_decay=0.9) # ε = 0.5 & ε_decay=0.9 (label=4)
    n_episode_5, sr_5, tr_5, resolution_time_5 = simulation(size_maze, ALGO, seed=SEED, eps_start=0.5, eps_decay=0.99) # ε = 0.5 & ε_decay=0.99 (label=5)
    n_episode_6, sr_6, tr_6, resolution_time_6 = simulation(size_maze, ALGO, seed=SEED, eps_start=0.5, eps_decay=0.999) # ε = 0.5 & ε_decay=0.999 (label=6)
    
    
    
    # Plot results
    # SUCCESS RATE:
    plt.figure(0)
    plt.plot(range(0,n_episode_1,10),sr_1[0::10], label=label1)
    plt.plot(range(0,n_episode_2,10),sr_2[0::10], label=label2)
    plt.plot(range(0,n_episode_3,10),sr_3[0::10], label=label3)
    plt.plot(range(0,n_episode_4,10),sr_4[0::10], label=label4)
    plt.plot(range(0,n_episode_5,10),sr_5[0::10], label=label5)
    plt.plot(range(0,n_episode_6,10),sr_6[0::10], label=label6)
    plt.title("Influence of epsilon on the success rate with {} training. Maze of size = {}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Simulations/influence_epsilon/size{0}/{1}/figures/epsilon-influence_successRate.png".format(size_maze, ALGO))
    
    # TOTAL REWARD:
    plt.figure(1)
    plt.plot(range(0,n_episode_1,10),tr_1[0::10], label=label1)
    plt.plot(range(0,n_episode_2,10),tr_2[0::10], label=label2)
    plt.plot(range(0,n_episode_3,10),tr_3[0::10], label=label3)
    plt.plot(range(0,n_episode_4,10),tr_4[0::10], label=label4)
    plt.plot(range(0,n_episode_5,10),tr_5[0::10], label=label5)
    plt.plot(range(0,n_episode_6,10),tr_6[0::10], label=label6)
    plt.title("Influence of epsilon on the total reward with {} training. Maze of size = {}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Simulations/influence_epsilon/size{0}/{1}/figures/epsilon-influence_totalReward.png".format(size_maze, ALGO))
    
    
    # RESOLUTION TIME
    plt.figure(2)
    plt.plot(range(0, n_episode_1, 10),resolution_time_1[0::10], label=label1)
    plt.plot(range(0, n_episode_2, 10),resolution_time_2[0::10], label=label2)
    plt.plot(range(0, n_episode_3, 10),resolution_time_3[0::10], label=label3)
    plt.plot(range(0, n_episode_4, 10),resolution_time_4[0::10], label=label4)
    plt.plot(range(0, n_episode_5, 10),resolution_time_5[0::10], label=label5)
    plt.plot(range(0, n_episode_6, 10),resolution_time_6[0::10], label=label6)
    plt.title("Influence of epsilon on the resolution time with {} training. Maze of size = {}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    plt.savefig("Simulations/influence_epsilon/size{0}/{1}/figures/epsilon-influence_resolutionTime.png".format(size_maze, ALGO))
    plt.show()



    if ((ALGO == "SARSA") or (ALGO == "Q_learning")):
        sr1 = asarray(sr_1)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr1.npy'.format(size_maze, ALGO), sr1)
        sr2 = asarray(sr_2)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr2.npy'.format(size_maze, ALGO), sr2)
        sr3 = asarray(sr_3)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr3.npy'.format(size_maze, ALGO), sr3)
        sr4 = asarray(sr_4)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr4.npy'.format(size_maze, ALGO), sr4)
        sr5 = asarray(sr_5)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr5.npy'.format(size_maze, ALGO), sr5)
        sr6 = asarray(sr_6)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr6.npy'.format(size_maze, ALGO), sr6)
        
        tr1 = asarray(tr_1)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr1.npy'.format(size_maze, ALGO), sr1)
        tr2 = asarray(tr_2)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr2.npy'.format(size_maze, ALGO), sr2)
        tr3 = asarray(tr_3)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr3.npy'.format(size_maze, ALGO), sr3)
        tr4 = asarray(tr_4)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr4.npy'.format(size_maze, ALGO), sr4)
        tr5 = asarray(tr_5)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr5.npy'.format(size_maze, ALGO), sr5)
        tr6 = asarray(tr_6)
        save('Simulations/influence_epsilon/size{0}/{1}/success_rate/sr6.npy'.format(size_maze, ALGO), sr6)
    
        resolution_time1 = asarray(resolution_time_1)
        save('Simulations/influence_epsilon/size{0}/{1}/resolution_time/rt1.npy'.format(size_maze, ALGO), resolution_time1)
        resolution_time2 = asarray(resolution_time_2)
        save('Simulations/influence_epsilon/size{0}/{1}/resolution_time/rt2.npy'.format(size_maze, ALGO), resolution_time2)
        resolution_time3 = asarray(resolution_time_3)
        save('Simulations/influence_epsilon/size{0}/{1}/resolution_time/rt3.npy'.format(size_maze, ALGO), resolution_time3)
        resolution_time4 = asarray(resolution_time_4)
        save('Simulations/influence_epsilon/size{0}/{1}/resolution_time/rt4.npy'.format(size_maze, ALGO), resolution_time4)
        resolution_time5 = asarray(resolution_time_5)
        save('Simulations/influence_epsilon/size{0}/{1}/resolution_time/rt5.npy'.format(size_maze, ALGO), resolution_time5)
        resolution_time6 = asarray(resolution_time_6)
        save('Simulations/influence_epsilon/size{0}/{1}/resolution_time/rt6.npy'.format(size_maze, ALGO), resolution_time6)

