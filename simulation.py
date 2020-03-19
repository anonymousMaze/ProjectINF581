import sys
import numpy as np
import math
import random
from collections import deque
import gym
import gym_maze

# The RL algorithms:
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
# ... and the methods for exploration:
SOFTMAX = "softmax"
EPSILON_GREEDY = "epsilon_greedy"

gamma = 0.99 # discount factor
learning_rate = 0.1
tau_inc = 0.01 # to update tau for softmax purpose
verbose = True
RENDER_MAZE = False
PRINT = False 
NUM_EPISODES = 1000
STREAK_TO_END = 120 # number of "success" (i.e. how quick the maze is solved) needed to assess the good performance of a process


# Two Reinforcement algorithms:
def sarsa_update(q,s,a,r,s_prime,a_prime,learning_rate):
    (i, j) = s # Beware states are tuple of size 2.
    (i_prime, j_prime) = s_prime 
    td = r + gamma * q[i_prime, j_prime, a_prime] - q[i, j, a]
    return q[i, j, a] + learning_rate * td

def q_learning_update(q,s,a,r,s_prime,learning_rate):
    (i, j) = s
    (i_prime, j_prime) = s_prime 
    td = r + gamma * np.max(q[i_prime, j_prime, :]) - q[i, j, a]
    return q[i, j, a] + learning_rate * td


# Two exploration methods
def softmax(q, tau):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

def act_with_softmax(s, q, tau):
    (i,j) = s
    prob_a = softmax(q[i, j, :], tau)
    cumsum_a = np.cumsum(prob_a)
    return int(np.where(np.random.rand() < cumsum_a)[0][0])


def act_with_epsilon_greedy(s, q, epsilon, env):
    a = int(np.argmax(q[s]))
    if np.random.rand() < epsilon:
        a = env.action_space.sample()
    return a


# env.reset() and env.step(action)[0] both return a state with shape (2,) that needs to be converted into a tuple for our us
def getTuple(s): # array([x, y]) --> (x, y)
    i, j = int(s[0]), int(s[1])
    return (i,j)





# Main function of the project
def simulation(RL_ALGO=SARSA, EXPLORE_METHOD=EPSILON_GREEDY, eps_start=0.5, eps_decay=1, size_maze=10, seed=13):

    # Random seed
    random.seed(seed)
    
    # Initialize the "maze" environment with the given size
    env_name = "maze-random-" + str(size_maze) + "x" + str(size_maze) + "-plus-v0"
    env = gym.make(env_name)
    
    NUM_ACTIONS = env.action_space.n  # = 4 : ["N", "S", "E", "W"]
    
    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.01
    DECAY_FACTOR = size_maze * size_maze / 10.0
    tau = 1 # used for softmax exploration

    '''
    Defining the simulation related constants
    '''
    MAX_T = size_maze * size_maze * 100 # limit of steps in one episode after time out 
    SOLVED_T = size_maze * size_maze # number of step not to exceed to have a success in a give episode

    '''
    Creating a Q-Table for each state-action pair
        q_table is a tensor of shape (size_maze, size_maze, NUM_ACTIONS)
    '''
    q_table = np.zeros((size_maze, size_maze, NUM_ACTIONS)) 

    # Store parametres
    sr_record = [] # success rate
    tr_record = [] # total reward
    rt_record = [] # resolution time
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
        state = getTuple(s0_array)
        total_reward = 0

        # Select the first action
        if EXPLORE_METHOD == SOFTMAX:
            action = act_with_softmax(state, q_table, tau)
        elif EXPLORE_METHOD == EPSILON_GREEDY:
            action = act_with_epsilon_greedy(state, q_table, epsilon, env)
        else:
            raise ValueError("Wrong Explore Method:".format(EXPLORE_METHOD))

        for t in range(MAX_T):
            
            action = int(action)
            obv, reward, done, info = env.step(action)
            state_prime = getTuple(obv)
            total_reward += np.power(gamma, episode) * reward

            # Select an action
            if EXPLORE_METHOD == SOFTMAX:
                action_prime = act_with_softmax(state_prime, q_table, tau)
            elif EXPLORE_METHOD == EPSILON_GREEDY:
                action_prime = act_with_epsilon_greedy(state_prime, q_table, epsilon, env)
            
            # Update of the Q-table wrt RL Algo
            (i, j) = state
            if RL_ALGO == SARSA:
                q_table[i, j, action] = sarsa_update(q_table,state,action,reward,state_prime,action_prime,learning_rate)
            elif RL_ALGO == Q_LEARNING:
                q_table[i, j, action] = q_learning_update(q_table,state,action,reward,state_prime,learning_rate)

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

                if t <= SOLVED_T: # episode solved quickly enough
                    num_streaks += 1
                    window.append(1)
                else:
                    num_streaks = 0
                    window.append(0)
                break

            elif t >= MAX_T - 1: # TIME OUT
                window.append(0)
                if PRINT: 
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Computes success rate
        sr_record.append(window.count(1)/np.size(window))
        
        # Store total_reward
        tr_record.append(total_reward)
       
        # Store resolution time
        rt_record.append(t)
        
        # Update parameters
        if (EPS_DECAY):
            epsilon = epsilon * eps_decay
        else: 
            epsilon = max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((episode+1)/DECAY_FACTOR)))
        tau += episode * tau_inc
    return episode, sr_record, tr_record, rt_record
    
# ep, sr, tr, rt = simulation(SARSA, EPSILON_GREEDY)

