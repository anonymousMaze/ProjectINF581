import sys
import numpy as np
import math
import random
from collections import deque
import gym
import gym_maze
import matplotlib.pyplot as plt

# Meta parameters for the RL agent
#tau = init_tau = 1
#tau_inc = 0.01
gamma = 0.99
learning_rate = 0.1
epsilon = 0.5
epsilon_decay = 0.999
verbose = True
RENDER_MAZE = False
PRINT = False 

# Define types of algorithms:
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
# ... and methods of exploration:
SOFTMAX = "softmax"
EPSILON_GREEDY = "epsilon_greedy"
GREEDY = "greedy"

NUM_EPISODES = 1000



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

# Draw a softmax sample
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



# A typical command is: python mazeSolver4.py size_maze ALGO exploration seed
def simulation(size_maze, RL_ALGO, EXPLORE_METHOD, eps_decay, seed):

    # Random seed
    np.random.RandomState(seed)
    
    # Initialize the "maze" environment with the given size
    env_name = "maze-random-" + str(size_maze) + "x" + str(size_maze) + "-plus-v0"
    env = gym.make(env_name)

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int)) # = (size_maze, size_maze)
    #MAZE_SIZE = (size_maze, size_maze)
    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.01
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
    
    tau = init_tau = 1
    tau_inc = 0.01

    '''
    Defining the simulation related constants
    '''
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100 # limit of steps in one episode after time out 
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int) # number of step not to exceed to have a success in a give episode
    STREAK_TO_END = 120 # number of "success" needed to assess the good performance of a process

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
    if ((len(sys.argv) < 5) or ((int(sys.argv[1]) != 10) and (int(sys.argv[1]) != 20))):
        print("The correct syntax is : \n")
        print("python compareRL_Algos.py size_maze exploration eps_decay seed\n")
        print("where     size_maze is either 10 or 20")
        print("          exploration denotes the exploration either \"epsilon_greedy\" or \"softmax\" (the latter does not work for the moment...)")
        print("          if eps_decay>=0: eps<-eps*eps_decay ; otherwise, other update of eps")
        print("          seed initiates the random process deterministically")
        sys.exit(1)
        
    # we define the arguments/parameters as variables
    size_maze = int(sys.argv[1])
    EXPLORE_METHOD = sys.argv[2] # either SOFTMAX, EPSILON_GREEDY, GREEDY
    eps_decay = float(sys.argv[3])
    seed = int(sys.argv[4])
    
    n_episode_S, sr_S, tr_S, resolution_time_S = simulation(size_maze, SARSA, EXPLORE_METHOD, eps_decay, seed)
    n_episode_Q, sr_Q, tr_Q, resolution_time_Q = simulation(size_maze, Q_LEARNING, EXPLORE_METHOD, eps_decay, seed)
    
    plt.figure(1)
    plt.plot(range(0,n_episode_S,10),sr_S[0::10], label=SARSA)
    plt.plot(range(0,n_episode_Q,10),sr_Q[0::10], label=Q_LEARNING)
    plt.title("Success rate in function of the episodes with size={0}".format(size_maze))
    plt.legend()
    
    plt.figure(2)
    plt.plot(range(0,n_episode_S,10),tr_S[0::10], label=SARSA)
    plt.plot(range(0,n_episode_Q,10),tr_Q[0::10], label=Q_LEARNING)
    plt.title("Total reward in function of the episodes with size={0}".format(size_maze))
    plt.legend()
    
    plt.figure(3)
    plt.plot(range(0,n_episode_S,10),resolution_time_S[0::10], label=SARSA)
    plt.plot(range(0,n_episode_Q,10),resolution_time_Q[0::10], label=Q_LEARNING)
    plt.title("Resolution time in function of the episodes with size={0}".format(size_maze))
    plt.legend()
    plt.show()