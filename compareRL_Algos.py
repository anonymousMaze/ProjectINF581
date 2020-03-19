import sys
import matplotlib.pyplot as plt

from simulation import simulation

# save numpy array as npy files
from numpy import asarray
from numpy import save

# The RL algorithms:
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
# ... and the methods for exploration:
SOFTMAX = "softmax"
EPSILON_GREEDY = "epsilon_greedy"

# A typical command to use this program from terminal is: python compareRL_Algos.py size_maze exploration_method eps_decay seed

if __name__ == "__main__":
    if ((len(sys.argv) < 5) or ((int(sys.argv[1]) != 10) and (int(sys.argv[1]) != 20)  and (int(sys.argv[1]) != 30))):
        print("The correct syntax is : \n")
        print("python compareRL_Algos.py size_maze exploration_method eps_decay seed\n")
        print("where     size_maze is either 10 or 20 or 50")
        print("          exploration_method denotes either \"epsilon_greedy\" or \"softmax\" (the latter does not work for the moment...)")
        print("          if eps_decay>=0: eps<-eps*eps_decay ; otherwise, other update of eps")
        print("          seed initializes the random process deterministically")
        sys.exit(1)
        
    # we define the arguments/parameters as variables
    size_maze = int(sys.argv[1])
    EXPLORE_METHOD = sys.argv[2] # either SOFTMAX, EPSILON_GREEDY
    eps_decay = float(sys.argv[3])
    SEED = int(sys.argv[4])
    
    n_episode_S, sr_S, tr_S, rt_S = simulation(SARSA, EXPLORE_METHOD, 0.5, eps_decay, size_maze, SEED)
    n_episode_Q, sr_Q, tr_Q, rt_Q = simulation(Q_LEARNING, EXPLORE_METHOD, 0.5, eps_decay, size_maze, SEED)
    
    plt.figure(1)
    plt.plot(range(0,n_episode_S,10),sr_S[0::10], label=SARSA)
    plt.plot(range(0,n_episode_Q,10),sr_Q[0::10], label=Q_LEARNING)
    plt.title("Success rate in function of the episodes with ε_decay={0} and size={1}".format(eps_decay, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Simulations/CompareRL_Algos/Figures/successRate_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORE_METHOD, eps_decay, size_maze))
    
    plt.figure(2)
    plt.plot(range(0,n_episode_S,10),tr_S[0::10], label=SARSA)
    plt.plot(range(0,n_episode_Q,10),tr_Q[0::10], label=Q_LEARNING)
    plt.title("Total reward in function of the episodes with ε_decay={0} and size={1}".format(eps_decay, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("Simulations/CompareRL_Algos/Figures/totalReward_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORE_METHOD, eps_decay, size_maze))
    
    plt.figure(3)
    plt.plot(range(0,n_episode_S,10),rt_S[0::10], label=SARSA)
    plt.plot(range(0,n_episode_Q,10),rt_Q[0::10], label=Q_LEARNING)
    plt.title("Resolution time in function of the episodes with ε_decay={0} and size={1}".format(eps_decay, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    plt.savefig("Simulations/CompareRL_Algos/Figures/resolutionTime_exploration_with_{0}_decay{1}_size{2}.png".format(EXPLORE_METHOD, eps_decay, size_maze))
    plt.show()
    
    
    # and we store the arrays in .npy files
    ALGO = SARSA
    srS = asarray(sr_S)
    save('Simulations/CompareRL_Algos/Arrays/SuccessRate/srS-exploration_with_{0}_{1}_eps_decay{2}_size{3}_seed{4}.npy'.format(EXPLORE_METHOD, ALGO, eps_decay, size_maze, SEED), srS)
    trS = asarray(tr_S)
    save('Simulations/CompareRL_Algos/Arrays/TotalReward/trS-exploration_with_{0}_{1}_eps_decay{2}_size{3}_seed{4}.npy'.format(EXPLORE_METHOD, ALGO, eps_decay, size_maze, SEED), trS)
    rtS = asarray(rt_S)
    save('Simulations/CompareRL_Algos/Arrays/ResolutionTime/rtS-exploration_with_{0}_{1}_eps_decay{2}_size{3}_seed{4}.npy'.format(EXPLORE_METHOD, ALGO, eps_decay, size_maze, SEED), rtS)
        
    ALGO = Q_LEARNING
    srQ = asarray(sr_Q)
    save('Simulations/CompareRL_Algos/Arrays/SuccessRate/srQ-exploration_with_{0}_{1}_eps_decay{2}_size{3}_seed{4}.npy'.format(EXPLORE_METHOD, ALGO, eps_decay, size_maze, SEED), srQ)
    trQ = asarray(tr_Q)
    save('Simulations/CompareRL_Algos/Arrays/TotalReward/trQ-exploration_with_{0}_{1}_eps_decay{2}_size{3}_seed{4}.npy'.format(EXPLORE_METHOD, ALGO, eps_decay, size_maze, SEED), trQ)
    rtQ = asarray(rt_Q)
    save('Simulations/CompareRL_Algos/Arrays/ResolutionTime/rtQ-exploration_with_{0}_{1}_eps_decay{2}_size{3}_seed{4}.npy'.format(EXPLORE_METHOD, ALGO, eps_decay, size_maze, SEED), rtQ)
        