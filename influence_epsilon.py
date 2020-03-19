import sys
import matplotlib.pyplot as plt

from simulation import simulation

# save numpy array as npy files
from numpy import asarray
from numpy import save


SARSA = "SARSA"
Q_LEARNING = "Q_learning"

EPSILON_GREEDY = "epsilon_greedy"


# A typical command to use this program from terminal is: python influence_epsilon.py size_maze RL_ALGO seed

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
    
    n_episode_1, sr_1, tr_1, rt_1 = simulation(ALGO, EPSILON_GREEDY, 0, 1, size_maze, SEED) # ε = 0 --> greedy (label=1)
    n_episode_2, sr_2, tr_2, rt_2 = simulation(ALGO, EPSILON_GREEDY, 0.01, 1, size_maze, SEED) # ε = 0.01 (label=2)
    n_episode_3, sr_3, tr_3, rt_3 = simulation(ALGO, EPSILON_GREEDY, 0.1, 1, size_maze, SEED) # ε = 0.1 (label=3) 
    n_episode_4, sr_4, tr_4, rt_4 = simulation(ALGO, EPSILON_GREEDY, 0.5, 0.9, size_maze, SEED) # ε = 0.5 & ε_decay=0.9 (label=4)
    n_episode_5, sr_5, tr_5, rt_5 = simulation(ALGO, EPSILON_GREEDY, 0.5, 0.99, size_maze, SEED) # ε = 0.5 & ε_decay=0.99 (label=5)
    n_episode_6, sr_6, tr_6, rt_6 = simulation(ALGO, EPSILON_GREEDY, 0.5, 0.999, size_maze, SEED) # ε = 0.5 & ε_decay=0.999 (label=6)
    
    
    # Plot results
    # ... for SUCCESS RATE:
    plt.figure(0)
    plt.plot(range(0,n_episode_1,10),sr_1[0::10], label=label1)
    plt.plot(range(0,n_episode_2,10),sr_2[0::10], label=label2)
    plt.plot(range(0,n_episode_3,10),sr_3[0::10], label=label3)
    plt.plot(range(0,n_episode_4,10),sr_4[0::10], label=label4)
    plt.plot(range(0,n_episode_5,10),sr_5[0::10], label=label5)
    plt.plot(range(0,n_episode_6,10),sr_6[0::10], label=label6)
    plt.title("Influence of epsilon on the success rate with {} training. Size={}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.savefig("Simulations/Influence_Epsilon/{0}/Figures/influence_of_EpsilonONsuccessRate_with_ALGO_{0}_size{1}.png".format(ALGO, size_maze))
    
    # ... for TOTAL REWARD:
    plt.figure(1)
    plt.plot(range(0,n_episode_1,10),tr_1[0::10], label=label1)
    plt.plot(range(0,n_episode_2,10),tr_2[0::10], label=label2)
    plt.plot(range(0,n_episode_3,10),tr_3[0::10], label=label3)
    plt.plot(range(0,n_episode_4,10),tr_4[0::10], label=label4)
    plt.plot(range(0,n_episode_5,10),tr_5[0::10], label=label5)
    plt.plot(range(0,n_episode_6,10),tr_6[0::10], label=label6)
    plt.title("Influence of epsilon on the total reward with {} training. Size={}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("Simulations/Influence_Epsilon/{0}/Figures/influence_of_EpsilonONtotalReward_with_ALGO_{0}_size{1}.png".format(ALGO, size_maze))
    
    # ... for RESOLUTION TIME
    plt.figure(2)
    plt.plot(range(0, n_episode_1, 10),rt_1[0::10], label=label1)
    plt.plot(range(0, n_episode_2, 10),rt_2[0::10], label=label2)
    plt.plot(range(0, n_episode_3, 10),rt_3[0::10], label=label3)
    plt.plot(range(0, n_episode_4, 10),rt_4[0::10], label=label4)
    plt.plot(range(0, n_episode_5, 10),rt_5[0::10], label=label5)
    plt.plot(range(0, n_episode_6, 10),rt_6[0::10], label=label6)
    plt.title("Influence of epsilon on the resolution time with {} training. Size={}".format(ALGO, size_maze))
    plt.xlabel("Episodes")
    plt.ylabel("Resolution Time")
    plt.legend()
    plt.savefig("Simulations/Influence_Epsilon/{0}/Figures/influence_of_EpsilonONresolutionTime_with_ALGO_{0}_size{1}.png".format(ALGO, size_maze))
    plt.show()
    
    
    if ((ALGO == "SARSA") or (ALGO == "Q_learning")):
        sr1 = asarray(sr_1)
        save('Simulations/Influence_Epsilon/{0}/Arrays/SuccessRate/sr1-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), sr1)
        sr2 = asarray(sr_2)
        save('Simulations/Influence_Epsilon/{0}/Arrays/SuccessRate/sr2-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), sr2)
        sr3 = asarray(sr_3)
        save('Simulations/Influence_Epsilon/{0}/Arrays/SuccessRate/sr3-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), sr3)
        sr4 = asarray(sr_4)
        save('Simulations/Influence_Epsilon/{0}/Arrays/SuccessRate/sr4-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), sr4)
        sr5 = asarray(sr_5)
        save('Simulations/Influence_Epsilon/{0}/Arrays/SuccessRate/sr5-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), sr5)
        sr6 = asarray(sr_6)
        save('Simulations/Influence_Epsilon/{0}/Arrays/SuccessRate/sr6-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), sr6)
        
        tr1 = asarray(tr_1)
        save('Simulations/Influence_Epsilon/{0}/Arrays/TotalReward/tr1-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), tr1)
        tr2 = asarray(tr_2)
        save('Simulations/Influence_Epsilon/{0}/Arrays/TotalReward/tr2-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), tr2)
        tr3 = asarray(tr_3)
        save('Simulations/Influence_Epsilon/{0}/Arrays/TotalReward/tr3-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), tr3)
        tr4 = asarray(tr_4)
        save('Simulations/Influence_Epsilon/{0}/Arrays/TotalReward/tr4-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), tr4)
        tr5 = asarray(tr_5)
        save('Simulations/Influence_Epsilon/{0}/Arrays/TotalReward/tr5-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), tr5)
        tr6 = asarray(tr_6)
        save('Simulations/Influence_Epsilon/{0}/Arrays/TotalReward/tr6-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), tr6)
        
        rt1 = asarray(rt_1)
        save('Simulations/Influence_Epsilon/{0}/Arrays/ResolutionTime/rt1-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), rt1)
        rt2 = asarray(rt_2)
        save('Simulations/Influence_Epsilon/{0}/Arrays/ResolutionTime/rt2-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), rt2)
        rt3 = asarray(rt_3)
        save('Simulations/Influence_Epsilon/{0}/Arrays/ResolutionTime/rt3-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), rt3)
        rt4 = asarray(rt_4)
        save('Simulations/Influence_Epsilon/{0}/Arrays/ResolutionTime/rt4-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), rt4)
        rt5 = asarray(rt_5)
        save('Simulations/Influence_Epsilon/{0}/Arrays/ResolutionTime/rt5-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), rt5)
        rt6 = asarray(rt_6)
        save('Simulations/Influence_Epsilon/{0}/Arrays/ResolutionTime/rt6-influence_eps_{0}_size{1}_seed{2}.npy'.format(ALGO, size_maze, SEED), rt6)