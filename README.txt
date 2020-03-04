README.txt    -   Project INF581

1) Organisation of the repertory Project

Before excecuting the 2 .py files compareRL_Algos.py and influence_epsilon.py, make sure your working repertory (here called "Project" is organised as follows: 

/Project 
	|____ compareRL_Algos.py
	|	 
	|____ influence_epsilon.py
	|
	|
	|____/Simulations
	|	  |
	|	  |____/Influence_Epsilon
	|	  |		|
	|	  |		|__/SARSA
	|	  |		|
	|	  |		|__/Q-learning
	|	  |	  
	|	  |
	|	  |____ /CompareRL_Algos
	|
	|____ batch_simulation.py



* Eventually, these 3 repertories /SARSA, /Q-learning and /CompareRL_Algos have the same architectures: 

  |__/Figures
  |
  |__/Arrays 
        |__/SuccessRate
        |
        |__/TotalReward
        |
        |__/ResolutionTime


The subdirectory Figures enables to compare how well the algorithms behave with the size of the maze, or if a greedy algorithm (eps = 0) is interesting or not in some cases and so on...


The subdirectory Arrays will be used to handle measures with batch: indeed, we will consider a batch with for n generated mazes and compute the means on arrays obtained. We will explain precisely how computations will be realised.


2) The file compareRL_Algos.py

This file enables a clear comparison with the two main RL algorithms: SARSA (on-policy), and Q-Learning (off policy) plotting clear and fully detailed curves. 

This file must be launched from the terminal with the following command: 
** python compareRL_Algos.py size_maze exploration_method eps_decay seed **

Details will be given if the command is wrong.




3) The file influence_epsilon.py

This file aims at analysing the learning efficiency with different exploration methods as eps varies differently.
We designed 6 different modes:
* 3 modes where 
- eps = 0 through all episodes: this is the greediest approach possible.
- eps = 0,01
- eps = 0,1
* And 3 modes where eps decreases starting from eps = 0,5 to zero with a multiplication factor eps_decay < 1; the little eps_decay, the faster eps tends to zero and thhe faster the algorithm fosters exploitation method at the expense of (random) explorations.
- eps_decay = 0,9  
- eps_decay = 0,99
- eps_decay = 0,999


This file must be launched from the terminal with the following command: 
** python influence_epsilon.py size_maze ALGO eps_decay seed **

Details will be given if the command is wrong.


3) Figures and Arrays saved

Each execution will have figures saved in adequate repertories to enable analysis. 

Arrays giving resolution times, success rates or total rewards (for each episode) will be respectively saved in these three folders after the execution: ResolutionTime, SuccessRate and TotalReward. 
The idea will be to generate a batch of n random mazes and save them in these three directories. Then we will compute, for each of the three directories, an average array of the n arrays stored in the folder and plot these three arrays to propose a better analysis of our learning algorithms. 
This will all be managed by the file batch_simulation.py.




