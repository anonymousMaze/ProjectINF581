README.txt    -   Project INF581

1) Organisation of the repertory Project

Before excecuting the three .py files - compareRL_Algos.py, influence_greedy.py and simulation.py -, make sure your working repertory, called "Project" here, is organised as follows: 

/Project 
	|___ compareRL_Algos.py
	|	 
	|___ influence_greedy.py
	|
	|___ simulation.py
	|
	|___ btch_simulation.py
	|
	|____/Simulations
		  |
		  |____/CompareRL_Algos  
		  |
		  |
		  |____/Influence_Epsilon
		  		|
		  		|__/SARSA
		  		|
		  		|__/Q_learning
	
	


* Eventually, these 3 repertories /SARSA, /Q-learning and /CompareRL_Algos have the same architecture: 

  |__/Figures
  |
  |__/Arrays 
        |__/SuccessRate
        |
        |__/TotalReward
        |
        |__/ResolutionTime


The folder Figures enables to collect useful figures to compare how well the algorithms behave with the size of the maze, or if a greedy algorithm (eps = 0) is interesting or not in some cases and so on...

The folder Arrays have been designed to store arrays in order to collect measures over a given batch. It has not been used for the moment even though it may provide relevant analysis of the learning evolution. 

2) The file simulation.py

Before going further into the details, let us define main function of the project - simulation, located in the .py file having the same name - and the three key parameters used to assess the performance of the learning. 
The function takes as parameters the reinforcement learning (RL) algorithm used, the exploration method (softmax or epsilon-greedy)
It returns a tuple of four components, necessary to describe how well the agent is learning through the episode.
* The first one gives the number of episodes nb_of_episode needed to "solve" the maze, i.e. solve the maze in less than SOLVED_T steps per episode and for NUM_STREAKS consecutive episodes. 
Note: if an episode reaches MAX_T steps, it will necessary time out. 
* The second one is an array of size nb_of_episodes defining the Success Rate for a given episode, that is the proportion of episodes quickly solved so far, i.e. requiring less than SOLVED_T steps. 
* The third one is also an array of size nb_of_episodes. It measures the Total Reward obtained per episode to reach the goal cell. 
* Eventually, the fourth component is an array of the same size giving the Resolution Time that is the the number of steps for each episode. 

This file will be imported in the files compareRL_Algos.py and influence_greedy.py in order to plot figures. 


2) The file compareRL_Algos.py

This file enables a clear comparison with the two main RL algorithms: SARSA (on-policy), and Q-Learning (off policy) plotting clear and fully detailed curves. 

This file must be launched from the terminal with the following command: 
** python compareRL_Algos.py size_maze exploration_method eps_decay seed **

Note: Details will be given if the command is wrong.




3) The file influence_greedy.py

This file aims at analysing the learning efficiency with different exploration methods as eps varies differently.
We designed 6 different modes:
* 3 modes where 
- eps = 0 through all episodes: this is the greediest approach possible.
- eps = 0,01
- eps = 0,1
* And 3 modes where eps decreases starting from eps = 0,5 to zero with a multiplication factor eps_decay < 1; the little eps_decay, the faster eps tends to zero, the faster the algorithm advocates the exploitation method at the expense of (random) explorations.
- eps_decay = 0,9  
- eps_decay = 0,99
- eps_decay = 0,999


This file must be launched from the terminal with the following command: 
** python influence_greedy.py size_maze ALGO eps_decay seed **

Note: Details will be given if the command is wrong.


3) Figures and Arrays saved

Each execution of compareRL_Algos.py and influence_greedy.py will have figures saved in adequate repertories to enable analysis. 

Arrays giving resolution times, success rates or total rewards (for each episode) will be saved in the adequate subdirectories in the folder Arrays. 



