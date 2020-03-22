README.txt    -    Project INF581

1) Organisation of the repertory Project

Before executing the file batch_simulation.py, make sure your working repertory, here called "Project", is organised as follows so that it assures the loading of necessary functions in simulation.py and so that figures can be stored in appropriate folders tase analysed afterwards. 


/Project 
	|___ simulation.py
	|	 
	|___ batch_simulation.py
	|
	|____/Figures
		  |
		  |____/CompareRL_Algos  
		  |
		  |____/CompareExplorationMethods
		  |
		  |____/Influence_Epsilon
		  		|
		  		|__/SARSA
		  		|
		  		|__/Q_learning
	
  

Figures obtained will be used to compare how well the algorithms behave with the size of the maze, or if a greedy algorithm (eps = 0) is interesting or not in some cases and so on...


2) The file simulation.py

Before going further into the details, let us define the main function of the project - simulation, located in the .py file having the same name - and the three key parameters used to assess the performance of the learning. 
The main argument of this function are the reinforcement learning (RL) algorithm (SARSA or Q-Learning) and the exploration method (softmax or epsilon-greedy).

It returns a tuple of four components, necessary to describe how well the agent is learning through the episode.
* The first one gives the number of episodes (nb_of_episodes) needed to "solve" the maze, i.e. solve the maze in less than SOLVED_T steps per episode and for NUM_STREAKS consecutive episodes. 
Note: if an episode reaches MAX_T steps, it will necessary time out. 
* The second one is an array of size nb_of_episodes defining the Success Rate for a given episode, that is the proportion of episodes quickly solved so far, i.e. requiring less than SOLVED_T steps. 
* The third one is also an array of size nb_of_episodes. It measures the Total Reward obtained per episode to reach the goal cell. 
* Eventually, the fourth component is an array of the same size giving the Resolution Time that is the the number of steps for each episode. 

This file will be imported while executing the file batch_simulation.py in order to plot figures. 



3) The file batch_simulation.py
This file is used to obtain average curves by executing the function simulation on different randomly generated mazes in order to give more credits to the figures rendered.
It enables to perform three different studies with, for each case, three figures representing Success Rate, Total Reward and Resolution Time in function of the episodes. These results will be carefully described and discussed in the report.
	
	a) To compare the Reinforcement algorithms SARSA and Q-Learning
This file enables a clear comparison with the two main RL algorithms: SARSA (on-policy), and Q-Learning (off policy) plotting clear and fully detailed curves. One can choose additional parameters such as eps_start and eps_decay to impose how the exploration/exploitation parameter varies.  

	b) To analyse the trade-off between exploration and exploitation
The figures plotted tends to give an idea about the most performant epsilon decreasing to exploit the exploration/exploitation trade-off.
To do so we used six different modes having different variations of the hyperparameter eps.
* 3 modes where epsilon is constant through the episodes:
- eps = 0 -> this is the greediest approach possible.
- eps = 0,01
- eps = 0,1
* And 3 modes where eps decreases linearly starting from eps = 0,5 to zero with a multiplication factor eps_decay < 1; the little eps_decay, the faster eps tends to zero and, thus, the faster the algorithm advocates the exploitation method at the expense of (random) explorations.
- eps_decay = 0,9
- eps_decay = 0,99
- eps_decay = 0,999

 	c) To compare the exploration methods: EPSILON_GREEDY vs SOFTMAX
The figures plotted in this case tend to enhance the discrepancy between two different methods used to select actions: one using epsilon greedy selection and the other softmax.

