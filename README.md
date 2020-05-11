# Maze Solving Problem Value-based RL(Reinfocement Learning)

![Figure_1](https://user-images.githubusercontent.com/51239551/80976415-658c8780-8e5e-11ea-858b-9518be91b921.png)

## 10×10 Maze Solver using Reinforcement Learning
This is implementation of a single maze problem.
In this work, we will find the shortest path for the agent to reach the goal. 
Though the agent can move in four directions(up, down, left, right), the agent can't get out of 10×10 maze grid at the edge or corner.
I implemented 2 ways of value based RL(Reinforcemnt Learning) algorithms, Q-Learning and SARSA.
Q-Leaning is off-policy whereas SARSA is on-policy type, which means SARSA use real action at the next state to calculate Q value but calculation in Q-Learning is not related to what the next action is like.
The agent follows epsilon-greedy, chooses the next action randomly in the beginning but gets to move around with high Q value over the time.


## Usage
Run the following command

```python config.py -h```

You will see the pickle file containing hyperparamer, then run this

```python main.py -h```

```main.py``` generates result figure how the path is optimized

![Figure_1](https://user-images.githubusercontent.com/51239551/80980702-ebf79800-8e63-11ea-892e-1e2b81fd346c.png)
![Figure_2](https://user-images.githubusercontent.com/51239551/80980709-eef28880-8e63-11ea-8b62-f51391456120.png)
