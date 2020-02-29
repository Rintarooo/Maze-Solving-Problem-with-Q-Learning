#Q-learning algorithm with an epsilon-greedy exploration strategy.
import numpy as np
import matplotlib.pyplot as plt
import argparse
from env import MazeEnv

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iter', metavar = 'I', type = int, default = 150, help = 'iteration')
	parser.add_argument('-e', '--eps', metavar = 'E', type = float, default = 0.5, help = 'epsilon greedy strategy')
	parser.add_argument('-w', '--wall', metavar = 'W', type = int, default = 120, help = 'penalty')
	parser.add_argument('-t', '--time', metavar = 'T', type = int, default = 1, help = 'penalty')
	parser.add_argument('-g', '--goal', metavar = 'G', type = int, default = 100, help = 'reward')
	parser.add_argument('--eta', metavar = 'ETA', type = float, default = 0.1, help = 'learning rate for updateing Q value')
	parser.add_argument('--gamma', metavar = 'GAM', type = float, default = 0.9, help = 'discount factor for updateing Q value')
	args = parser.parse_args()
	
	iteration = args.iter
	epsilon = args.eps
	crash_wall_penalty = args.wall
	time_penalty = args.time
	goal_reward = args.goal
	eta = args.eta
	gamma = args.gamma	
	
	Q = np.random.rand(10,10,4)
	z = []	
	for i in range(1, iteration+1):
		agent = MazeEnv(start_x = 1, start_y = 1, goal_x = 8, goal_y = 8)
		steps = []
		while True:
			action_id = agent.get_action(Q, epsilon)
			steps.append(action_id)
			epsilon = epsilon / 2
			reward = agent.get_next_state(action_id, crash_wall_penalty, time_penalty, goal_reward)
			Q = agent.Q_learning(action_id, reward, Q, eta, gamma)
			agent.update_state()
			
			if agent.x == agent.goal_x and agent.y == agent.goal_y:
				print('episode{}: {} steps'.format(i, len(steps)))
				z.append(len(steps))
				
				if i == iteration:
					fig=plt.figure()
					ax=fig.add_subplot(111)
					ax.plot(range(iteration), z)
					ax.set_title('Maze Solver Q-learning')
					ax.set_xlabel('Episodes')
					ax.set_ylabel('Step Number')
					ax.grid(True)
					plt.show()
					plt.savefig('maze_result.png')
				break
