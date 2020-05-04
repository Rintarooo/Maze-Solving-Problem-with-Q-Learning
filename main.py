#Q-learning algorithm with an epsilon-greedy exploration strategy.
import numpy as np
import matplotlib.pyplot as plt
import argparse
from config import Config, load_pkl
from env import MazeEnv

def pkl_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--pkl_path', metavar = 'P', type = str, default = './maze_q_learning.pkl', help = 'pkl file name')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = pkl_parser() 
	cfg = load_pkl(args.pkl_path) 
		
	agent = MazeEnv(start_x = 1, start_y = 1, goal_x = 8, goal_y = 8)
	Q = np.random.rand(10, 10, 4)
	each_steps = []	
	for i in range(1, cfg.iter+1):
		agent.reset()
		steps = []
		action_id = agent.get_action_at_current_state(Q, cfg.eps)
		while True:
			steps.append(action_id)
			agent.get_next_state(action_id)
			agent.get_reward(cfg.wall, cfg.time, cfg.goal)
			
			if cfg.mode == 'q_learning':
				Q = agent.Q_learning(action_id, Q, cfg.eta, cfg.gamma)
			
			next_action_id = agent.get_action_at_next_state(Q, cfg.eps)
			cfg.eps = cfg.eps / 2#Explore->Exploit
			if cfg.mode == 'sarsa':
				Q = agent.SARSA(action_id, next_action_id, Q, cfg.eta, cfg.gamma)
			
			agent.go_next_state()
			action_id = next_action_id
			# ~ print(agent.y,agent.x)
			if agent.x == agent.goal_x and agent.y == agent.goal_y:
				print('episode{}: {} steps'.format(i, len(steps)))
				each_steps.append(len(steps))
				break
				
		if i == cfg.iter:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			ax.plot(range(cfg.iter), each_steps)
			ax.set_title('Maze Solver_' + cfg.mode)
			ax.set_xlabel('Episodes')
			ax.set_ylabel('Step Number')
			ax.grid(True)
			plt.show()
			plt.savefig('maze_result.png')
