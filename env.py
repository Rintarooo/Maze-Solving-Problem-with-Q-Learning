import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MazeEnv():
	def __init__(self, start_x = 1, start_y = 1, goal_x = 8, goal_y = 8):
		
		self.maze=np.array([[0,0,0,0,0,0,0,0,0,0],
							[0,1,0,1,1,1,1,1,1,0],
							[0,1,1,1,0,1,1,0,1,0],
							[0,0,0,1,0,1,0,0,1,0],
							[0,1,1,1,1,1,1,1,1,0],
							[0,1,0,1,0,0,0,0,0,0],
							[0,1,0,1,0,1,1,1,1,0],
							[0,1,0,0,0,1,0,0,1,0],
							[0,1,1,1,1,1,1,0,1,0],
							[0,0,0,0,0,0,0,0,0,0],])
		
		self.y, self.x = start_y, start_x
		self.goal_y, self.goal_x = goal_y, goal_x
		
	def get_action(self, Q, epsilon):
		#epsilon greedy
		if np.random.rand() < epsilon:#Explore: select a random action
			action_id = np.random.randint(4)#random number(0~3)
		else:#Exploit: select the action with max value
			action_id = np.argmax(Q[self.y][self.x], axis=0)#take index(0~3) that makes Q value maximum
		return action_id

	def get_next_state(self, action_id, crash_wall_penalty, time_penalty, goal_reward):
		direction_option = ['up','right','down','left']
		direction = direction_option[action_id]
		reward = 0
		
		if direction == 'up':
			if self.y == 0:#if agent cannot go up anymore
				self.next_y, self.next_x = self.y, self.x#stays at the same position			
			else:
				self.next_y, self.next_x = self.y-1, self.x
	
		elif direction == 'right':
			if self.x == 9:
				self.next_y, self.next_x = self.y, self.x
			else:
				self.next_y, self.next_x = self.y, self.x+1
				
		elif direction == 'down':
			if self.y == 9:
				self.next_y, self.next_x = self.y, self.x
			else:
				self.next_y, self.next_x = self.y+1, self.x
				
		else:#direction == 'left' 
			if self.x == 0:
				self.next_y, self.next_x = self.y, self.x
			else:
				self.next_y, self.next_x = self.y, self.x-1
			
		if self.maze[self.next_y][self.next_x] == 0:
			reward -= crash_wall_penalty
			
		if self.next_y == self.goal_y and self.next_x == self.goal_x:
			reward += goal_reward
			
		reward -= time_penalty
		return reward    

	def Q_learning(self, action_id, reward, Q, eta, gamma):
		#Q[s,a] = Q[s,a] + eta*(r + gamma*np.nanmax(Q[s_next,:]) - Q[s,a])
		#r is reward, eta is learning rate, gammma is discount factor
		Q[self.y][self.x][action_id] = Q[self.y][self.x][action_id]\
										+ eta*(reward+gamma*np.max(Q[self.next_y][self.next_x], axis=0)\
										- Q[self.y][self.x][action_id])
		return Q
		
	def update_state(self):
		self.y, self.x = self.next_y, self.next_x

if __name__ == '__main__':
	agent = MazeEnv()
	print(pd.DataFrame(agent.maze))
