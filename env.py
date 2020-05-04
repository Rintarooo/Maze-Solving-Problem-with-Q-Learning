import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

class MazeEnv():
	def __init__(self, start_x = 1, start_y = 1, goal_x = 8, goal_y = 8):
		
		self.maze=np.array([
							[0,0,0,0,0,0,0,0,0,0],
							[0,1,0,1,1,1,1,1,1,0],
							[0,1,1,1,0,1,1,0,1,0],
							[0,0,0,1,0,1,0,0,1,0],
							[0,1,1,1,1,1,1,1,1,0],
							[0,1,0,1,0,0,0,0,0,0],
							[0,1,0,1,0,1,1,1,1,0],
							[0,1,0,0,0,1,0,0,1,0],
							[0,1,1,1,1,1,1,0,1,0],
							[0,0,0,0,0,0,0,0,0,0]])
		
		self.start_y, self.start_x = start_y, start_x
		self.goal_y, self.goal_x = goal_y, goal_x
		
	def reset(self):
		self.y, self.x = self.start_y, self.start_x
		
	def get_action_at_current_state(self, Q, epsilon):
		#epsilon greedy
		if np.random.rand() < epsilon:#Explore: select a random action
			action_id = np.random.randint(4)
		else:#Exploit: select the action with max value
			action_id = np.argmax(Q[self.y][self.x], axis=0)
		return action_id
		
	def get_action_at_next_state(self, Q, epsilon):
		#epsilon greedy
		if np.random.rand() < epsilon:#Explore: select a random action
			action_id = np.random.randint(4)
		else:#Exploit: select the action with max value
			action_id = np.argmax(Q[self.next_y][self.next_x], axis=0)
		return action_id

	def get_next_state(self, action_id):
		direction_option = ['up','right','down','left']
		direction = direction_option[action_id]
		
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
		
	def get_reward(self, crash_wall_penalty, time_penalty, goal_reward):
		self.reward = 0
		if self.maze[self.next_y][self.next_x] == 0:
			self.reward -= crash_wall_penalty
		elif self.next_y == self.goal_y and self.next_x == self.goal_x:
			self.reward += goal_reward
		self.reward -= time_penalty  

	def Q_learning(self, action_id, Q, eta, gamma):
		#Q[s,a] = Q[s,a] + eta*(r + gamma*np.nanmax(Q[s_next,:]) - Q[s,a])
		#eta is learning rate, gammma is discount factor
		if self.next_y == self.goal_y and self.next_x == self.goal_x:
			Q[self.y][self.x][action_id] = Q[self.y][self.x][action_id] + eta*(self.reward-Q[self.y][self.x][action_id])
		else:
			Q[self.y][self.x][action_id] = Q[self.y][self.x][action_id] + eta*(self.reward+gamma*np.max(Q[self.next_y][self.next_x], axis=0) - Q[self.y][self.x][action_id])
		return Q
	
	def SARSA(self, action_id, next_action_id, Q, eta, gamma):
		#Q[s,a] = Q[s,a] + eta*(r + gamma*(Q[s_next,a_next]) - Q[s,a])
		#eta is learning rate, gammma is discount factor
		if self.next_y == self.goal_y and self.next_x == self.goal_x:
			Q[self.y][self.x][action_id] = Q[self.y][self.x][action_id] + eta*(self.reward-Q[self.y][self.x][action_id])
		else:
			Q[self.y][self.x][action_id] = Q[self.y][self.x][action_id] + eta*(self.reward+gamma*Q[self.next_y][self.next_x][next_action_id] - Q[self.y][self.x][action_id])
		return Q
		
	def go_next_state(self):
		self.y, self.x = self.next_y, self.next_x

if __name__ == '__main__':
	agent = MazeEnv()
	plt.figure()
	plt.text(1,1.5,'START', color = 'b')
	plt.text(8,8.5,'GOAL', color = 'b')
	sb.heatmap(agent.maze, cbar = False, cmap = 'hot')
	plt.show()
