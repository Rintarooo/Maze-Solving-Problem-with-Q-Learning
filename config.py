import pickle
import os
import argparse

def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iter', metavar = 'I', type = int, default = 150, help = 'iteration')
	parser.add_argument('-e', '--eps', metavar = 'E', type = float, default = 0.5, help = 'epsilon greedy strategy')
	parser.add_argument('-w', '--wall', metavar = 'W', type = int, default = 200, help = 'crash wall penalty')
	parser.add_argument('-t', '--time', metavar = 'T', type = int, default = 1, help = 'time penalty')
	parser.add_argument('-g', '--goal', metavar = 'G', type = int, default = 100, help = 'reward')
	parser.add_argument('--eta', metavar = 'ETA', type = float, default = 0.1, help = 'learning rate for updateing Q value')
	parser.add_argument('--gamma', metavar = 'GAM', type = float, default = 0.9, help = 'discount factor for updateing Q value')
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, default = 'q_learning', choices = ['q_learning', 'sarsa'], help = 'file name')
	args = parser.parse_args()
	return args

class Config():
	def __init__(self,**kwargs):	
		for k,v in kwargs.items():
			self.__dict__[k] = v
		self.cfg_path = './%s.pkl'%(self.mode)	

def dump_pkl(args):
	cfg = Config(**vars(args))
	with open(cfg.cfg_path, 'wb') as f:
		pickle.dump(cfg, f)
		print('pickle dump!\n')
	
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		if verbose:
			kwargs = vars(cfg)
			print(''.join('%s: %s\n'%item for item in kwargs.items()))
	return cfg
			
if __name__ == '__main__':
	args = argparser()
	dump_pkl(args)
	pkl_path1 = './q_learning.pkl'
	pkl_path2 = './sarsa.pkl' 
	cfg = load_pkl(pkl_path1)
	cfg = load_pkl(pkl_path2)
