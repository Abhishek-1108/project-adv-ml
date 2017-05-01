'''
Reads and returns a pickle object
'''
import cPickle as pickle
import sys

def read_pickle(filename):
	data = ""
	with open(filename,"rb") as f:
		data = pickle.load(f)
	return data

if __name__ == '__main__':
	read_pickle(sys.argv[1])
