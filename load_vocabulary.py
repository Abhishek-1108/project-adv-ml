'''
Returns a mapping from label index to the name 
Reads from vocabulary.csv file 
'''
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def load_vocabulary(filename):
	flag=0
	labels_id2name = {}
	with open(filename, "r") as f:
		
		for line in f:
			if flag==0:
				flag=1
				continue
			csv_line = line.strip().split(",")
			idx = int(csv_line[0])
			name = csv_line[3]
			labels_id2name[idx] = name

	return labels_id2name



if __name__ == '__main__':
	load_vocabulary(sys.argv[1])