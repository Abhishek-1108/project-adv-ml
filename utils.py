import tsne
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from read_pickle import read_pickle
import numpy as np
from sklearn.manifold import TSNE
from load_vocabulary import load_vocabulary
from collections import Counter
# from utils import 
MOST_COMMON = 4
VISULIZATION_SAMPLES = 500
VOCAB_FILE = "../models/vocabulary.csv"


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


def generate_data_for_labels(data_data_fileName, labels, chosen_labels, Transpose=0):
    '''
    This generates the X,Y over the labels that we need 
    '''
    data = read_pickle(data_data_fileName)
    # fix for DCA transposed Vectors
    if not Transpose:
        data = data.T
    
    new_data = []
    new_labels = []
    for label, data_point in zip(labels, data):
        if label in chosen_labels:
            new_data.append(data_point)
            new_labels.append(label)
    new_data = np.array(new_data)
    new_labels = np.array(new_labels)
    return new_data, new_labels

def plot_tsne_data(data, labels, labels_id2name,title,filename="tsne.png"):
    '''
    Creates a sample of data that we require for TSNE
    '''
    random_idx = range(len(data))
    np.random.shuffle(random_idx)
    random_idx = random_idx[:VISULIZATION_SAMPLES]
  
    data = data[random_idx]
    
    labels = labels[random_idx]
    
    labels = [labels_id2name[label] for label in labels]

    init_embed_size = len(data[0])

    print "Initial dimension", init_embed_size
    print "Visualizing random shuffle of "+str(VISULIZATION_SAMPLES)+" entries"
    data = np.array(data)

    model = TSNE(n_components=2, random_state=0)
    
    Y = model.fit_transform(data)
    print "fit done" 
    print "fused_data Shape",data.shape
    print "Labels Shape", len(labels)

    # Y = tsne.tsne(fused_data,2, init_embed_size, 20 )
    print "Plot Shape", Y.shape

    print labels
    # raw_input()
    print Y.shape
    # raw_input()
    tsne.plot_with_labels(Y, labels, title=title, filename=filename)
