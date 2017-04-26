'''
t-sne visualizer for cca_fused
Modify Filenames in main function for use
Assumes that you have the .npy files in the respect folders
'''
import tsne
import sys
from read_pickle import read_pickle
import numpy as np
from sklearn.manifold import TSNE
from load_vocabulary import load_vocabulary
from collections import Counter

MOST_COMMON = 4
VISULIZATION_SAMPLES = 500
VOCAB_FILE = "../models/vocabulary.csv"


def generate_data_for_labels(data_data_fileName, labels, chosen_labels):
    '''
    This generates the X,Y over the labels that we need 
    '''
    data = read_pickle(data_data_fileName)
    new_data = []
    new_labels = []
    for label, data_point in zip(labels, data):
        if label in chosen_labels:
            new_data.append(data_point)
            new_labels.append(label)
    new_data = np.array(new_data)
    new_labels = np.array(new_labels)

    return new_data, new_labels

def plot_tsne_data(data, labels, chosen_labels, labels_id2name,title,filename="tsne.png"):
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

def plot_data(data_fileName, labels, chosen_labels, labels_id2name,title="None", filename="tsne.png"):
    data, labels = generate_data_for_labels(data_fileName, labels, chosen_labels)
    plot_tsne_data(data, labels, chosen_labels,labels_id2name, title=title, filename=filename)

if __name__ == '__main__':


    fused_labels = read_pickle("../models/matlab/cca_fuse/labels_train.npy").flatten()
    video_labels = read_pickle("../models/matlab/cca_fuse/labels_train.npy").flatten()
    audio_labels = read_pickle("../models/matlab/cca_fuse/labels_train.npy").flatten()

    unique_labels = np.array(list(set(fused_labels)))
    print "Original Unique Labels", len(unique_labels)
    c = Counter(fused_labels)
    
    chosen_labels= []
    for word, count in c.most_common(MOST_COMMON):
        chosen_labels.append(word)
    chosen_labels= np.array(chosen_labels)


    labels_id2name = load_vocabulary(VOCAB_FILE)


    fused_data_fileName = "../models/matlab/cca_fuse/fused_train.npy"
    video_data_fileName = "../models/matlab/cca_fuse/video_train.npy"
    audio_data_fileName = "../models/matlab/cca_fuse/audio_train.npy"    

    print "Plotting Fused Data"
    plot_data(fused_data_fileName, fused_labels, chosen_labels, labels_id2name, "Fused", "cca_fused.png")

    print "Waiting for input"
    raw_input()
    plot_data(video_data_fileName,video_labels,chosen_labels,labels_id2name,"Video", "cca_video.png")

    print "Waiting for input"
    raw_input()
    plot_data(audio_data_fileName,audio_labels, chosen_labels, labels_id2name, "Audio", "cca_audio.png")
    



