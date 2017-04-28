import tsne
# import sys
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from read_pickle import read_pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from cvxpy import *
# from utils import 
MOST_COMMON = 4
VISULIZATION_SAMPLES = 500
VOCAB_FILE = "../models/vocabulary.csv"
import math

import ipdb
# import keras
# from keras.models import Sequential
# from keras.layers.core import Dense, Lambda
# from keras.losses import kullback_leibler_divergence
from utils import generate_data_for_labels, plot_tsne_data, load_vocabulary
from cvxpy import *
# import tensorflow as tf


# def mean(numbers):
#     return sum(numbers)/float(len(numbers))

# def stdev(numbers):
#     avg = mean(numbers)
#     variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
#     return math.sqrt(variance)

# def separateByClass(dataset, labels):
#     separated = {}
#     for vector, label in zip(dataset,labels):
        
#         if (label not in separated):
#             separated[label] = []
#         separated[label].append(vector)
#     return separated
# def summarize(dataset):
#     summaries = [(mean(attribute), stdev(attribute)) for attribute in (dataset)]
#     return summaries

# def summarizeByClass(dataset, labels):
#     separated = separateByClass(dataset,labels)
#     summaries = {}
#     for classValue, instances in separated.iteritems():
#         summaries[classValue] = summarize(instances)
#     return summaries

# probabilities = calculateClassProbabilities(summaries, inputVector)

# def calculateProbability(x, mean, stdev):
#     exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
#     return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# def calculateClassProbabilities(summaries, inputVector):
#     probabilities = {}
#     for classValue, classSummaries in summaries.iteritems():
#         print classValue
#         probabilities[classValue] = 1
#         for i in range(len(classSummaries)):
#             mean, stdev = classSummaries[i]
#             x = inputVector[i]
#             probabilities[classValue] *= calculateProbability(x, mean, stdev)
#     return probabilities


def calculate_probability(projected):
        # ipdb.set_trace()
        
        clf = GaussianNB()
        print '----', type(projected)
        print '----', projected.shape
        # ipdb.set_trace()
        clf.fit(projected.eval(), XA_labels)
        print '-----', clf.predict_proba(projected).shape
        return clf.predict_proba(projected)


def build_model(input_shape):

    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    
    model.add(Lambda(lambda x: calculate_probability(x)))

    return model



if __name__ == '__main__':
    
    YA = "../models/matlab/dca/video_new.npy"
    XA = "../models/matlab/dca/audio_new.npy"
    labels = read_pickle("../models/matlab/dca/labels.npy").flatten()

    unique_labels = np.array(list(set(labels)))
    print "Original Unique Labels", len(unique_labels)
    c = Counter(labels)

    chosen_labels= []
    for word, count in c.most_common(MOST_COMMON):
        chosen_labels.append(word)
    chosen_labels= np.array(chosen_labels)

    YA, YA_labels = generate_data_for_labels(YA, labels, chosen_labels, Transpose=0)
    XA, XA_labels = generate_data_for_labels(XA, labels, chosen_labels, Transpose=0)
    # labels_id2name = load_vocabulary(VOCAB_FILE)
    print YA.shape
    print XA.shape

    XA = XA[:100]
    YA = YA[:100]

    XA_labels = XA_labels[:100]
    print "Using 1000 samples only"
    # ipdb.set_trace()
    row, col = XA.shape
    w1 = Variable(col, 10)
    m = 1000
    row, col = YA.shape 
    w2 = Variable(col,10)

    loss =  norm(XA*w1 - YA*w2, "fro")


    reg = norm(w1, 1)+norm(w2,1)
    lambd = Parameter(sign="positive")
    prob = Problem(Minimize(loss/m + 0.5*reg), constraints=[(w1>0), (w2>0)])


    # print labels
    # clf = GaussianNB()
    # clf.fit(XA,XA_labels)
    # probability_XA = clf.predict_proba(XA)
    
    # concat = np.hstack((XA, YA))
    # batch_size, from_dim = concat.shape
    # print "Concatenation Done"
    
    # sess = tf.Session()

    # with sess.as_default():
    
    #     model = build_model((from_dim,))
    #     print "Building Model"

    #     model.compile(loss=kullback_leibler_divergence, optimizer="Adam")
    #     print "Model Built"
    #     ipdb.set_trace()

    #     model.fit(concat, probability_XA, batch_size = batch_size, shuffle=False)