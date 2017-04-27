'''
Need to set paths correctly
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cPickle as pickle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import tensorflow as tf
import numpy as np
# from IPython.display import YouTubeVideo

import os.path
import glob, os

fileList = []
baseDir = "/proj/audioset_v1_embeddings/bal_train/"
dumpDir = "/proj/pickles_audioset/bal_train/"


for file in glob.glob(baseDir + "*.tfrecord"):
    fileList.append(file)

for frame_lvl_record in fileList:
    print frame_lvl_record

    fileName = frame_lvl_record.split("/")[-1]
    fileName = fileName.split(".")[0]
    fileDump = dumpDir + fileName + ".pkl"
    if os.path.isfile(fileDump):
        print ("File exists %s " % fileName)
        continue

    for example in tf.python_io.tf_record_iterator(frame_lvl_record):
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
        sess = tf.InteractiveSession()
        frames = []
        # iterate through frames
        for i in range(n_frames):
            frames.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0], tf.uint8)
                , tf.float32).eval())
        sess.close()

        break
    features = np.asarray(frames)

    with open(fileDump, "w") as f:
        pickle.dump(features, f)
