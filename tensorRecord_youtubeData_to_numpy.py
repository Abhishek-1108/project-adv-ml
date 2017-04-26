'''
Need to set paths correctly
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
baseDir = "/proj/youtube-8M/frame_level/"
dumpDir = "/proj/youtube-8M/frame_level_dump/"

# baseDir = "/home/mudgal/data/yt8m/"
# dumpDir  = "/home/mudgal/data/dump/"
for file in glob.glob(baseDir+"*.tfrecord"):
    fileList.append(file)

for frame_lvl_record in fileList:
    print frame_lvl_record
    feat_rgb = []
    feat_audio = []
    fileName = frame_lvl_record.split("/")[-1]
    fileName = fileName.split(".")[0]
    fileDump = dumpDir+fileName
    if os.path.isfile(fileDump+"-mean-rgb.pkl") and os.path.isfile(fileDump+"-mean-audio.pkl"):
        print ("File exists %s "% fileName)
        continue



    for example in tf.python_io.tf_record_iterator(frame_lvl_record):        
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        sess = tf.InteractiveSession()
        rgb_frame = []
        audio_frame = []
        # iterate through frames
        for i in range(n_frames):
            rgb_frame.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval())
            audio_frame.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval())

        sess.close()
        feat_rgb.append(rgb_frame)
        feat_audio.append(audio_frame)
        
        break
    feat_rgb =np.array(np.mean(feat_rgb[:],axis=1)[0])
    feat_audio =  np.array(np.mean(feat_audio[:],axis=1)[0])
    
    
    with open(fileDump+"-mean-rgb.pkl","wb") as f:
        pickle.dump(feat_rgb, f)
    with open(fileDump+"-mean-audio.pkl","wb") as f:
        pickle.dump(feat_audio, f)
        
    
