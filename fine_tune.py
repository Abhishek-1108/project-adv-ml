import numpy as np
import datetime
import os
from collections import Counter

import keras
from sklearn.naive_bayes import MultinomialNB
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.callbacks import ModelCheckpoint
import sklearn.preprocessing

from utils import generate_data_for_labels, read_pickle

MOST_COMMON = 4

def build_model(input_shape, n_classes):
    model = Sequential()
    model.add(
        Dense(128, input_shape=input_shape)
    )
    model.add(
        Dense(n_classes, activation='softmax')
    )
    return model


def get_callbacks(prefix, base_dir):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = base_dir + 'logs/' # .format(prefix, timestamp)
    print('log_dir {}'.format(log_dir))
    model_checkpoint_dir = base_dir + 'models/' # .format(prefix, timestamp)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    # # build a tensorboard callback object using the above log dir
    # tensorboard_callback = TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=0,
    #     write_graph=True,
    #     write_images=False
    # )
    # # build a model checkpoint callback object using the above model checkpoint dir
    checkpoint_callback = ModelCheckpoint(
        model_checkpoint_dir + 'weights.{epoch:02d}.hdf5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    return [checkpoint_callback]


def get_pd(x, y):
    clf = MultinomialNB()
    clf.fit(x, y)
    return clf.predict_proba(x)


def fine_tune_model(model, videos, audios, labels, base_dir):
    model.layers[-1].trainable = False

    print('Layers after freezing')
    for layer in model.layers:
        print(layer)

    model.compile(
        optimizer=Adam(),
        loss=losses.kld
    )
    pd = get_pd(audios, labels)

    model.fit(
        videos,
        pd,
        callbacks=get_callbacks('second_stage', base_dir)
    )


def main():
    base_dir = '/Users/mridul/Downloads/finetune/'
    videos_file = base_dir + 'video_new.pkl'
    video_labels_file = base_dir + 'labels_dca.pkl'
    video_labels = read_pickle(video_labels_file).flatten()
    unique_labels = np.array(list(set(video_labels)))
    print "Original Unique Labels", len(unique_labels)
    c = Counter(video_labels)

    chosen_labels = []
    for word, count in c.most_common(MOST_COMMON):
        chosen_labels.append(word)
    chosen_labels = np.array(chosen_labels)

    videos, labels = generate_data_for_labels(videos_file, video_labels, chosen_labels)

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(len(chosen_labels)))
    labels = label_binarizer.transform(labels)

    input_shape = (videos.shape[1],)
    model = build_model(input_shape, MOST_COMMON)

    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer=Adam()
    )
    model.fit(
        videos, labels,
        epochs=50,
        verbose=1,
        validation_split=0.15,
        callbacks=get_callbacks('first_stage', base_dir)
    )

    fine_tune_model(model, videos, audios, labels, base_dir)

    # model fit -> W1, W2
    # freeze W2
    # model.compile(loss=kld, ...)
    # model.fit(videos, prob, ...)




if __name__ == '__main__':
    main()