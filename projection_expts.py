from models import *
import cPickle as pickle
import random

from sklearn.model_selection import train_test_split
import numpy as np


def main():
    models = [
        'videos_projected_128', 'concat_projected_128',]  # 'concat_projected_256', 'concat_projected_64',
    # ]
    basedir = '/Users/mridul/Downloads/projections/'

    for m in models:
        xfullpath = basedir + m + ".pkl"
        yfullpath = basedir + m + "_labels.pkl"
        with open(xfullpath) as xfile:
            x = pickle.load(xfile)
        with open(yfullpath) as yfile:
            y = pickle.load(yfile)

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1, random_state=42)

        xgb_classify(X_train, y_train, X_test, y_test)



if __name__ == '__main__':
    main()

