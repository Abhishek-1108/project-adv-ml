# This file contains different models used to train various datasets
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def xgb_classify(train,labels_train,test,labels_test):
    model = xgb.XGBClassifier(max_depth=10,n_estimators=100)
    model.fit(train, labels_train)
    predictions = model.predict(test)
    accuracy = accuracy_score(labels_test, predictions)
    cf = confusion_matrix(labels_test,predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print cf
    return [confusion_matrix,accuracy]


def kernelSVM_classify(train,labels_train,test,labels_test):
    model = svm.SVC(kernel='sigmoid')
    model.fit(train, labels_train)
    predictions = model.predict(test)
    accuracy = accuracy_score(labels_test, predictions)
    cf = confusion_matrix(labels_test,predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print cf
    return [confusion_matrix,accuracy]

def kNN(train,labels_train,test,labels_test):
    model = KNeighborsClassifier()
    model.fit(train, labels_train)
    predictions = model.predict(test)
    accuracy = accuracy_score(labels_test, predictions)
    cf = confusion_matrix(labels_test,predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print cf
