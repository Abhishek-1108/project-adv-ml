
from  models import *
import scipy.io as sio

labels = sio.loadmat('labels.mat')
labels = labels['labels'][0]

#Loading data

data = sio.loadmat('ccaFuse_sum.mat')

labels_test = np.transpose(data['labels_test'])[0]
labels_train = np.transpose(data['labels_train'])[0]
testZ_sum = data['testZ_sum']
testX = data['testX']
testY = data['testY']
trainX = data['trainX']
trainY = data['trainY']
trainZ_sum = data['trainZ_sum']

# Top 4 frequent labels selected

temp = np.unique(labels_train, return_counts=True)
indices = np.argsort(temp[1])
values = temp[0][indices]
high = [values[-1],values[-2],values[-3],values[-4]]
print values[-1],values[-2],values[-3],values[-4]

# Picking data corresponding to top 4 labels

Fused_labels_train = []
Fused_labels_test = []
Fused_train = []
Fused_test = []
testX_new = []
testY_new = []
trainX_new = []
trainY_new = []

for i in xrange(len(labels_train)):
    if labels_train[i] in high:
        Fused_labels_train.append(labels_train[i])
        Fused_train.append(trainZ_sum[i])
print high
        trainX_new.append(trainX[i])
        trainY_new.append(trainY[i])

print len(Fused_labels_train)
Fused_labels_train = np.array(Fused_labels_train)
Fused_train = np.array(Fused_train)
trainX_new = np.array(trainX_new)
trainY_new = np.array(trainY_new)
print len(Fused_train), len(trainX_new), len(trainY_new)


for i in xrange(len(labels_test)):
    if labels_test[i] in high:
        Fused_labels_test.append(labels_test[i])
        Fused_test.append(testZ_sum[i])
        testX_new.append(testX[i])
        testY_new.append(testY[i])


print len(Fused_labels_test)
Fused_labels_test = np.array(Fused_labels_test)
Fused_test = np.array(Fused_test)
testX_new = np.array(testX_new)
testY_new = np.array(testY_new)
print len(Fused_test), len(testX_new), len(testY_new)

# Video only

kernelSVM_classify(trainX_new,Fused_labels_train,testX_new,Fused_labels_test)

# Audio only

kernelSVM_classify(trainY_new,Fused_labels_train,testY_new,Fused_labels_test)

# Fused

kernelSVM_classify(Fused_train,Fused_labels_train,Fused_test,Fused_labels_test)


# Video Only

xgb_classify(trainX_new,Fused_labels_train,testX_new,Fused_labels_test)


# Audio Only

xgb_classify(trainY_new,Fused_labels_train,testY_new,Fused_labels_test)


# Fused

xgb_classify(Fused_train,Fused_labels_train,Fused_test,Fused_labels_test)


