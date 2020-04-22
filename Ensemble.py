import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import ConvNetwork as CN
from keras.models import model_from_json

#Change as needed to change output
MODEL_PATH = "./models/"
MJSON = "./models/model.json"
WEIGHTED = False

#returns aggregated ensemble activations or distributions
def getPredictions(models, X, weighted):
    preds = np.zeros((X.shape[0], 9))
    smax = True
    if weighted:
        smax = False
    net = CN.ConvNetwork(dilation = 4, factor = 5, splitAll = True,
            single = True, p = 1, softmax=smax)
    model = net.m
    for num in models:
        #print("getting ./m" + str(num+1))
        weightPath = MODEL_PATH + 'm' + str(num) + "model.h5"
        model.load_weights(weightPath)
        Y_Pred = model.predict(X)
        #a = np.argmax(Y_Pred, axis=-1)
        #b = np.zeros((a.size, 9))
        #b[np.arange(a.size), a] = 1
        preds = preds + Y_Pred
    return preds

accs = []
for n in range(0,10):
    print("Validation participant: ", n+1)
    mList = list(range(1,11))
    mList.pop(n)
    #generate test set
    test = CN.ConvNetwork(dilation=4, factor=5, single=True, p=n+1,
                epochs = 1)
    test.TrainModel(CN.readData(CN.SUM_PATH))
    X = test.X_Val
    Y = np.argmax(test.Y_Val, axis=-1)
    print("******************************************")
    print("Validation participant: ", n+1)
    print("******************************************")
    preds = getPredictions(mList, X, WEIGHTED)
    preds = np.argmax(preds, axis=-1)
    accs.append(len(preds[preds==Y]) / len(preds))
    print("Accuracy: ", accs[-1])
print("******************************************")
print("Accuracies: ", accs)
print("Mean: ", np.mean(accs))
print("******************************************")
