import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import ConvNetwork as CN
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix

#Change as needed to change output
MODEL_PATH = "./models"
MJSON = "./models/model.json"
WEIGHTED = False
LOO = True
PARTICIPANT = 5

#returns aggregated ensemble activations or distributions
def getPredictions(models, X, weighted):
    preds = np.zeros((X.shape[0], 9))
    smax = True
    if weighted:
        smax = False
    net = CN.ConvNetwork(dilation = 4, factor = 5, splitAll = True,
            single = True, p = PARTICIPANT, softmax=smax)
    model = net.m
    for num in models:
        #print("getting ./m" + str(num+1))
        weightPath = MODEL_PATH + "/m" + str(num) + "model.h5"
        model.load_weights(weightPath)
        Y_Pred = model.predict(X)
        #a = np.argmax(Y_Pred, axis=-1)
        #b = np.zeros((a.size, 9))
        #b[np.arange(a.size), a] = 1
        preds = preds + Y_Pred
    return preds

#generate test set
test = CN.ConvNetwork(dilation=4, factor=5, single=True, p=PARTICIPANT,
            epochs = 1)
test.TrainModel(CN.readData(CN.SUM_PATH))

#LOO Confusion Matrix
loo = "/loo" + str(PARTICIPANT) + ".h5"
mp = MODEL_PATH + loo
test.m.load_weights(mp)
preds = np.argmax(test.m.predict(test.X_Val), axis=-1)
print("******************************************")
print("Validation Participant = ", PARTICIPANT)
print("Leave-One-Out Matrix:")
print(confusion_matrix(preds, np.argmax(test.Y_Val, axis=-1)))
print("******************************************")

#Ensemble confusion Matrix
mList = list(range(1,11))
mList.pop(PARTICIPANT-1)
preds = np.argmax(getPredictions(mList, test.X_Val, WEIGHTED), axis=-1)
print("******************************************")
print("Ensemble Matrix:")
print(confusion_matrix(preds, np.argmax(test.Y_Val, axis=-1)))
print("******************************************")
