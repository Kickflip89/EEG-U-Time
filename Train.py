import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import ConvNetwork as CN

#This program trains 10 models, one for each participant and
#plots the training validation accuracy vs Epoch and lists
#the mean accuracy from the 10 models.
sns.set()

accs = []
hists = []
parts = list(range(1,11))
for participant in parts:
    network = CN.ConvNetwork(dilation=4, factor=5, single=True,
                            p=participant)
    histories = network.TrainModel(CN.readData(CN.SUM_PATH))
    hists.append(histories.history['val_acc'])
    accs.append(np.max(histories.history['val_acc']))

#plot results:
acc = np.mean(accs)
plt.figure()
for participant in parts:
    epochs = list(range(len(hists[participant-1])))
    plt.plot(epochs, hists[participant-1],
                label="P" + str(participant))
plt.title("Mean Validation Accuracy: " + str(acc))
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
