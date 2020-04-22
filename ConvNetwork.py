import numpy as np
import pandas as pd
import random
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Conv1D, BatchNormalization, add, Activation, MaxPooling1D, UpSampling1D
from keras.layers import AveragePooling1D, Concatenate, Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from os import listdir

#constants and hyperparameters
DEF_LR = .01
DEF_LR_DEC = .01
POOLS = [8,6,4,2]
STEPS = 384
NUM_CLASSES = 9
FILT_FACTOR = 5
KERNEL_SIZE = 7
DILATION = 4
EPOCHS = 30
C0 = 'decline_075'
C1 = 'decline_selfpaced'
C2 = 'incline_075'
C3 = 'incline_selfpaced'
C4 = 'level_050'
C5 = 'level_075'
C6 = 'level_100'
C7 = 'level_125'
C8 = 'level_selfpaced'
DATA_PATH = './alldata'
SUM_PATH = './summary.txt'

#Utilities

#Makes summaries for files in './data'
def getData():
    data = dict()
    files = listdir(DATA_PATH)
    files.sort()
    for file in files:
        path = DATA_PATH + '/' + file
        mat = loadmat(path)  # load mat-file
        mdata = mat['EEG']  # variable in mat file
        mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
        # * SciPy reads in structures as structured NumPy arrays of dtype object
        # * The size of the array is the size of the structure array, not the number
        #   elements in any particular field. The shape defaults to 2-dimensional.
        # * For convenience make a dictionary of the data using the names from dtypes
        # * Since the structure has only one element, but is 2-D, index it at [0, 0]
        ndata = {n: mdata[n][0, 0] for n in mdtype.names}
        length = ndata['data'].shape[1]
        print(file, ndata['data'].shape[0])
        data[path] = length
    fhandle = open(SUM_PATH, 'w')
    for key in data:
        fhandle.write(key + ' ' + str(data[key]) + '\n')
        fhandle.close()
    return data


#reads the data summaries from the file at FPATH
def readData(FPATH):
    try:
        fhandle = open(FPATH, 'r')
    except:
        return getData()
    dat = dict()
    for line in fhandle:
        words = line.split()
        dat[words[0]] = int(words[1])
    return dat

#Builds a dataframe from a .mat file (fname)
def getFrame(fname):
    mat = loadmat(fname)  # load mat-file
    mdata = mat['EEG']  # variable in mat file
    mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
    # * SciPy reads in structures as structured NumPy arrays of dtype object
    # * The size of the array is the size of the structure array, not the number
    #   elements in any particular field. The shape defaults to 2-dimensional.
    # * For convenience make a dictionary of the data using the names from dtypes
    # * Since the structure has only one element, but is 2-D, index it at [0, 0]
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    #create dataframe dictionary
    cols = dict()
    curr = 0
    for chan in range(128):
        string = "chan" + str(curr)
        cols[string] = ndata['data'][curr]
        curr += 1
    df=pd.DataFrame(cols)

    #rescale from 0 - 512
    maxes = df.apply(np.max)
    mins = df.apply(np.min)
    for n in range(len(maxes)):
        maxes[n] = maxes[n] - mins[n]
        df[maxes.index[n]] = (df[maxes.index[n]] - mins[n])/maxes[n]*511
    df = df.astype(int)
    return df

#Returns the class number based on 3 or 9 class versions
def getClass(number, allClasses):
    if allClasses:
        return number % 9
    elif number < 2:
        return 0
    elif number < 4:
        return 1
    return 2

#Creates the arrays for validation data
def makeValArrays(data, allClasses):
    keyList = list(data.keys())
    arrayChunks = []
    for n in range(len(keyList)):
        print(keyList[n])
        df = getFrame(keyList[n])
        for chunk in data[keyList[n]]:
            cf = df[chunk[0]:chunk[1]].copy()
            if allClasses:
                classes = 9
            else:
                classes = 3
            target = np.zeros((classes))
            num = getClass(n, allClasses)
            target[num] = 1
            arrayChunks.append([np.array(cf), target])
    random.shuffle(arrayChunks)
    X, Y = zip(*arrayChunks)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    return (X, Y)

#converts a number into a string for file processing
def getPart(number):
    if number < 10:
        return '0' + str(number) + '_'
    else:
        return str(number) + '_'

#Class for the model architecture and training
class ConvNetwork:

    def __init__(self, batches=45, alpha = DEF_LR, alpha_dec = DEF_LR_DEC, pools = POOLS, steps = STEPS,
                filts = NUM_CLASSES, kernel=KERNEL_SIZE, dilation = DILATION,
                factor = FILT_FACTOR, splitAll = True, allClasses=True, single = False, p=0,
                softmax = True, epochs = EPOCHS, saveAs = 'best_model.h5'):
        self.batchSize = batches
        self.channels = 128
        self.steps = steps
        self.pools = pools
        self.alpha = alpha
        #whether to train on 9 or 3 classes
        if allClasses:
            self.filters = filts
        else:
            self.filters = filts//3
        self.kSize = kernel
        self.dRate = dilation
        self.alpha_decay = alpha_dec
        #whether to train on all data or LOO
        self.trainAll = splitAll
        self.fact = factor
        #participants to train on
        self.trainParts = list(range(1,11))
        self.allClasses = allClasses
        #whether to train on a single model
        self.single = single
        if single:
            self.trainAll = True
        #validation participant or model participant
        self.participant = p
        self.epochs = epochs
        if softmax:
            self.act = 'softmax'
        else:
            self.act = 'relu'
        self.mpath = saveAs
        self.m = self.buildModel()

    #This function grew into a function that is probably
    #too large but handles building test/training sets
    #and preparing the model to use the batch fit_generator
    #to start training
    def TrainModel(self, data):
        if self.trainAll:
            if self.single:
                nDat = dict()
                self.train = dict()
                self.test = dict()
                for key in data:
                    tstring = './alldata/L' + getPart(self.participant)
                    if key.startswith(tstring):
                        print(data[key])
                        self.train[key] = data[key]
                        self.test[key] = data[key]
                        nDat[key] = data[key]
            #Training on all participants
            else:
                self.train = data.copy()
                self.test = data.copy()
                nDat = data.copy()
            #number of chunks to sample for validation
            val = 24
        #Leave-One-Out training
        else:
            if(self.participant==0):
                self.testPart = self.trainParts.pop(np.random.randint(0,9))
            else:
                self.testPart = self.trainParts.pop(self.participant-1)
            self.train = dict()
            self.test = dict()
            self.minPart = 2
            print("test set: ", self.testPart)
            val = min(data.values()) // 384 // 5
            nDat = data.copy()
        self.indices = nDat.copy()
        random.seed()
        allDat = 0
        val = min(data.values()) // 384 // 5

        #create indices for chunks for validation and training splits
        #create dictionary for number of chunks in training set
        for key in nDat:
            total = data[key] - 750
            num = total // STEPS
            over = total % STEPS
            start = np.random.randint(0,over)
            curr = 750 + start
            chunks = []
            while curr + STEPS < total + 750:
                chunks.append([curr, curr+STEPS])
                curr += STEPS
            random.shuffle(chunks)
            if self.trainAll:
                self.train[key] = chunks[val+1:]
                self.test[key] = chunks[0:val]
                entries = len(chunks) - val - 1
                self.indices[key] = list(range(entries))
                allDat += entries
            elif key.startswith('./alldata/L' + getPart(self.testPart)):
                self.test[key] = chunks[val:(2*val)]
            else:
                self.train[key] = chunks
                entries = len(chunks)
                self.indices[key] = list(range(entries))
                allDat += entries

        #prepare first batch data
        self.numDat = allDat
        self.currDat = 0
        self.batch = self.indices.copy()
        #list of participants to train for each class
        self.parts = np.random.choice(self.trainParts, 9, replace=True)
        if self.single:
            for n in range(len(self.parts)):
                self.parts[n] = self.participant
        #Build list of DataFrames for each file / class
        self.cnames = [C0, C1, C2, C3, C4, C5, C6, C7, C8]
        self.used = [None]*9
        self.keys = []
        self.frames = []
        for n in range(len(self.cnames)):
            part = getPart(self.parts[n])
            self.used[n] = [self.parts[n]]
            fname = DATA_PATH + '/L' + part + self.cnames[n] + '.mat'
            self.frames.append(getFrame(fname))
            self.keys.append(fname)

        self.X_Val, self.Y_Val = makeValArrays(self.test, self.allClasses)
        #print(self.batch)
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1,
                    patience=10)
        mc = ModelCheckpoint(self.mpath, monitor='val_acc', mode='max',
                            verbose=1, save_best_only=True)
        print(self.numDat, self.batchSize)
        histories = self.m.fit_generator(self.batchGenerator(),
                        (self.numDat // self.batchSize + 1), self.epochs,
                        validation_data = (self.X_Val, self.Y_Val),
                        callbacks=[es,mc])
        return histories

    #Makes batches of training data by getting receiving
    #A dictionary of what chunks to include and returning
    #a np array
    def makeArrays(self, data):
        arrayChunks = []
        for n in range(len(data)):
            #print(keyList[n])
            for chunk in data[n]:
                cf = self.frames[n][chunk[0]:chunk[1]].copy()
                target = np.zeros((self.filters))
                num = getClass(n, self.allClasses)
                target[num] = 1
                arrayChunks.append([np.array(cf), target])
        random.shuffle(arrayChunks)
        #print(arrayChunks[0][0].shape, arrayChunks[0][1].shape)
        X, Y = zip(*arrayChunks)
        #print(len(X), len(Y))
        X = np.array(X)
        Y = np.array(Y)
        #print(X.shape, Y.shape)
        return (X, Y)

    #creates batches by selecting 5 chunks of data
    #from each file being used for training.
    #Also handles resetting the data for EPOCHS
    #and replacing files that have been fully trained
    def batchGenerator(self):
        reset = False
        self.numBatches = 0
        while True:
            currBatch = [None]*9
            newFrames = dict()
            for n in range(len(self.cnames)):
                key = self.keys[n]
                numChunks = len(self.batch[key])
                #print(key, numChunks)
                indices = []
                #get five chunks
                if numChunks >= 5:
                    for index in np.random.randint(0,numChunks-4,5):
                        indices.append(self.batch[key].pop(index))
                    self.currDat += 5
                #or the remaining chunks and move to a different file
                else:
                    self.currDat += numChunks
                    for index in self.batch[key]:
                        indices.append(index)
                    self.batch[key] = []
                    if not self.single:
                        tries = 0
                        while self.parts[n] in self.used[n] and tries < 100:
                            self.parts[n] = np.random.choice(self.trainParts, 1)
                            tries += 1
                        if tries > 100:
                            reset = True
                            print("\nTries triggered")
                        self.used[n].append(self.parts[n])
                        self.keys[n] = DATA_PATH + '/L' + getPart(self.parts[n]) + self.cnames[n] + '.mat'
                        newFrames[n] = self.keys[n]

                #make data chunks for batch
                chunks = []
                for index in indices:
                    if index < len(self.train[key]):
                        chunks.append(self.train[key][index].copy())
                currBatch[n] = chunks

            #make batch
            #print(currBatch)
            batch_x, batch_y = self.makeArrays(currBatch)
            self.numBatches += 1

            #book keeping for new batch or new frames
            for key in newFrames:
                self.frames[key] = getFrame(newFrames[key])

            if reset or self.currDat >= (self.numDat - 2) or self.numBatches >= ((self.numDat
                                                                                 //self.batchSize) + 1):
                print("\nResetting", self.currDat, self.numDat,
                      self.numBatches, (self.numDat // self.batchSize +1) )
                for key in self.train:
                    entries = len(self.train[key])
                    self.batch[key] = list(range(entries))
                self.parts = np.random.choice(self.trainParts, 9, replace=True)
                if self.single:
                    for n in range(len(self.parts)):
                        self.parts[n] = self.participant
                print(self.parts)
                for n in range(len(self.cnames)):
                    part = getPart(self.parts[n])
                    self.used[n] = [self.parts[n]]
                    fname = DATA_PATH + '/L' + part + self.cnames[n] + '.mat'
                    self.frames[n] = getFrame(fname)
                    self.keys[n] = fname
                reset = False
                self.currDat = 0
                self.numBatches = 0
            yield(batch_x,batch_y)

    #Builds the model architecture based on hparams
    def buildModel(self):
        data = Input(shape=(self.steps, self.channels))
        featureMaps = []

        #Build Encoder
        pconv1 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                        activation = 'relu', dilation_rate=self.dRate)(data)
        pbn1 = BatchNormalization()(pconv1)
        pconv2 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                        activation = 'relu', dilation_rate=self.dRate)(pbn1)
        pbn2 = BatchNormalization()(pconv2)
        featureMaps.append(pbn2)
        lastIter = MaxPooling1D(self.pools[0])(pbn2)
        for i in range(1,len(self.pools)):
            conv1 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                           activation = 'relu', dilation_rate=self.dRate)(lastIter)
            bn1 = BatchNormalization()(conv1)
            conv2 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                           activation = 'relu', dilation_rate=self.dRate)(bn1)
            bn2 = BatchNormalization()(conv2)
            featureMaps.append(bn2)
            lastIter = MaxPooling1D(pool_size = self.pools[i])(bn2)

        #Intermediate Convolutions:
        post1 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                       activation = 'relu', dilation_rate=self.dRate)(lastIter)
        postbn1 = BatchNormalization()(post1)
        post2 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                       activation = 'relu', dilation_rate=self.dRate)(postbn1)
        lastIter = BatchNormalization()(post2)

        #Build Decoder:
        for i in range(len(self.pools)):
            ups = UpSampling1D(size=self.pools[3-i])(lastIter)
            c1 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                        activation = 'relu', dilation_rate=self.dRate)(ups)
            b1 = BatchNormalization()(c1)
            conc = Concatenate()([b1, featureMaps[3-i]])
            c2 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                        activation = 'relu', dilation_rate=self.dRate)(conc)
            b2 = BatchNormalization()(c2)
            c3 = Conv1D(self.fact*self.filters, self.kSize, padding='same',
                        activation = 'relu', dilation_rate=self.dRate)(b2)
            lastIter = BatchNormalization()(c3)

        #Classifier:
        segment = AveragePooling1D(pool_size=self.steps, strides=self.steps)(lastIter)
        output = Conv1D(self.filters, 1, activation=self.act, padding='same')(segment)
        output = Flatten()(output)
        model = Model(inputs=data, outputs=output)
        model.compile(optimizer="Adam", loss='categorical_crossentropy',
                      metrics=['acc'])
        return model
