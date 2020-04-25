# EEG-U-Time
Uses a U-Time Network to Process EEG data (classifier)

./models contains saved versions of the models trained in various ways.

LOO is leave-one-out, where the number is the participant left out.
m# is a model trained on a subset of participant#'s data.

ConvNetwork.py contains network architecture and utilities to generate
batches, training and test data, as well as work with the raw data files.

SUM_PATH and DATA_PATH point to summary.txt that has information
about the data, and it is assumed the full data set in .mat files is kept
in ./alldata

Confusion.py generates a confusion matrix for a participant declared at
the top of the program for both LOO models and ensemble models

Ensemble.py does a 10-fold cross-validation of an ensemble method using the
saved models in ./models.

Train.py shows an example of training individual models and outputs a graph
showing the epochs and the validation accuracy.

Summary.txt contains information ABOUT the data files, which makes generating
training / test splits and batches for training easier.
