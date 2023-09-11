# import packages
import numpy as np
from numpy import load

# load 100 realisations of IHDP train set (672 units)
ihdp_train = load("ihdp_npci_1-100.train.npz")
files_train = ihdp_train.files

# load 100 realisations of IHDP test set (72 units)
ihdp_test = load('ihdp_npci_1-100.test.npz')
files_test = ihdp_test.files

# Pre-process the data to be in the same format as fully-synthetic data
ihdp_train_processed = [[] for _ in range(100)]
ihdp_train_x = [[] for _ in range(100)]
ihdp_train_tau = ihdp_train['mu1'] - ihdp_train['mu0']  # compute tau
# x --> make it a list, each entry being one realisation, containing all features
for i in range(100):
    realisation = np.zeros(shape=(672, 25))
    for n in range(672):
        realisation[n, :] = ihdp_train['x'][n][:, i][:, np.newaxis].T
    ihdp_train_x[i] = realisation
# concatenate [y, x, t, tau]
for i in range(100):
    temporary_set = np.concatenate((np.expand_dims(ihdp_train['yf'][:, i], axis=1)
                                    , ihdp_train_x[i], np.expand_dims(ihdp_train['t'][:, i], axis=1),
                                    np.expand_dims(ihdp_train_tau[:, i], axis=1)), axis=1)
    ihdp_train_processed[i] = temporary_set

# save the pre-processed training data
np.save('ihdp_train_processed', ihdp_train_processed)

# Same for test set
ihdp_test_processed = [[] for _ in range(100)]
ihdp_test_x = [[] for _ in range(100)]
ihdp_test_tau = ihdp_test['mu1'] - ihdp_test['mu0']

# process x
for i in range(100):
    realisation = np.zeros(shape=(75, 25))
    for n in range(75):
        realisation[n, :] = ihdp_test['x'][n][:, i][:, np.newaxis].T
    ihdp_test_x[i] = realisation

# concatenate [y, x, t, tau]
for i in range(100):
    temporary_set = np.concatenate((np.expand_dims(ihdp_test['yf'][:, i], axis=1)
                                    , ihdp_test_x[i], np.expand_dims(ihdp_test['t'][:, i], axis=1),
                                    np.expand_dims(ihdp_test_tau[:, i], axis=1)), axis=1)
    ihdp_test_processed[i] = temporary_set

# save the processed testing data
np.save('ihdp_test_processed', ihdp_test_processed)
