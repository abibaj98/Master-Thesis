# from MetaLearner import *
from Meta import *
import numpy as np
# import time
import jsonpickle

# set dtype standard
tf.keras.backend.set_floatx('float64')

# set seed
tf.keras.utils.set_random_seed(8953)

# load data
# file_name = "data_12setups.json"
train = np.load('ihdp_train_processed.npy')
test = np.load('ihdp_test_processed.npy')

# some values
runs = 2

# All MetaLearners in combination with each Baselearner (Random Forest, Lasso, Neural Network)
learners = [TLearner('rf'), SLearner('rf'), XLearner('rf'), RLearner('rf'), DRLearner('rf'), RALearner('rf'),
            PWLearner('rf'), ULearner('rf'),
            TLearner('lasso'), SLearner('lasso'), XLearner('lasso'), RLearner('lasso'), DRLearner('lasso'),
            RALearner('lasso'), PWLearner('lasso'), ULearner('lasso'),
            TLearner('nn'), SLearner('nn'), XLearner('nn'), RLearner('nn'), DRLearner('nn'), RALearner('nn'),
            PWLearner('nn'), ULearner('nn')]

# empty results list
results = []

for baselearner in range(3):
    results.append([])

dim = train[0].shape[1]  # dimension of data frame


# helper function to get y, x, w, tau, out of data[][][][]
def get_variables_ihdp(dataset, run):
    y = dataset[run][:, 0]
    x = dataset[run][:, 1:(dim - 2)]
    w = dataset[run][:, (dim - 2)]
    tau = dataset[run][:, (dim - 1)]
    return y, x, w, tau


# loop NEW

# all mses
all_mse = np.empty(shape=(0, 24))

for i in range(runs):
    print(f'Run: {i + 1}')
    # mses for one run
    mses = np.empty(shape=(1, 24))
    # get data for specific setup, samplesize and run.
    temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables_ihdp(dataset=train, run=i)
    temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables_ihdp(dataset=test, run=i)
    # restart index for metalearner
    m = 0
    for learn in learners:
        if m % 8 == 0:
            print(f'BaseLearner: {(m / 8) + 1}')
        print(f'Learner {m + 1}: {learn}')
        # training and testing MetaLearner.
        learner = learn
        learner.fit(temp_x_train, temp_y_train, temp_w_train)
        predictions = learner.predict(temp_x_test)
        temp_mse = ((predictions - temp_tau_test) ** 2).mean()
        # append mse of specific metalearner to 'mses'.
        mses[0, m] = temp_mse
        # update index
        m += 1
    # append 'mses' to 'all_mse'.
    all_mse = np.append(all_mse, mses, axis=0)
    # update index

# append to results
results[0] = all_mse[:, 0:8]  # random forest
results[1] = all_mse[:, 8:16]  # lasso
results[2] = all_mse[:, 16:24]  # neural network

# file name
# file_name = "results_12_setups.json"
file_name = "results_ihdp_1e-3.json"

# SAVE LIST AS JSON FILE
f = open(file_name, 'w')
json_obj = jsonpickle.encode(results)
f.write(json_obj)
f.close()

# end

# TODO: maybe just simulate betas once in a run, not each sample size!!
