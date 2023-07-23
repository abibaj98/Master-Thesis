from MetaLearner import *
import numpy as np
# import time
import jsonpickle

# set dtype standard
tf.keras.backend.set_floatx('float64')

# set seed
tf.keras.utils.set_random_seed(8953)

# load data
file_name = "data_12setups.json"

# load with jsonpickle
f = open(file_name, 'r')
json_str = f.read()
data = jsonpickle.decode(json_str)

# some values
n_setups = 12  # TODO: change
sample_sizes = [500, 1000, 2000, 5000]
n_runs = 1  # TODO: change

# All MetaLearners in combination with each Baselearner (Random Forest, Lasso, Neural Network)
learners = [TLearner('rf'), SLearner('rf'), XLearner('rf'), RLearner('rf'), DRLearner('rf'), RALearner('rf'),
            PWLearner('rf'), ULearner('rf'),
            TLearner('lasso'), SLearner('lasso'), XLearner('lasso'), RLearner('lasso'), DRLearner('lasso'),
            RALearner('lasso'), PWLearner('lasso'), ULearner('lasso'),
            TLearner('nn'), SLearner('nn'), XLearner('nn'), RLearner('nn'), DRLearner('nn'), RALearner('nn'),
            PWLearner('nn'), ULearner('nn')]

# empty results list
results = []

for i in range(n_setups):
    results.append([])

for i in range(n_setups):
    for baselearner in range(3):
        results[i].append([])

dim = data[0][0][0][0].shape[1]  # dimension of x


# helper function to get y, x, w, tau, out of data[][][][]
def get_variables(dataset, setup, run, samplesize, train_test):
    y = dataset[setup][run][samplesize][train_test][:, 0]
    x = dataset[setup][run][samplesize][train_test][:, 1:(dim - 2)]
    w = dataset[setup][run][samplesize][train_test][:, (dim - 2)]
    tau = dataset[setup][run][samplesize][train_test][:, (dim - 1)]
    return y, x, w, tau


# set index
b = 0
m = 0
s = 0

"""
# loop
for i in range(n_setups):
    print(f'Setup: {i + 1}')
    # array for all mses for one setup
    setup_mse = np.empty(shape=(0, 24))
    s = 0  # restart index for samplesize
    for size in sample_sizes:
        print(f'Sample Size: {s + 1}')
        # array for all mses for one setup and samplesize.
        size_mse = np.empty(shape=(0, 24))
        for r in range(n_runs):
            print(f'Run: {r + 1}')
            # array for all mses in one setup, samplesize and run.
            mses = np.empty(shape=(1, 24))
            # get data for specific setup, samplesize and run.
            temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables(dataset=data, setup=i,
                                                                                     samplesize=s, run=r,
                                                                                     train_test=0)
            temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables(dataset=data, setup=i,
                                                                                 samplesize=s, run=r,
                                                                                 train_test=1)
            # restart index for metalearner
            m = 0
            for learn in learners:
                if m % 8 == 0:
                    print(f'BaseLearner: {(m / 8) + 1}')
                print(f'Learner {m + 1}: {learn}')
                # tic = time.time()
                # training and testing MetaLearner.
                learner = learn
                learner.fit(temp_x_train, temp_y_train, temp_w_train)
                predictions = learner.predict(temp_x_test)
                temp_mse = ((predictions - temp_tau_test) ** 2).mean()
                # append mse of specific metalearner to 'mses'.
                mses[0, m] = temp_mse
                # print time
                # toc = time.time()
                # print(f'Time: {round(toc - tic, 4)} seconds.')
                # update index
                m += 1
            # append 'mses' to 'size_mse'.
            size_mse = np.append(size_mse, mses, axis=0)
        # append 'size_mse' to 'setup_mse'.
        setup_mse = np.append(setup_mse, size_mse, axis=0)
        # update index
        s += 1
    # append to results
    results[i][0] = setup_mse[:, 0:8]  # random forest
    results[i][1] = setup_mse[:, 8:16]  # lasso
    results[i][2] = setup_mse[:, 16:24]  # neural network
    
"""

# loop NEW
for i in range(n_setups):
    print(f'Setup: {i + 1}')
    # array for all mses for one setup
    setup_mse = np.empty(shape=(0, 24))
    for r in range(n_runs):
        print(f'Run: {r + 1}')
        s = 0  # restart index for samplesize
        for size in sample_sizes:
            print(f'Sample Size: {s + 1}')
            # array for all mses for one setup and samplesize.
            run_mse = np.empty(shape=(0, 24))
            # array for all mses in one setup, samplesize and run.
            mses = np.empty(shape=(1, 24))
            # get data for specific setup, samplesize and run.
            temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables(dataset=data, setup=i,
                                                                                     samplesize=s, run=r,
                                                                                     train_test=0)
            temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables(dataset=data, setup=i,
                                                                                 samplesize=s, run=r,
                                                                                 train_test=1)
            # restart index for metalearner
            m = 0
            for learn in learners:
                if m % 8 == 0:
                    print(f'BaseLearner: {(m / 8) + 1}')
                print(f'Learner {m + 1}: {learn}')
                # tic = time.time()
                # training and testing MetaLearner.
                learner = learn
                learner.fit(temp_x_train, temp_y_train, temp_w_train)
                predictions = learner.predict(temp_x_test)
                temp_mse = ((predictions - temp_tau_test) ** 2).mean()
                # append mse of specific metalearner to 'mses'.
                mses[0, m] = temp_mse
                # print time
                # toc = time.time()
                # print(f'Time: {round(toc - tic, 4)} seconds.')
                # update index
                m += 1
            # append 'mses' to 'size_mse'.
            run_mse = np.append(run_mse, mses, axis=0)
            # update index
            s += 1
        # append 'size_mse' to 'setup_mse'.
        setup_mse = np.append(setup_mse, run_mse, axis=0)
    # append to results
    results[i][0] = setup_mse[:, 0:8]  # random forest
    results[i][1] = setup_mse[:, 8:16]  # lasso
    results[i][2] = setup_mse[:, 16:24]  # neural network

# file name
file_name = "results_12setups.json"

# SAVE LIST AS JSON FILE
f = open(file_name, 'w')
json_obj = jsonpickle.encode(results)
f.write(json_obj)
f.close()

# end
