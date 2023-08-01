# import from files
from Meta import *
# import packages
import numpy as np
import tensorflow as tf
import jsonpickle
import configargparse

# set dtype standard
tf.keras.backend.set_floatx('float64')

# set seed
tf.keras.utils.set_random_seed(KERAS_SEED)

# load data
# file_name = "data_12setups.json"
train = np.load('/Users/arberimbibaj/Documents/Master Thesis ETH/DataSets /IHDP/ihdp_train_processed.npy')
test = np.load('/Users/arberimbibaj/Documents/Master Thesis ETH/DataSets /IHDP/ihdp_test_processed.npy')

# All MetaLearners in combination with each Baselearner (Random Forest, Lasso, Neural Network)

learners = [TLearner('rf'), SLearner('rf'), XLearner('rf'), RLearner('rf'), DRLearner('rf'), RALearner('rf'),
            PWLearner('rf'), ULearner('rf'),  # baselearner = random forest
            TLearner('lasso'), SLearner('lasso'), XLearner('lasso'), RLearner('lasso'), DRLearner('lasso'),
            RALearner('lasso'), PWLearner('lasso'), ULearner('lasso'),  # baselearner = linear models
            TLearner('nn'), SLearner('nn'), XLearner('nn'), RLearner('nn'), DRLearner('nn'), RALearner('nn'),
            PWLearner('nn'), ULearner('nn')]  # baselearner = neural network

# empty results list
results = [[] for _ in range(3)]  # for the 3 base learners

dim = train[0].shape[1]  # dimension of data frame


# helper function to get y, x, w, tau, out of data[][][][]
def get_variables_ihdp(dataset, run):
    y = dataset[run][:, 0]
    x = dataset[run][:, 1:(dim - 2)]
    w = dataset[run][:, (dim - 2)]
    tau = dataset[run][:, (dim - 1)]
    return y, x, w, tau


# one setup
def run_ihdp(runs):
    print("------------------------------------------")
    print("Running the experiment on the IHDP dataset")
    print("------------------------------------------")
    all_mse = np.empty(shape=(0, 24))
    for run in range(runs):
        print(f'Run: {run + 1}')
        # mses for one run
        mses = np.empty(shape=(1, 24))
        # get data for specific setup, samplesize and run.
        temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables_ihdp(dataset=train, run=run)
        temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables_ihdp(dataset=test, run=run)
        # restart index for metalearner
        for m, learn in enumerate(learners):
            if m % 8 == 0:
                print('---------------')
            print(f'Fitting Learner {m + 1}: {learn.name}')
            # training and testing MetaLearner.
            learner = learn
            learner.fit(temp_x_train, temp_y_train, temp_w_train)
            predictions = learner.predict(temp_x_test)
            temp_mse = ((predictions - temp_tau_test) ** 2).mean()
            # append mse of specific metalearner to 'mses'.
            mses[0, m] = temp_mse
        # append 'mses' to 'all_mse'.
        all_mse = np.append(all_mse, mses, axis=0)
        print('-------------------------')
    print("Done")
    print('-------------------------')
    return all_mse


def parser_arguments():
    p = configargparse.ArgParser()
    p.add('--runs', default=100, type=int, choices=range(1, 101),
          help='Type in how many runs you would like to run. Can be from 1 to 100.')
    return p.parse_args()


def main():
    argument = parser_arguments()
    # run experiment (ihdp)
    res = run_ihdp(argument.runs)
    # append to results
    results[0] = res[:, 0:8]  # random forest
    results[1] = res[:, 8:16]  # lm
    results[2] = res[:, 16:24]  # neural network

    # save results
    results_file_name = f"results_ihdp_{argument.runs}_runs.json"
    r = open(results_file_name, 'w')
    json_obj = jsonpickle.encode(results)
    r.write(json_obj)
    r.close()


if __name__ == "__main__":
    main()
