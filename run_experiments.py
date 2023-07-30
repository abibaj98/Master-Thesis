# from MetaLearner import *
from Meta import *
import numpy as np
import jsonpickle

# argparse
import configargparse

# set dtype standard
tf.keras.backend.set_floatx('float64')

# set seed
tf.keras.utils.set_random_seed(8953)

# load data with jsonpickle
file_name = "simulated_data.json"
# file_name = "/Users/arberimbibaj/Documents/Master Thesis ETH/DataSets /Generated/simulated_data.json"

f = open(file_name, 'r')
json_str = f.read()
data = jsonpickle.decode(json_str)

# some parameters
n_setups = 1
sample_sizes = [500, 1000, 2000, 5000]
n_runs = 10

# All MetaLearners in combination with each Baselearner (Random Forest, Linear Models, Neural Network)
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

dim = data[0][0][0][0].shape[1]  # dimension of x


# helper function to get y, x, w, tau, out of data
def get_variables(dataset, setup, run, samplesize, train_test):
    y = dataset[setup][run][samplesize][train_test][:, 0]
    x = dataset[setup][run][samplesize][train_test][:, 1:(dim - 2)]
    w = dataset[setup][run][samplesize][train_test][:, (dim - 2)]
    tau = dataset[setup][run][samplesize][train_test][:, (dim - 1)]
    return y, x, w, tau


def run_experiment(setting):
    print(f'Running Setup: {setting + 1}')
    # array; all mses for the setup
    setup_mse = np.empty(shape=(0, 24))
    for r in range(n_runs):
        print(f'Run: {r + 1}')
        s = 0  # restart index for samplesize
        # array; all mses for one specific run.
        run_mse = np.empty(shape=(0, 24))
        for _ in sample_sizes:
            print(f'Sample Size: {s + 1}')
            # array; all mses for one specific run and sample size
            mses = np.empty(shape=(1, 24))
            # get data for specific setup, run and sample size.
            temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables(dataset=data, setup=setting,
                                                                                     samplesize=s, run=r,
                                                                                     train_test=0)
            temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables(dataset=data, setup=setting,
                                                                                 samplesize=s, run=r,
                                                                                 train_test=1)
            # restart index for metalearner
            m = 0
            for learn in learners:
                if m % 8 == 0:
                    print(f'BaseLearner: {int((m / 8) + 1)}')
                print(f'Learner {m + 1}: {learn.name}')
                # training and testing MetaLearner.
                learner = learn
                learner.fit(temp_x_train, temp_y_train, temp_w_train)
                predictions = learner.predict(temp_x_test)
                temp_mse = ((predictions - temp_tau_test) ** 2).mean()
                # append mse of specific metalearner to 'mses'.
                mses[0, m] = temp_mse
                # update index
                m += 1
            # append 'mses' to 'size_mse'.
            run_mse = np.append(run_mse, mses, axis=0)
            # update index
            s += 1
            # append 'size_mse' to 'setup_mse'.
        setup_mse = np.append(setup_mse, run_mse, axis=0)
    # append to results
    results[0] = setup_mse[:, 0:8]  # random forest
    results[1] = setup_mse[:, 8:16]  # lasso
    results[2] = setup_mse[:, 16:24]  # neural network


def parser_arguments():
    p = configargparse.ArgParser()
    p.add('--setting', required=True, type=int, help='Type in which setting you would like to run.')
    return p.parse_args()


def main():
    argument = parser_arguments()
    # run experiment for one setting (setting = run_setting + 1)
    print(argument.setting)
    # run_setting = 0
    run_experiment(argument.setting-1)
    # save results
    results_file_name = f"results_setup_{argument.setting}.json"
    r = open(results_file_name, 'w')
    json_obj = jsonpickle.encode(results)
    r.write(json_obj)
    r.close()


if __name__ == "__main__":
    main()
