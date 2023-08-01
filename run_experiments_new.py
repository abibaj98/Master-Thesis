# import from files
from Meta import *
from simulation_data import *
# import packages
import numpy as np
import jsonpickle
import configargparse
import time
# import r packages (only needed to generate the cov matrix of X)
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

rpackages.importr("mpower")
mpower = robjects.packages.importr("mpower")
set_seed = robjects.r('set.seed')

# seeds for reproducibility
set_seed(2023)  # R seed
np.random.seed(2023)  # numpy seed
tf.keras.utils.set_random_seed(2023)  # keras seed

# set dtype standard
tf.keras.backend.set_floatx('float64')

# some parameters  # TODO: add to default_paramters
sample_sizes = SAMPLE_SIZES  # default sample sizes
test_size = TEST_SIZE  # default test size
n_runs = N_RUNS  # default number of runs
d = DIMENSION  # default dimension of X

# All MetaLearners in combination with each Baselearner (Random Forest, Linear Models, Neural Network)
learners = [TLearner('rf'), SLearner('rf'), XLearner('rf'), RLearner('rf'), DRLearner('rf'), RALearner('rf'),
            PWLearner('rf'), ULearner('rf'),
            TLearner('lasso'), SLearner('lasso'), XLearner('lasso'), RLearner('lasso'), DRLearner('lasso'),
            RALearner('lasso'), PWLearner('lasso'), ULearner('lasso'),
            TLearner('nn'), SLearner('nn'), XLearner('nn'), RLearner('nn'), DRLearner('nn'), RALearner('nn'),
            PWLearner('nn'), ULearner('nn')]

# empty results list
results = [[] for _ in range(3)]  # for the 3 base learners

dim = DIMENSION + 3  # dimension of X + 3 (y, w, tau)


# helper function to get y, x, w, tau, out of data
def get_variables(dataset):
    y = dataset[:, 0]
    x = dataset[:, 1:(dim - 2)]
    w = dataset[:, (dim - 2)]
    tau = dataset[:, (dim - 1)]
    return y, x, w, tau


def run_experiment(setting, n_runs):
    print("------------------------------------------")
    print("Running the experiment on synthetic data")
    print("------------------------------------------")
    # array; all mses for the setup
    setup_mse = np.empty(shape=(0, 24))
    # mean of X
    mean_x = np.zeros(d)
    for r in range(n_runs):
        print(f'Run: {r + 1}')
        # array; all mses for one specific run.
        run_mse = np.empty(shape=(0, 24))
        # cov_x, betas, beta_0 and betas_1 generated once per run
        cov_x = np.array(mpower.cvine(d=d, alpha=0.5, beta=0.5))
        betas_run = random.uniform(low=-1, high=1, size=d)
        betas_0_run = random.uniform(low=-0.5, high=0.5, size=d)
        betas_1_run = random.uniform(low=-0.5, high=0.5, size=d)
        for s, size in enumerate(sample_sizes):
            print(f'Sample Size: {s + 1}: {size}')
            # array; all mses for one specific run and sample size
            mses = np.empty(shape=(1, 24))
            # get data for specific setup, run and sample size.
            train = generate_data(mean=mean_x, cov=cov_x, ex=exs[setting], cate=cates[setting], sample_size=size,
                                  betas=betas_run,
                                  betas_0=betas_0_run, betas_1=betas_1_run)
            test = generate_data(mean=mean_x, cov=cov_x, ex=exs[setting], cate=cates[setting], sample_size=size,
                                 betas=betas_run,
                                 betas_0=betas_0_run, betas_1=betas_1_run)
            temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables(dataset=train)
            temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables(dataset=test)
            for m, learn in enumerate(learners):
                start = time.time()
                print(f'Learner {m + 1}: {learn.name}')
                # training and get predictions
                learner = learn
                learner.fit(temp_x_train, temp_y_train, temp_w_train)
                predictions = learner.predict(temp_x_test)
                # mean squared error
                temp_mse = ((predictions - temp_tau_test) ** 2).mean()
                # append mse of specific metalearner to 'mses'.
                mses[0, m] = temp_mse
                end = time.time()
                print(end - start)
            # append 'mses' to 'size_mse'.
            run_mse = np.append(run_mse, mses, axis=0)
            # append 'size_mse' to 'setup_mse'.
        setup_mse = np.append(setup_mse, run_mse, axis=0)
        print("--------")
    # append to results
    results[0] = setup_mse[:, 0:8]  # random forest
    results[1] = setup_mse[:, 8:16]  # lasso
    results[2] = setup_mse[:, 16:24]  # neural network
    print("Done")
    print('-------------------------')


def parser_arguments():
    p = configargparse.ArgParser()
    p.add_argument('--setting', type=int, default=1, choices=range(1, 19),
                   help='Type in which setting you would like to run. There exist settings 1-18.')
    p.add_argument('--runs', type=int, default=n_runs,
                   help='Type in how many runs you would like to run. Should be a positive integer.')
    return p.parse_args()


def main():
    argument = parser_arguments()
    print('-------------------------')
    print(f'Setting chosen: {argument.setting}')
    print(f'Number of Runs chosen: {argument.runs}')
    # run experiment for one setting (setting = run_setting + 1)
    run_experiment(argument.setting - 1, argument.runs)
    # save results
    results_file_name = f"results_setup_{argument.setting}_large.json"  # TODO: change!!!
    r = open(results_file_name, 'w')
    json_obj = jsonpickle.encode(results)
    r.write(json_obj)
    r.close()


if __name__ == "__main__":
    main()
