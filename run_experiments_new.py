# import from files
from meta_learner import *
from data_generation_process import *

# import packages
import numpy as np
import jsonpickle
import configargparse
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

# import r packages (only needed to generate the cov matrix of X)
rpackages.importr("mpower")
mpower = robjects.packages.importr("mpower")
set_seed = robjects.r('set.seed')

# seeds for reproducibility
set_seed(R_SEED)  # R seed
np.random.seed(NP_SEED)  # numpy seed
tf.keras.utils.set_random_seed(KERAS_SEED)  # keras seed

# set dtype standard
tf.keras.backend.set_floatx('float64')

##################
# Default Values #
##################
sample_sizes = SAMPLE_SIZES  # default sample sizes
test_size = TEST_SIZE  # default test size
n_runs = N_RUNS  # default number of runs
d = DIMENSION  # default dimension of X

####################
# All Metalearners # (In combination with each Baselearner (Random Forest, Linear Models, Neural Network))
####################


learners = [TLearner('nn'), SLearner('nn'), XLearner('nn'), RLearner('nn'), DRLearner('nn'), RALearner('nn')]

"""
learners = [TLearner('rf'), SLearner('rf'), XLearner('rf'), RLearner('rf'), DRLearner('rf'), RALearner('rf'),
            PWLearner('rf'), ULearner('rf'),
            TLearner('lasso'), SLearner('lasso'), XLearner('lasso'), RLearner('lasso'), DRLearner('lasso'),
            RALearner('lasso'), PWLearner('lasso'), ULearner('lasso'),
            TLearner('nn'), SLearner('nn'), XLearner('nn'), RLearner('nn'), DRLearner('nn'), RALearner('nn'),
            PWLearner('nn'), ULearner('nn')]
"""

############################################################
# Function which runs the experiment on the simulated data #
############################################################
def run_experiment(setting, runs, results, learners):
    print("------------------------------------------")
    print(f'Setting chosen: {setting + 1}')
    print(f'Number of Runs chosen: {runs}')
    print("------------------------------------------")
    print("Running the Experiment on Simulated Data")
    print("------------------------------------------")
    # array; all mses for the setup
    setup_mse = np.empty(shape=(0, 24))
    # same all the time
    mean_x = np.zeros(d)
    for r in range(runs):
        print(f'Run: {r + 1}')
        # array; all mses for one specific run.
        run_mse = np.empty(shape=(0, 24))
        # cov_x, betas, beta_0 and betas_1 generated once per run
        cov_x = np.array(mpower.cvine(d=d, alpha=5, beta=5))
        betas_run = random.uniform(low=-1, high=1, size=d)
        betas_0_run = random.uniform(low=-0.5, high=0.5, size=d)
        betas_1_run = random.uniform(low=-0.5, high=0.5, size=d)
        for s, size in enumerate(sample_sizes):
            print(f'Sample Size #{s + 1}: {size}')
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
                keras.backend.clear_session()  # clear keras session to lower memory consumption
                print(f'Learner {m + 1}: {learn.name}')
                # training and get predictions
                learner = learn
                learner.fit(temp_x_train, temp_y_train, temp_w_train)
                predictions = learner.predict(temp_x_test)
                # mean squared error
                temp_mse = ((predictions - temp_tau_test) ** 2).mean()
                # append mse of specific metalearner to 'mses'.
                mses[0, m] = temp_mse
            # append 'mses' to 'size_mse'.
            run_mse = np.append(run_mse, mses, axis=0)
            print("----------------")
            # append 'size_mse' to 'setup_mse'.
        setup_mse = np.append(setup_mse, run_mse, axis=0)
        print("-----------------------")
    # append to results
    """
    results[0] = setup_mse[:, 0:8]  # random forest
    results[1] = setup_mse[:, 8:16]  # lasso
    results[2] = setup_mse[:, 16:24]  # neural network
    """
    results[0] = np.concatenate((setup_mse[:, 0:8], np.tile(sample_sizes, reps=runs)[:, np.newaxis]), axis=1)
    results[1] = np.concatenate((setup_mse[:, 8:16], np.tile(sample_sizes, reps=runs)[:, np.newaxis]), axis=1)
    results[2] = np.concatenate((setup_mse[:, 16:24], np.tile(sample_sizes, reps=runs)[:, np.newaxis]), axis=1)
    print("Done")
    print("------------------------------------------")


##########################################################
# Function which runs the experiment on the ihdp dataset #
##########################################################


# one setup
def run_ihdp(runs, results):
    train_ihdp = np.load('ihdp_train_processed.npy')
    test_ihdp = np.load('ihdp_test_processed.npy')
    print("------------------------------------------")
    print(f'Number of Runs chosen: {runs}')
    print("------------------------------------------")
    print("Running the Experiment on the IHDP Dataset")
    print("------------------------------------------")
    all_mse = np.empty(shape=(0, 24))
    for run in range(runs):
        print(f'Run: {run + 1}')
        # mses for one run
        mses = np.empty(shape=(1, 24))
        # get data for specific setup, samplesize and run.
        temp_y_train, temp_x_train, temp_w_train, temp_tau_train = get_variables_ihdp(dataset=train_ihdp, run=run)
        temp_y_test, temp_x_test, temp_w_test, temp_tau_test = get_variables_ihdp(dataset=test_ihdp, run=run)
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
    # append to results
    results[0] = all_mse[:, 0:8]  # random forest
    results[1] = all_mse[:, 8:16]  # lm
    results[2] = all_mse[:, 16:24]  # neural network
    print("Done")
    print('-------------------------')


#########################################################################################
# ArgParser: can choose setting and number of runs when running the file from the shell #
#########################################################################################

# helper function
def check(number):
    number = int(number)
    if number <= 0:
        raise configargparse.ArgumentTypeError("The number has to be positive")
    return number


# define options
def parser_arguments():
    p = configargparse.ArgParser()
    p.add_argument('--setting', type=int, default=1, choices=range(1, 25),
                   help='Type in which setting you would like to run. There exist settings 1-24.')
    p.add_argument('--runs', type=check, default=n_runs,
                   help='Type in how many runs you would like to run. Should be a positive integer.')
    p.add_argument('--runs_ihdp', type=int, default=100, choices=range(1, 101),
                   help='Type in how many runs you would like to run. Can be from 1 to 100.')
    p.add_argument('--data', type=str, default="simulated", choices=['simulated', 'ihdp'],
                   help='Select Data to run experiment on. Can be "simulated" or "ihdp".')
    return p.parse_args()


#######################################
# main function: what is actually run #
#######################################

def main():
    # arguments from the ArgParser
    argument = parser_arguments()
    # results
    results = [[] for _ in range(3)]  # for the 3 base learners

    if argument.data == "simulated":
        # run experiment for one setting
        run_experiment(setting=argument.setting - 1, runs=argument.runs, results=results, learners=learners)
        # results json name
        results_file_name = f'results_simulated_setting{argument.setting}_{argument.runs}run(s)_3sept.json'  # TODO: change!!!

    elif argument.data == "ihdp":
        # run experiment
        run_ihdp(runs=argument.runs_ihdp, results=results)
        # results json name
        results_file_name = f'results_ihdp_{argument.runs_ihdp}run(s)_3sept.json'

    else:
        raise NotImplementedError

    # save as json file
    r = open(results_file_name, 'w')
    json_obj = jsonpickle.encode(results)
    r.write(json_obj)
    r.close()


if __name__ == "__main__":
    main()
