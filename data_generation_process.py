# from files
from default_parameters import *
# import packages
import numpy as np
from numpy import random
from scipy import stats
# import r packages (only needed to generate the cov matrix of X)
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

rpackages.importr("mpower")
mpower = robjects.packages.importr("mpower")

#############################
# List of all settings 1-24 #
#############################

# all settings of e_x (propensity)
exs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # for settings 1-6 (constant ex = 0.5)
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # for settings 7-12 (constant ex = 0.1)
       'beta_confounded', 'beta_confounded', 'beta_confounded', 'beta_confounded', 'beta_confounded',
       'beta_confounded',  # for settings 13-18 (beta confounded, balanced)
       'step_confounded', 'step_confounded', 'step_confounded', 'step_confounded', 'step_confounded',
       'step_confounded'  # for settings 19-24 (step func confounded, unbalanced)
       ]

# all settings for the cate tau (each setting three times in combination with the propensity settings)
cates = ['linear_response', 'non_linear_response', 'indicator_cate', 'linear_cate', 'complex_linear_cate',
         'complex_non_linear_cate'] * 4


###############################################
# helper functions for the different settings #
###############################################

# Response Surface Settings (mu_0, mu_1, tau)
# i): zero cate (linear response surface)
def linear_response(x, betas):
    # train set
    mu_0 = np.matmul(x, betas) + 5 * x[:, 0]
    mu_1 = mu_0
    tau = np.zeros(len(x))

    return mu_0, mu_1, tau


# ii): zero cate (non-linear response surface)
def non_linear_response(x):
    # train set
    mu_0 = 5 * np.arctan(x[:, 0]) * np.arctan(x[:, 1])
    mu_1 = mu_0
    tau = np.zeros(len(x))

    return mu_0, mu_1, tau


# iii): simple indicator cate
def simple_indicator_cate(x, betas):
    # train set
    mu_0 = np.matmul(x, betas) + 5 * np.int8(x[:, 0] > 0.5)
    mu_1 = mu_0 + 8 * np.int8(x[:, 1] > 0.1)
    tau = mu_1 - mu_0

    return mu_0, mu_1, tau


# iv): simple linear cate
def simple_linear_cate(x, betas):
    # train set
    mu_0 = np.matmul(x, betas) + 5 * x[:, 0]
    mu_1 = mu_0 + 4 * x[:, 1] + 2  # TODO: what to change?
    tau = mu_1 - mu_0

    return mu_0, mu_1, tau


# v): complex linear cate
def complex_linear_cate(x, betas_0, betas_1):
    # train set
    mu_0 = np.matmul(x, betas_0) + 5 * x[:, 0]
    mu_1 = np.matmul(x, betas_1) + 5 * x[:, 0]
    tau = mu_1 - mu_0

    return mu_0, mu_1, tau


# vi): complex non-linear cate
# helper function varsigma
def varsigma_function(x):
    return 2 / (1 + np.exp(-12 * x))  # TODO: before, it was (x - 0,5).


def complex_non_linear_cate(x):
    # train set
    mu_0 = -4 / 2 * varsigma_function(x[:, 0]) * varsigma_function(x[:, 1])
    mu_1 = 4 / 2 * varsigma_function(x[:, 0]) * varsigma_function(x[:, 1])
    tau = mu_1 - mu_0

    return mu_0, mu_1, tau


# Beta confounded propensity (balanced)
def beta_confounded(x):
    beta_dist = stats.beta(a=2, b=4)  # set beta distribution

    # train
    cdf_values = stats.norm.cdf(x[:, 0])
    beta_values = beta_dist.pdf(cdf_values)  # calculate pdf values for x1
    e_x = 1 / 4 * (1 + beta_values)

    return e_x


# Step function confounded propensity (unbalanced)
def step_function_confounded(x):
    x = x[:, 0]
    e_x = np.where(x <= 1, 0.01, 0.6)

    return e_x


#####################################
# helper functions to generate data #
#####################################


# generate one dataset
def generate_data(mean, cov, ex, cate, sample_size, betas, betas_0, betas_1):
    # 1: generated x
    x = random.multivariate_normal(mean=mean, cov=cov, size=sample_size, check_valid='warn')

    # 2: generate e_0 & e_1
    e_0 = random.normal(loc=0.0, scale=1.0, size=sample_size)
    e_1 = random.normal(loc=0.0, scale=1.0, size=sample_size)

    # 3: compute mu_0 & mu_1 --> based on setting
    if cate == 'linear_response':  # 'linear response' setting (no treatment effect)
        mu_0, mu_1, tau = linear_response(x, betas)

    elif cate == 'non_linear_response':  # 'non-linear response' setting (no treatment effect)
        mu_0, mu_1, tau = non_linear_response(x)

    elif cate == 'indicator_cate':  # 'simple indicator cate' setting
        mu_0, mu_1, tau = simple_indicator_cate(x, betas)

    elif cate == 'linear_cate':  # 'simple linear cate' setting
        mu_0, mu_1, tau = simple_linear_cate(x, betas)

    elif cate == 'complex_linear_cate':  # 'complex linear cate' setting
        mu_0, mu_1, tau = complex_linear_cate(x, betas_0, betas_1)

    elif cate == 'complex_non_linear_cate':  # 'complex non-linear cate' setting
        mu_0, mu_1, tau = complex_non_linear_cate(x)

    else:
        raise NotImplementedError('No or incorrect setting specified.')

    # 4: create potential outcomes y_0 & y_1
    y_0 = mu_0 + e_0
    y_1 = mu_1 + e_1

    # 5: Set propensity score e_x --> based on setting
    if isinstance(ex, float or int):
        e_x = ex

    elif ex == 'beta_confounded':
        e_x = beta_confounded(x)

    elif ex == 'step_confounded':
        e_x = step_function_confounded(x)

    else:
        raise NotImplementedError('Propensity method not or incorrectly specified.')

    # 6: Generate treatment assignment W
    w = random.binomial(size=sample_size, n=1, p=e_x)

    # 7: Create observed variable Y
    y = np.multiply(w, y_1) + np.multiply(np.ones(sample_size) - w, y_0)

    # 8: Create dataset --> columns: [y, X, w, tau]
    dataset = np.concatenate(
        (np.reshape(y, (sample_size, 1)), x, np.reshape(w, (sample_size, 1)), np.reshape(tau, (sample_size, 1))),
        axis=1)

    return dataset


#####################################################
# helper functions to return variables y, X, w, tau #
#####################################################
def get_variables(dataset):
    y = dataset[:, 0]
    x = dataset[:, 1:(DIMENSION + 3 - 2)]
    w = dataset[:, (DIMENSION + 3 - 2)]
    tau = dataset[:, (DIMENSION + 3 - 1)]
    return y, x, w, tau


# same for the ihdp dataset
def get_variables_ihdp(dataset, run):
    y = dataset[run][:, 0]
    x = dataset[run][:, 1:23]
    w = dataset[run][:, 23]
    tau = dataset[run][:, 24]
    return y, x, w, tau
