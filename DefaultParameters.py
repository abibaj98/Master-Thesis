import numpy as np

''''''''' GENERAL '''''''''

# epsilon for stable results
EPSILON = 1e-7

# data generation
TEST_SIZE = 1000
TRAIN_SIZES = [500, 1000, 2000, 5000]
FEATURE_DIMENSION = 20

''''''' NEURAL NETWORKS '''''''

# ARCHITECTURE
BATCH_SIZE = 100
N_UNITS_FIRST_PART = 200
N_UNITS_SECOND_PART = 100
NON_LINEARITY = 'elu'

# EARLY STOPPING
PATIENCE = 10
START_FROM = 200
VALIDATION_SPLIT = 0.3

# OPTIMIZER
N_EPOCHS = 400
LEARNING_RATE = 0.0001

# INPUT/OUTPUT
INPUT_SIZE = 25
OUTPUT_SIZE = 1

''''''' RANDOM FORESTS '''''''
# https://github.com/soerenkuenzel/forestry/blob/master/R/forestry.R
N_TREES = 2000
MAX_DEPTH = 99
RF_RANDOM_STATE = 2023
MAX_FEATURES = 1.0  # justified in: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html



''''''' LASSO '''''''
LASSO_RANDOM_STATE = 2023
K_FOLDS = 10
MAX_ITER = 100000  # LOOK WHICH NUMBER
TOLERANCE = 1e-2
DEGREE_POLYNOMIALS = 3

