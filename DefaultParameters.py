''''''''' GENERAL '''''''''

# cross fitting
CF_FOLDS: int = 2

# seeds
KERAS_SEED: int = 8953

# epsilon for stable results
EPSILON: float = 1e-5

# data generation
TEST_SIZE: int = 1000
TRAIN_SIZES = [500, 1000, 2000, 5000]
FEATURE_DIMENSION: int = 20

''''''' NEURAL NETWORKS '''''''

# ARCHITECTURE
BATCH_SIZE: int = 100
N_LAYERS_FIRST_PART: int = 3
N_LAYERS_SECOND_PART: int = 2
N_UNITS_FIRST_PART: int = 200
N_UNITS_SECOND_PART: int = 100
NON_LINEARITY = 'relu'
REGULARIZER = 0  # TODO: add

# EARLY STOPPING
PATIENCE: int = 10
START_FROM: int = 200
VALIDATION_SPLIT: float = 0.3

# OPTIMIZER
N_EPOCHS: int = 400
LEARNING_RATE: float = 1e-4

# INPUT/OUTPUT
INPUT_SIZE: int = 25
OUTPUT_SIZE: int = 1

''''''' RANDOM FORESTS '''''''
# https://github.com/soerenkuenzel/forestry/blob/master/R/forestry.R
N_TREES: int = 1000
MAX_DEPTH: int = 99
RF_RANDOM_STATE: int = 2023
MAX_FEATURES: float = 0.3  # justified in: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


''''''' LASSO '''''''
LASSO_RANDOM_STATE: int = 2023
K_FOLDS: int = 10
MAX_ITER: int = 1000000  # LOOK WHICH NUMBER
TOLERANCE: float = 1
DEGREE_POLYNOMIALS: int = 3
