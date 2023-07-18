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
FEATURE_DIMENSION_1: int = FEATURE_DIMENSION + 1

''''''' NEURAL NETWORKS '''''''

# ARCHITECTURE
BATCH_SIZE: int = 100
N_LAYERS_FIRST_PART: int = 3
N_LAYERS_SECOND_PART: int = 2
N_UNITS_FIRST_PART: int = 200
N_UNITS_SECOND_PART: int = 100
NON_LINEARITY: str = 'relu'
PENALTY: float = 1e-4

# EARLY STOPPING
PATIENCE: int = 10
START_FROM: int = 50  # TODO: check if this makes it better?
VALIDATION_SPLIT: float = 0.3

# OPTIMIZER
N_EPOCHS: int = 10000
LEARNING_RATE: float = 1e-4
LABEL_SMOOTHING: float = 0.1

# INPUT/OUTPUT
INPUT_SIZE: int = 25
OUTPUT_SIZE: int = 1

''''''' RANDOM FORESTS '''''''
# https://github.com/soerenkuenzel/forestry/blob/master/R/forestry.R
N_TREES: int = 1000
MAX_DEPTH: int = 99
MAX_FEATURES: float = 0.3
# justified in: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


''''''' LASSO '''''''
K_FOLDS: int = 10
MAX_ITER: int = 1000000  # LOOK WHICH NUMBER
TOLERANCE: float = 1
DEGREE_POLYNOMIALS: int = 3

""" RANDOM STATE """
RANDOM: int = 2023


