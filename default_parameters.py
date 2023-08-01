"""'''''' GENERAL ''''''"""

# cross fitting folds, 1 means no cross-fitting
CF_FOLDS: int = 1

# seeds
KERAS_SEED: int = 2023

# small epsilon added to not divide by zero
EPSILON: float = 1e-6

# data generation
TEST_SIZE: int = 1000
TRAIN_SIZES = [1000, 2000, 5000, 10000]
N_RUNS = 10
DIMENSION: int = 20  # Dimension of X
N_SETUPS: int = 18  # the 18 different settings

''''''' NEURAL NETWORKS '''''''

# ARCHITECTURE
BATCH_SIZE: int = 100
NON_LINEARITY: str = 'elu'
PENALTY: float = 1e-3  # not needed
WEIGHT_DECAY: float = 1e-3

# EARLY STOPPING
PATIENCE: int = 5
START_FROM: int = 50
VALIDATION_SPLIT: float = 0.3

# OPTIMIZER
N_EPOCHS: int = 500
LEARNING_RATE: float = 1e-4
LABEL_SMOOTHING: float = 0.1

''''''' RANDOM FORESTS '''''''
N_TREES: int = 1000
MAX_DEPTH: int = 7
MAX_FEATURES: float = 0.33
# justified in: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


''''''' LASSO '''''''
K_FOLDS: int = 10
MAX_ITER: int = 1000000  # LOOK WHICH NUMBER
TOLERANCE: float = 1
DEGREE_POLYNOMIALS: int = 3

""" RANDOM STATE """
RANDOM: int = 2023
