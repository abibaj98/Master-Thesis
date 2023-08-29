"""'''' Default Parameters ''''"""

''''''' Generals  '''''''

# seeds
KERAS_SEED: int = 2023  # keras seed
R_SEED: int = 2023  # r seed
NP_SEED: int = 2023  # numpy seed
RANDOM: int = 2023  # random_state for random forest and linear models

# Clip values for the predicted propensities
MIN_CLIP: float = 0.05  # TODO: changed was 0.01
MAX_CLIP: float = 0.95  # TODO: changed was 0.99

# data generation
TEST_SIZE: int = 1000
SAMPLE_SIZES: list = [500, 1000, 2000, 5000]
N_RUNS: int = 10
DIMENSION: int = 20  # Dimension of X
N_SETUPS: int = 24  # the 18 different settings, TODO:  NOT NEEDED!

''''''' NEURAL NETWORKS '''''''

# ARCHITECTURE
NON_LINEARITY: str = 'elu'

# EARLY STOPPING
PATIENCE: int = 5  # TODO: changed was 10 when tried.
START_FROM: int = 50  # TODO: changed was 200 when tried
VALIDATION_SPLIT: float = 0.3

# OPTIMIZER
BATCH_SIZE: int = 100
WEIGHT_DECAY: float = 1e-3  # TODO: changed, was 1e-4, when tried
N_EPOCHS: int = 100  # TODO: change, was 500
LEARNING_RATE: float = 1e-4
LABEL_SMOOTHING: float = 0.0  # TODO: changed, was 0.1

''''''' RANDOM FORESTS '''''''
N_TREES: int = 1000
MAX_DEPTH: int = 7
MAX_FEATURES: float = 0.33
# justified in: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


''''''' LASSO '''''''
K_FOLDS: int = 10
MAX_ITER: int = 1000000  # LOOK WHICH NUMBER
TOLERANCE: float = 1.0
DEGREE_POLYNOMIALS: int = 3
