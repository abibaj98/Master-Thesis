"""'''' Default Parameters ''''"""

''''''' Generals  '''''''

# Seeds
KERAS_SEED: int = 2023  # keras seed
R_SEED: int = 2023  # r seed
NP_SEED: int = 2023  # numpy seed
RANDOM = None  # not needed (random_state for RandomForest and Lasso)

# Clip values for the predicted propensities
MIN_CLIP: float = 0.05
MAX_CLIP: float = 0.95

# Data generation
TEST_SIZE: int = 1000
SAMPLE_SIZES: list = [500, 1000, 2000, 5000]
N_RUNS: int = 10
DIMENSION: int = 20  # dimension of X

''''''' NEURAL NETWORKS '''''''

# Architecture
NON_LINEARITY: str = 'elu'
DROP_OUT: float = 0.3  # dropout rate

# Early Stopping, not used since Callbacks = None (in neural_networks.py)
VALIDATION_SPLIT: float = 0.0  # i.e., not used

# Optimizer
BATCH_SIZE: int = 100
WEIGHT_DECAY: float = 1e-4
N_EPOCHS: int = 100
LEARNING_RATE: float = 1e-3
LABEL_SMOOTHING: float = 0.0  # i.e., not used

''''''' RANDOM FORESTS '''''''
N_TREES: int = 1000
MAX_DEPTH: int = 7
MAX_FEATURES: float = 0.33

''''''' LASSO-BASED REGRESSION '''''''
K_FOLDS: int = 10
MAX_ITER: int = 1000000
TOLERANCE: float = 1.0
DEGREE_POLYNOMIALS: int = 3
