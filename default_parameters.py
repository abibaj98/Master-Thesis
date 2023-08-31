"""'''' Default Parameters ''''"""

''''''' Generals  '''''''

# seeds
KERAS_SEED: int = 2023  # keras seed
R_SEED: int = 2023  # r seed
NP_SEED: int = 2023  # numpy seed
RANDOM: None  # random_state for random forest and linear models # TODO: change?

# Clip values for the predicted propensities
MIN_CLIP: float = 0.05
MAX_CLIP: float = 0.95

# data generation
TEST_SIZE: int = 1000
SAMPLE_SIZES: list = [500, 1000, 2000, 5000]
N_RUNS: int = 10
DIMENSION: int = 20  # Dimension of X

''''''' NEURAL NETWORKS '''''''

# ARCHITECTURE
NON_LINEARITY: str = 'elu'

# EARLY STOPPING
PATIENCE: int = 5  # changed was 10 when tried.
START_FROM: int = 50
VALIDATION_SPLIT: float = 0.1  # TODO: THIS!!!

# OPTIMIZER
BATCH_SIZE: int = 100
WEIGHT_DECAY: float = 1e-3  # changed, was 1e-4, when tried
N_EPOCHS: int = 100
LEARNING_RATE: float = 1e-4
LABEL_SMOOTHING: float = 0.0  # 0 means no smoothing, changed, was 0.1

''''''' RANDOM FORESTS '''''''
N_TREES: int = 1000
MAX_DEPTH: int = 7
MAX_FEATURES: float = 0.33


''''''' LASSO '''''''
K_FOLDS: int = 10
MAX_ITER: int = 1000000  # LOOK WHICH NUMBER
TOLERANCE: float = 1.0
DEGREE_POLYNOMIALS: int = 3
