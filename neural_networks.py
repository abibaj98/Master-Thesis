# import packages
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Dropout
# import from files
from default_values import *

# Sequential Model Architecture
NN_SEQUENTIAL: Sequential = Sequential([
    Dense(units=200, activation=NON_LINEARITY, name="layer1"),
    Dropout(rate=DROP_OUT),
    Dense(units=200, activation=NON_LINEARITY, name="layer2"),
    Dropout(rate=DROP_OUT),
    Dense(units=200, activation=NON_LINEARITY, name="layer3"),
    Dropout(rate=DROP_OUT),
    Dense(units=100, activation=NON_LINEARITY, name="layer4"),
    Dropout(rate=DROP_OUT),
    Dense(units=100, activation=NON_LINEARITY, name="layer5"),
    Dropout(rate=DROP_OUT),
    Dense(units=1, activation="linear", name="layer6"),
], name="NN_SEQUENTIAL")

# early stopping setting, not used
CALLBACK = None


# function to compile neural network for a regression task
def clone_nn_regression(model):
    # clone model
    cloned_model = tf.keras.models.clone_model(model)
    # compile model
    cloned_model.compile(
        # optimizer
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        # loss function
        loss=keras.losses.MeanSquaredError(),
        # list of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError()],
        # weighted metrics (for weighted regressions)
        weighted_metrics=[]
    )
    return cloned_model


# function to compile neural network for a classification task
def clone_nn_classification(model):
    # clone model
    cloned_model = tf.keras.models.clone_model(model)
    # compile model
    cloned_model.compile(
        # optimizer
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        # loss function
        loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=LABEL_SMOOTHING),
        # list of metrics to monitor
        metrics=keras.metrics.BinaryAccuracy(),
        # weighted metrics (for weighted regressions)
        weighted_metrics=[]
    )
    return cloned_model
