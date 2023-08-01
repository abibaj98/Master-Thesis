# import packages
import tensorflow as tf
import keras
from keras import layers, Sequential
from keras.callbacks import EarlyStopping
# import from files
from default_parameters import *

# Sequential Model Architecture
NN_SEQUENTIAL: Sequential = keras.Sequential([
    layers.Dense(units=200, activation=NON_LINEARITY, name="layer1"),
    layers.Dense(units=200, activation=NON_LINEARITY, name="layer2"),
    layers.Dense(units=200, activation=NON_LINEARITY, name="layer3"),
    layers.Dense(units=100, activation=NON_LINEARITY, name="layer4"),
    layers.Dense(units=100, activation=NON_LINEARITY, name="layer5"),
    layers.Dense(units=1, activation="linear", name="layer6"),
], name="NN_SEQUENTIAL")

# early stopping setting
CALLBACK: EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                        start_from_epoch=START_FROM)


# function to compile neural network for a regression task
def clone_nn_regression(model):
    # clone model
    cloned_model = tf.keras.models.clone_model(model)
    # compile model
    cloned_model.compile(
        # optimizer
        optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),  # TODO: change to decay
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
        optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        # loss function
        loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=LABEL_SMOOTHING),
        # list of metrics to monitor
        metrics=keras.metrics.BinaryAccuracy(),
        # weighted metrics (for weighted regressions)
        weighted_metrics=[]
    )
    return cloned_model
