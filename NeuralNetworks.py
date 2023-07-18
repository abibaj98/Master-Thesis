import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, Sequential
from DefaultParameters import *
from keras.src.callbacks import EarlyStopping

# float64 as standard
tf.keras.backend.set_floatx('float64')

# sequential neural network
nn_sequential: Sequential = keras.Sequential([
    keras.Input(shape=(FEATURE_DIMENSION,)),
    layers.Dense(units=200, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer1"),
    layers.Dense(units=200, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer2"),
    layers.Dense(units=200, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer3"),
    layers.Dense(units=100, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer4"),
    layers.Dense(units=100, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer5"),
    layers.Dense(units=1, activation="linear", name="layer6"),
], name="nn_sequential")

# only for the S-Learner
nn_sequential_1: Sequential = keras.Sequential([
    keras.Input(shape=(FEATURE_DIMENSION_1,)),
    layers.Dense(units=200, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer1"),
    layers.Dense(units=200, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer2"),
    layers.Dense(units=200, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer3"),
    layers.Dense(units=100, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer4"),
    layers.Dense(units=100, activation="relu", kernel_regularizer=regularizers.L2(PENALTY), name="layer5"),
    layers.Dense(units=1, activation="linear", name="layer6"),

], name="nn_sequential_1")

# early stopping setting
CALLBACK: EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                           start_from_epoch=START_FROM)
