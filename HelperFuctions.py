from DefaultParameters import *
import tensorflow as tf
import keras

"""
def r_u_pseudo_outcomes(y, w, probs, mu, eps):
    return (y - mu) / (w - probs + eps)


def dr_pseudo_outcomes(y, w, probs, mu0, mu1, eps):
    mu_w = w * mu1 + (1 - w) * mu0
    return (w - probs) / (probs * (1 - probs) + eps) * (y - mu_w) + mu1 - mu0


def ra_pseudo_outcomes(y, w, mu0, mu1):
    return w * (y - mu0) + (1 - w) * (mu1 - y)


def pw_pseudo_outcomes(y, w, probs, eps):
    return (w / (probs + eps) - (1 - w) / ((1 - probs) + eps)) * y
"""


def clone_nn_regression(model):
    cloned_model = tf.keras.models.clone_model(model)
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


def clone_nn_classification(model):
    cloned_model = tf.keras.models.clone_model(model)
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
