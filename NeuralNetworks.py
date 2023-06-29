import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import load_model


''''' Basic Architecture ''''

# 3 layers with 200 units (elu activation), 2 layers with 100 units (elu activations), 1 output layer (linear
# activation)
model_25 = keras.Sequential([
    keras.Input(shape=(25,)),
    layers.Dense(units=200, activation="relu", name="layer1"),
    layers.Dense(units=200, activation="relu", name="layer2"),
    layers.Dense(units=200, activation="relu", name="layer3"),
    layers.Dense(units=100, activation="relu", name="layer4"),
    layers.Dense(units=100, activation="relu", name="layer5"),
    layers.Dense(units=1, activation="linear", name="layer6"),

], name="Dense_Neural_Network")

# compile the model
model_25.compile(
    optimizer=keras.optimizers.legacy.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # List of metrics to monitor
    metrics=[keras.metrics.MeanSquaredError()],
)

# save the model
model_25.save('model_25')


''''''' For T Learner Only '''''''

# 3 layers with 200 units (elu activation), 2 layers with 100 units (elu activations), 1 output layer (linear activation)
model_26 = keras.Sequential([
    keras.Input(shape=(26,)),
    layers.Dense(units=200, activation="relu", name="layer1"),
    layers.Dense(units=200, activation="relu", name="layer2"),
    layers.Dense(units=200, activation="relu", name="layer3"),
    layers.Dense(units=100, activation="relu", name="layer4"),
    layers.Dense(units=100, activation="relu", name="layer5"),
    layers.Dense(units=1, activation="linear", name="layer6"),

], name="Dense_Neural_Network")

# compile the model
model_26.compile(
    optimizer=keras.optimizers.legacy.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # List of metrics to monitor
    metrics=[keras.metrics.MeanSquaredError()],
)

# save the model
model_26.save('model_26')


''''''' For propensity score '''''''

model_ex = keras.Sequential([
    keras.Input(shape=(25,)),
    layers.Dense(units=200, activation="relu", name="layer1"),
    layers.Dense(units=200, activation="relu", name="layer2"),
    layers.Dense(units=200, activation="relu", name="layer3"),
    layers.Dense(units=100, activation="relu", name="layer4"),
    layers.Dense(units=100, activation="relu", name="layer5"),
    layers.Dense(units=1, activation="linear", name="layer6"),

], name="Dense_Neural_Network_Classification")

# compile the model
model_ex.compile(
    optimizer=keras.optimizers.legacy.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.5),
    # List of metrics to monitor
    metrics=keras.metrics.BinaryAccuracy(),
)

# save the model
model_ex.save('model_ex')


''''''' Callbacks '''''''
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=200)
