import numpy as np
import tensorflow as tf
from time import sleep

input = np.concatenate((np.array([0, 1, 2, 3, 4, 5, 6]), np.array([7, 8, 9])))
input = np.array([input])
inputs_size = 10 # inputs size
layers_size = [128, 32, 1] # layers sizes
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(inputs_size,)))
model.add(tf.keras.layers.Dense(layers_size[0], activation='relu', name='1'))
model.add(tf.keras.layers.Dense(layers_size[1], activation='relu', name='2'))
model.add(tf.keras.layers.Dense(layers_size[2], activation='sigmoid', name='3'))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=tf.keras.metrics.Accuracy(),
    )
print(input)
output = model.predict(input)
print('out', output)
sets = []
l1w = np.ones((inputs_size, layers_size[0]))
l1b = np.ones(layers_size[0])
l2w = np.ones((layers_size[0], layers_size[1]))
l2b = np.ones(layers_size[1])
l3w = np.ones((layers_size[1], layers_size[2]))
l3b = np.ones(layers_size[2])
sets.append(l1w)
sets.append(l1b)
sets.append(l2w)
sets.append(l2b)
sets.append(l3w)
sets.append(l3b)
model.set_weights(sets)
output = model.predict(input)
print('out', output)
#print(model.get_weights())