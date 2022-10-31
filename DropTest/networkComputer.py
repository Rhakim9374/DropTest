# imports
from doctest import OutputChecker
import numpy as np
from envstate import EnvState
import tensorflow as tf

class NetworkComputer:
    def __init__(self, mainEnvState, target=0):
        self.computerEnvState = mainEnvState
        self.TARGET_ALTITUDE = target
        self.TIMESTEP = self.computerEnvState.get_timestep()
        self.altitudes = np.zeros(256) + target
        self.throttles = np.zeros(256)

    def update_telemetry(self):
        self.altitudes = np.roll(self.altitudes, 1)
        self.throttles = np.roll(self.throttles, 1)
        self.altitudes[0], self.throttles[0] = self.computerEnvState.get_telemetry()

    def control_throttle(self):
        input = np.concatenate((self.altitudes, self.throttles))
        input = np.array([input])
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, activation='relu', name='1'))
        model.add(tf.keras.layers.Dense(8, activation='relu', name='1'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='3'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=tf.keras.metrics.Accuracy(),
        )
        output = model(input, training=False)
        print(output)




        if self.altitudes[0] < 0.5:
            self.computerEnvState.update_throttle(-1)
        else:
            self.computerEnvState.update_throttle(0)