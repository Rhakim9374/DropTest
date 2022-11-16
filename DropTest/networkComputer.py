# imports
import numpy as np
import tensorflow as tf
import math
from time import sleep

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class NetworkComputer:
    def __init__(self, mainEnvState, network, target=0):
        self.computerEnvState = mainEnvState
        self.TARGET_ALTITUDE = target
        self.TIMESTEP = self.computerEnvState.get_timestep()
        self.altitudes = np.zeros(5) + target # initialize like this so i starts zero
        self.throttles = np.zeros(5)
        self.network = network

    def update_telemetry(self):
        self.altitudes = np.roll(self.altitudes, 1)
        self.throttles = np.roll(self.throttles, 1)
        self.altitudes[0], self.throttles[0] = self.computerEnvState.get_telemetry()

    def control_throttle(self, p2=0.2, d2=0.00001):
        velocity = (self.altitudes[-4] - self.altitudes[-1]) / 3*self.TIMESTEP
        input = np.zeros(2)
        input[0] = self.altitudes[0]
        input[1] = velocity
        input = np.array([input])
        output = self.network.predict(input)
        target_throttle = float(output)
        current_throttle = self.throttles[0]
        current_throttle_vel = (self.throttles[0] - self.throttles[3]) / (self.TIMESTEP * 3)
        prop2 = -(current_throttle-target_throttle)
        diff2 = -current_throttle_vel
        delta_throttle = p2*prop2 + d2*diff2
        if self.altitudes[0] < 0.2:
            self.computerEnvState.update_throttle(-0.05)
        else:
            self.computerEnvState.update_throttle(delta_throttle)

    def get_network(self):
        return self.network
        