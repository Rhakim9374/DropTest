# imports
import numpy as np
from envstate import EnvState

class ComputerPID:
    def __init__(self, mainEnvState, target=0):
        self.computerEnvState = mainEnvState
        self.TARGET_ALTITUDE = target
        self.altitudes = np.zeros(256) + target
        self.throttles = np.zeros(256)
        self.TIMESTEP = self.computerEnvState.get_timestep()

    def update_telemetry(self):
        self.altitudes = np.roll(self.altitudes, 1)
        self.throttles = np.roll(self.throttles, 1)
        self.altitudes[0], self.throttles[0] = self.computerEnvState.get_telemetry()

    def control_throttle(self, p=0.1, d=1.5, i=0.0005, p2=0.2, d2=0.00001):
        target_altitude = self.TARGET_ALTITUDE
        current_altitude = self.altitudes[0]
        current_velocity = (self.altitudes[0] - self.altitudes[3]) / (self.TIMESTEP * 3)
        proportional = -(current_altitude-target_altitude)
        differential = -current_velocity
        integral = -(np.mean(self.altitudes) - target_altitude) * len(self.altitudes)
        print('p', round(p*proportional, 3))
        print('d', round(d*differential, 3))
        print('i', round(i*integral, 3))
        target_throttle = p*proportional + d*differential + i*integral
        current_throttle = self.throttles[0]
        current_throttle_vel = (self.throttles[0] - self.throttles[3]) / (self.TIMESTEP * 3)
        prop2 = -(current_throttle-target_throttle)
        diff2 = -current_throttle_vel
        delta_throttle = p2*prop2 + d2*diff2
        if self.altitudes[0] < 0.5:
            self.computerEnvState.update_throttle(-1)
        else:
            self.computerEnvState.update_throttle(delta_throttle)