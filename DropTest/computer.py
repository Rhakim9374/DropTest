# imports
import numpy as np
from envstate import EnvState

class Computer:
    def __init__(self, mainEnvState):
        self.computerEnvState = mainEnvState
        self.altitudes = np.zeros(10)
        self.throttles = np.zeros(10)
        self.TIMESTEP = self.computerEnvState.get_timestep()

    def update_telemetry(self):
        self.altitudes = np.roll(self.altitudes, 1)
        self.throttles = np.roll(self.throttles, 1)
        self.altitudes[0], self.throttles[0] = self.computerEnvState.get_telemetry()

    def control_throttle(self):
        velocity_estimate = (self.altitudes[0] - self.altitudes[4]) / (self.TIMESTEP * 4)
        old_velocity_estimate = (self.altitudes[5] - self.altitudes[9]) / (self.TIMESTEP * 4)
        acceleration_estimate = (velocity_estimate - old_velocity_estimate) / (self.TIMESTEP * 5)
        target_acceleration =  -(np.abs(velocity_estimate) * velocity_estimate) / (2*self.altitudes[0])
        print('computer >>> ', round(target_acceleration, 3))
        if self.altitudes[0] > 3000:
            pass
        elif self.altitudes[0] < 0.5:
            self.computerEnvState.update_throttle(-1)
        elif target_acceleration > acceleration_estimate:
            self.computerEnvState.update_throttle(.002)
        else:
            self.computerEnvState.update_throttle(-.002)


        
