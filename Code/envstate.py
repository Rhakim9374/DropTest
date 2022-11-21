# imports
import numpy as np

class EnvState:
    def __init__(self, altitude=5000):
        self.TIMESTEP = 0.5
        self.GRAVITY = -9.81
        self.MAX_THRUST = 100
        self.MASS = 5
        self.altitude = altitude
        self.velocity = np.random.uniform(-250, 250)
        self.throttle = 0.5
    
    def update_state(self):
        self.altitude += self.velocity*self.TIMESTEP
        self.velocity += self.GRAVITY*self.TIMESTEP
        self.velocity += (self.throttle*self.MAX_THRUST/self.MASS)*self.TIMESTEP
        self.velocity += np.random.uniform(0, 1)*self.TIMESTEP

    def update_throttle(self, delta_throttle):
        self.throttle += delta_throttle
        self.throttle += np.random.uniform(-0.001, 0.001)
        if self.throttle > 1:
            self.throttle = 1
        if self.throttle < 0:
            self.throttle = 0

    def get_telemetry(self):
        return self.altitude, self.throttle
    
    def get_timestep(self):
        return self.TIMESTEP