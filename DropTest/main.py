# imports
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from envstate import EnvState
from computer import Computer

def main():
    print('>>> start >>>')
    mainEnvState = EnvState()
    mainComputer = Computer(mainEnvState)
    altitude, throttle = mainEnvState.get_telemetry()
    altitudes = []
    throttles = []
    steps = []

    i = 0
    while altitude > 0:
        print('telemetry: ', round(altitude, 0), round(throttle, 3))
        sleep(mainEnvState.get_timestep())
        mainComputer.update_telemetry()
        mainComputer.control_throttle()
        mainEnvState.update_state()
        altitude, throttle = mainEnvState.get_telemetry()
        altitudes.append(altitude)
        throttles.append(throttle)
        steps.append(i)
        i+=1

    print('Touchdown Velocity: ', (altitudes[-1] - altitudes[-2]) / mainEnvState.get_timestep())
    
    for l in np.linspace(0, 99, 100):
        altitudes.append(0)
        throttles.append(0)
        steps.append(i)
        i+=1

    plt.plot(steps, altitudes)
    plt.show()

    plt.plot(steps, throttles)
    plt.show()

    print('>>> complete >>>')

if __name__ == "__main__":
    main()


