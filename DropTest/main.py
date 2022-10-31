# imports
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from envstate import EnvState
from computer import Computer
from computerPID import ComputerPID
from networkComputer import NetworkComputer

def main():
    print('>>> start >>>')
    mainEnvState = EnvState(500)
    # Choose Computer ----------------
    #mainComputer = Computer(mainEnvState)
    #mainComputer = ComputerPID(mainEnvState, 1)
    mainComputer = NetworkComputer(mainEnvState, 1)
    # --------------------------------
    altitude, throttle = mainEnvState.get_telemetry()
    altitudes = []
    throttles = []
    steps = []

    i = 0
    while (altitude > 0):
        print('telemetry >>> ', round(altitude, 0))
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


