# imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import sleep
from envstate import EnvState
from computer import Computer
from computerPID import ComputerPID
from networkComputer import NetworkComputer
from neuralNetwork import Network

def main(network, starting_altitude=40, target_altitude=0):
    starting_altitude = starting_altitude
    target_altitude = target_altitude
    print('>>> start >>>')
    mainEnvState = EnvState(starting_altitude)
    # Choose Computer ----------------
    #mainComputer = Computer(mainEnvState)
    #mainComputer = ComputerPID(mainEnvState, target_altitude)
    mainComputer = NetworkComputer(mainEnvState, network, target_altitude)
    # --------------------------------
    altitude, throttle = mainEnvState.get_telemetry()
    altitudes = []
    throttles = []
    steps = []

    i = 0
    while (altitude > 0 and len(steps)<=501):
        sleep(mainEnvState.get_timestep())
        mainComputer.update_telemetry()
        mainComputer.control_throttle()
        mainEnvState.update_state()
        altitude, throttle = mainEnvState.get_telemetry()
        altitudes.append(altitude)
        throttles.append(throttle)
        steps.append(i)
        print('Distance >>> ', round(altitude-target_altitude, 1))
        print('Current throttle: ', round(throttle, 2))
        i+=1

    final_vel = np.abs((altitudes[-1] - altitudes[-2]) / mainEnvState.get_timestep())
    final_dist = np.abs(altitudes[-1] - target_altitude)
    
    '''
    for l in np.linspace(0, 99, 100):
        altitudes.append(0)
        throttles.append(0)
        steps.append(i)
        i+=1

    
    plt.plot(steps, altitudes)
    plt.show()

    plt.plot(steps, throttles)
    plt.show()
    '''

    print('>>> complete >>>')

    return final_vel, final_dist, len(steps)

if __name__ == "__main__":
    print('Main Run')
    sleep(5)
    main(tf.keras.models.load_model('saved_model/my_network'))


