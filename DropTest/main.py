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

def main(network, starting_altitude=100, target_altitude=0, training=False):
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
    while (altitude > 0 and len(steps)<=400):
        if training==True:
            sleep(.0005)
        else:
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

    # initialize neural network variables
    input_size = 2 # inputs size
    layer_sizes = [33, 11, 1] # must be 3 layers including output layer
    rescaling_factor = 1./1 # based on altitude diff

    # layer sizes
    l0w_size = input_size
    l0b_size = input_size
    l0r_size = 1
    l1w_size = input_size*layer_sizes[0]
    l1b_size = layer_sizes[0]
    l2w_size = layer_sizes[0]*layer_sizes[1]
    l2b_size = layer_sizes[1]
    l3w_size = layer_sizes[1]*layer_sizes[2]
    l3b_size = layer_sizes[2]

    # genetic algorithm parameters
    array_length = l1w_size+l1b_size+l2w_size+l2b_size+l3w_size+l3b_size+l0w_size+l0b_size+1

    # set neurons from saved weights
    def set_neurons(matrix, network):
        sets = []
        l1w = np.reshape(matrix[0:l1w_size], (input_size, layer_sizes[0]))
        l1b = np.reshape(matrix[l1w.size:l1w.size+l1b_size], (layer_sizes[0]))
        sizeNow = l1w.size+l1b.size
        l2w = np.reshape(matrix[sizeNow:sizeNow+l2w_size], (layer_sizes[0], layer_sizes[1]))
        l2b = np.reshape(matrix[sizeNow+l2w.size:sizeNow+l2w.size+l2b_size], (layer_sizes[1]))
        sizeNow = l1w.size+l1b.size+l2w.size+l2b.size
        l3w = np.reshape(matrix[sizeNow:sizeNow+l3w_size], (layer_sizes[1], layer_sizes[2]))
        l3b = np.reshape(matrix[sizeNow+l3w.size:sizeNow+l3w.size+l3b_size], (layer_sizes[2]))
        sizeNow = l1w.size+l1b.size+l2w.size+l2b.size + l3w.size + l3b.size
        l0w = np.reshape(matrix[sizeNow:sizeNow+l0w_size], input_size)
        l0b = np.reshape(matrix[sizeNow+l0w.size:sizeNow+l0w.size+l0b_size], input_size)
        l0r = np.array(matrix[-1])

        sets.append(l0w)
        sets.append(l0b)
        sets.append(l0r)
        sets.append(l1w)
        sets.append(l1b)
        sets.append(l2w)
        sets.append(l2b)
        sets.append(l3w)
        sets.append(l3b)
        network.set_weight_and_bias(sets)

    # load saved weights
    main_network = Network(input_size, layer_sizes=layer_sizes)
    main_network.get_model().build((1, 2))
    print('Init test passed: ', main_network.predict(np.array([np.array([1., 0.5])])))

    new_weights = np.load('model_weights.npy')
    set_neurons(new_weights, main_network)

    # ------- run main --------
    print('Main Run')
    sleep(10)
    final_vel, final_dist, steps = main(network=main_network, training=False)


