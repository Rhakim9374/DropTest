import random
import numpy as np
from time import sleep
import tensorflow as tf
from neuralNetwork import Network
from envstate import EnvState
from main import main

# initialize neural network
input_size = 2 # inputs size
layer_sizes = [33, 11, 1] # must be 3 layers including output layer
rescaling_factor = 1./1 # based on altitude diff
network = Network(input_size, layer_sizes=layer_sizes, rescaling_factor=rescaling_factor)

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

# some helper functions
def set_neurons_from_competitor(matrix, c):
    sets = []
    l1w = np.reshape(matrix[c, 0:l1w_size], (input_size, layer_sizes[0]))
    l1b = np.reshape(matrix[c, l1w.size:l1w.size+l1b_size], (layer_sizes[0]))
    sizeNow = l1w.size+l1b.size
    l2w = np.reshape(matrix[c, sizeNow:sizeNow+l2w_size], (layer_sizes[0], layer_sizes[1]))
    l2b = np.reshape(matrix[c, sizeNow+l2w.size:sizeNow+l2w.size+l2b_size], (layer_sizes[1]))
    sizeNow = l1w.size+l1b.size+l2w.size+l2b.size
    l3w = np.reshape(matrix[c, sizeNow:sizeNow+l3w_size], (layer_sizes[1], layer_sizes[2]))
    l3b = np.reshape(matrix[c, sizeNow+l3w.size:sizeNow+l3w.size+l3b_size], (layer_sizes[2]))
    sizeNow = l1w.size+l1b.size+l2w.size+l2b.size + l3w.size + l3b.size
    l0w = np.reshape(matrix[c, sizeNow:sizeNow+l0w_size], input_size)
    l0b = np.reshape(matrix[c, sizeNow+l0w.size:sizeNow+l0w.size+l0b_size], input_size)
    l0r = np.array(matrix[c, -1])

    #l1b = np.zeros(layer_sizes[0])
    #l2b = np.zeros(layer_sizes[1])
    #l3b = np.zeros(layer_sizes[2])
    #l0b = np.zeros(input_size)

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

def choose_parents(matrix, values):
    print('CP values: ', values)
    values = values + .01
    worst=max(values)
    best=min(values)
    indexofmin=np.argmin(values)
    values[indexofmin]+=0.5
    likelihoods= worst/values
    print('CP likelihoods: ', likelihoods)
    length=len(matrix[:, 0])
    choices = random.choices(range(length),weights=likelihoods,k=length)
    print('CP choices: ', choices)
    chosen_ones=matrix[choices,:]
    return chosen_ones

def crossover(matrix, values, index_to_consider):
    parents=choose_parents(matrix[index_to_consider:,:],values[index_to_consider:])
    length=len(matrix)
    children=np.empty(np.shape(matrix), dtype=np.float32)

    count=0
    while count<index_to_consider:
        children[count,:]=matrix[count,:]
        count+=1

    while count<length:
        beta=random.random()
        crossover_point=random.randint(0,(len(matrix[0, :])-1))
        second_crossover=random.randint(0,(len(matrix[0, :])-1))
        while second_crossover == crossover_point:
            second_crossover=random.randint(0,(len(matrix[0, :])-1))

        child1=parents[count-index_to_consider]
        child2=parents[count+1-index_to_consider]
    

        child1[crossover_point]=float((1.-beta)*float(child1[crossover_point])+beta*float(child2[crossover_point]))
        child2[crossover_point]=float((1.-beta)*float(child2[crossover_point])+beta*float(child1[crossover_point]))
        child1[second_crossover]=float((1.-beta)*float(child1[second_crossover])+beta*float(child2[second_crossover]))
        child2[second_crossover]=float((1.-beta)*float(child2[second_crossover])+beta*float(child1[second_crossover]))
        children[count]=child1
        children[count+1]=child2
        
        count+=2
    return children

def mutate(children,prob_mutation,max_mutation):
    temp = children
    for combo in temp:
        length = np.size(combo)
        choices=np.random.choice(np.arange(0, length), size=int(prob_mutation*length))
        for c in choices:
            combo[c]+= np.random.uniform(-max_mutation, max_mutation)
        if 0.01 > np.random.uniform(0, 1):
            choice = np.random.choice(np.arange(0, length), size=1)
            combo[choice] = np.random.uniform(-1.0, 1.0)
    return temp
    


# ------------------------------------------------

# initialize genetic algorithm
fake_env = EnvState()
maxsteps = 100 / fake_env.TIMESTEP # (0.2 * max meters) / (timestep)
initial_num_competitors = 32 # must be even (32)
num_competitors = 16 # must be even (16)
numruns = 100 # number of generations with constant drop (100)
index_to_consider = 0 # must be even?
prob_mutation = 0.05  # Percentage of nodes to get changed per generational mutation
max_mutation = 0.05  # Max introduced change

matrix = np.zeros((initial_num_competitors, array_length))
values = np.zeros(initial_num_competitors)

print('initial predict ', network.predict(np.array([np.ones(input_size)])))

generational_costs = []

for c in range(initial_num_competitors):
    start = random.uniform(500, 1500)
    end = random.uniform(500, 1500)
    avg_initial = np.random.normal(0, scale=0.1)
    spread_initial = np.random.uniform(0.2, 0.8)
    matrix[c, :] = np.random.normal(avg_initial, scale=spread_initial, size=array_length)
    set_neurons_from_competitor(matrix, c)
    final_vel, final_dist, steps = main(network=network,starting_altitude=start,target_altitude=end, training=True, maxsteps=maxsteps)
    values[c] = final_vel**2 + final_dist**4 + 0.1*steps
    print('Competitor ', c+1, ' Cost: ', values[c])
    sleep(5)
generational_costs.append(round(np.min(values), 2))

# ------filter from initial_num_competitors to num_competitors-------
best = []
track_values = []
print(values)
while len(best) < num_competitors:
    min_index = np.argmin(values)
    best.append(min_index)
    track_values.append(values[min_index])
    values[min_index] = 2.0*10**25
matrix = matrix[best]
values = np.array(track_values)
print(values)
print('Successful transition to second stage')
sleep(100)

matrix = crossover(matrix, values, index_to_consider=0)
matrix = mutate(matrix, prob_mutation=prob_mutation, max_mutation=max_mutation)
# -------------------------------------------------------------------

g = 0
while (g < numruns):
    print('Generation ', g+1)
    print('Min cost by generation: ', generational_costs)
    sleep(20)
    for c in range(num_competitors):
        set_neurons_from_competitor(matrix, c)
        acc = np.zeros(5)
        for trial in range(len(acc)):
            start = random.uniform(500, 1500)
            end = random.uniform(500, 1500)
            final_vel, final_dist, steps = main(network=network,starting_altitude=start,target_altitude=end, training=True, maxsteps=maxsteps)
            acc[trial] = final_vel**2 + final_dist**4 + 0.1*steps
        values[c] = np.mean(acc)
        print('Competitor ', c+1, ' Cost: ', values[c])
        sleep(5)
    generational_costs.append(round(np.min(values), 2))
    matrix = crossover(matrix, values, index_to_consider=0)
    matrix = mutate(matrix, prob_mutation=prob_mutation, max_mutation=max_mutation/(np.sqrt(g+1)))
    g+=1

print('Values for final generation: ', values)
print('Min cost final generation: ', generational_costs)

mean_weights = np.mean(matrix, axis=0)
np.save('model_weights.npy', mean_weights)
loaded_weights_check = np.load('model_weights.npy')

print('Sim Complete')






