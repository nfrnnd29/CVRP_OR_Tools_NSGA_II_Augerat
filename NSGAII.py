# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:20:22 2021

@author: asus
"""
##NOTE
# How to run NSGA-II program code

# File names for data input in the program: A-n32-k5, A-n33-k5, A-n33-k6, A-n34-k5, A-n36-k5, A-n37-k5, A-n39-k5, A-n45-k7, A-n46-k7, A-n48-k7, A-n54-k7, A-n63-k9, A-n80-k10

# 1. Ensure that data with json format is available in the data/json folder.
#If not available, convert text to json by using the program file ConvertTxt2Json.py.
#In the ConvertTxt2Json.py program file, just directly click run and the txt file will automatically change to json along with the euclidean distance calculation data into one json file.

# 2. In the NSGAII.py file, check the input json file for instances that have been converted to json files. There are three different locations for renaming instances:
# (a) In Class nsgaAlgo(), make changes to self.json_instance = load_instance('./data/json/A-n45-k7.json'), changes are made according to the name of the problem code listed in the file title such as "A- n32-k5" etc.
# (b) def nsga2vrp(), make changes to json_instance = load_instance('./data/json/A-n45-k7.json'), changes are made according to the name of the problem code listed in the file title like "A-n32-k5 " etc.
# (c) def main(), make changes to parser.add_argument('--instance_name', type=str, default="./data/json/A-n45-k7.json", required=False, help=" Enter the input Json file name"), changes are made according to the name of the problem code listed in the file title such as "A-n32-k5" and etc.

# 3. Ensure that the input parameters match the default NSGAII file, namely popsize = 500, crossProb = 0.85, mutProb = 0.02, and numGen = 1000

# 4. It is expected to wait a few minutes because the number of iterations is 1000 according to the iteration of the comparison paper with the Improved FA method.
#The results displayed are the final results of iterations and are not the results of each generation. The results shown are the best individual, the number of vehicles needed, the total distance cost, and the route of each vehicle. 

import os
import io
import random
# import numpy
# import fnmatch
# import csv
# import array
import argparse

#from csv import DictWriter
from json import load, dump
from deap import base, creator, tools, algorithms, benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


# Load the given problem, which can be a json file
def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None


# Take a route of given length, divide it into subroute where each subroute is assigned to vehicle
def routeToSubroute(individual, instance):
    """
    Inputs: Sequence of customers that a route has
            Loaded instance problem
    Outputs: Route that is divided in to subroutes
             which is assigned to each vechicle.
    """
    route = []
    sub_route = []
    vehicle_load = 0
    last_customer_id = 0
    vehicle_capacity = instance['vehicle_capacity']
    
    for customer_id in individual:
        # print(customer_id)
        demand = instance[f"customer_{customer_id}"]["demand"]
        # print(f"The demand for customer_{customer_id}  is {demand}")
        updated_vehicle_load = vehicle_load + demand

        if(updated_vehicle_load <= vehicle_capacity):
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
        else:
            route.append(sub_route)
            sub_route = [customer_id]
            vehicle_load = demand
        
        last_customer_id = customer_id

    if sub_route != []:
        route.append(sub_route)

    # Returning the final route with each list inside for a vehicle
    return route


def printRoute(route, merge=False):
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


# Calculate the number of vehicles required, given a route
def getNumVehiclesRequired(individual, instance):
    """
    Inputs: Individual route
            Json file object loaded instance
    Outputs: Number of vechiles according to the given problem and the route
    """
    # Get the route with subroutes divided according to demand
    updated_route = routeToSubroute(individual, instance)
    num_of_vehicles = len(updated_route)
    return num_of_vehicles


# Given a route, give its total cost
def getRouteCost(individual, instance, unit_cost=1):
    """
    Inputs : 
        - Individual route
        - Problem instance, json file that is loaded
        - Unit cost for the route (can be petrol etc)

    Outputs:
        - Total cost for the route taken by all the vehicles
    """
    total_cost = 0
    updated_route = routeToSubroute(individual, instance)

    for sub_route in updated_route:
        # Initializing the subroute distance to 0
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0

        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = instance["distance_matrix"][last_customer_id][customer_id]
            sub_route_distance += distance
            # Update last_customer_id to the new one
            last_customer_id = customer_id
        
        # After adding distances in subroute, adding the route cost from last customer to depot
        # that is 0
        sub_route_distance = sub_route_distance + instance["distance_matrix"][last_customer_id][0]

        # Cost for this particular sub route
        sub_route_transport_cost = unit_cost*sub_route_distance

        # Adding this to total cost
        total_cost = total_cost + sub_route_transport_cost
    
    return total_cost


# Get the fitness of a given route
def eval_indvidual_fitness(individual, instance, unit_cost):
    """
    Inputs: individual route as a sequence
            Json object that is loaded as file object
            unit_cost for the distance 
    Outputs: Returns a tuple of (Number of vechicles, Route cost from all the vechicles)
    """

    # we have to minimize number of vehicles
    # TO calculate req vechicles for given route
    vehicles = getNumVehiclesRequired(individual, instance)

    # we also have to minimize route cost for all the vehicles
    route_cost = getRouteCost(individual, instance, unit_cost)

    return (vehicles, route_cost)



# Crossover method with ordering
# This method will let us escape illegal routes with multiple occurences
#   of customers that might happen. We would never get illegal individual from this
#   crossOver
def cxOrderedVrp(input_ind1, input_ind2):
    # Modifying this to suit our needs
    #  If the sequence does not contain 0, this throws error
    #  So we will modify inputs here itself and then 
    #       modify the outputs too

    ind1 = [x-1 for x in input_ind1]
    ind2 = [x-1 for x in input_ind2]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    # print(f"The cutting points are {a} and {b}")
    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Finally adding 1 again to reclaim original input
    ind1 = [x+1 for x in ind1]
    ind2 = [x+1 for x in ind2]
    return ind1, ind2


def mutationShuffle(individual, indpb):
    """
    Inputs : Individual route
             Probability of mutation betwen (0,1)
    Outputs : Mutated individual according to the probability
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual,

# In the nsga Algo class, rename the .json file to be used for optimization
#Rename for this variable --> self.json_instance = load_instance('./data/json/A-n33-k5.json')
#Rename "A-n33-k5" with another file name in the data/json folder
class nsgaAlgo(object):

    def __init__(self):
        #Rename the .json file as stated in the data/json folder
        self.json_instance = load_instance('./data/json/A-n80-k10.json')
        self.ind_size = self.json_instance['Number_of_customers']
        self.pop_size = 100
        self.cross_prob = 0.85
        self.mut_prob = 0.02
        self.num_gen = 1000
        self.toolbox = base.Toolbox()
        #self.logbook, self.stats = createStatsObjs()
        self.createCreators()

    def createCreators(self):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # Registering toolbox
        self.toolbox.register('indexes', random.sample, range(1, self.ind_size + 1), self.ind_size)

        # Creating individual and population from that each individual
        self.toolbox.register('individual', tools.initIterate, creator.Individual, self.toolbox.indexes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        # Creating evaluate function using our custom fitness
        #   toolbox.register is partial, *args and **kwargs can be given here
        #   and the rest of args are supplied in code
        self.toolbox.register('evaluate', eval_indvidual_fitness, instance=self.json_instance, unit_cost=1)

        # Selection method
        self.toolbox.register("select", tools.selNSGA2)

        # Crossover method
        self.toolbox.register("mate", cxOrderedVrp)

        # Mutation method
        self.toolbox.register("mutate", mutationShuffle, indpb=self.mut_prob)


    def generatingPopFitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = list(map(self.toolbox.evaluate, self.invalid_ind))

        for ind, fit in zip(self.invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        self.pop = self.toolbox.select(self.pop, len(self.pop))

        #recordStat(self.invalid_ind, self.logbook, self.pop, self.stats, gen = 0)


    def runGenerations(self):
        # Running algorithm for given number of generations
        for gen in range(self.num_gen):
            #print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")

            # Selecting individuals
            # Selecting offsprings from the population, about 1/2 of them
            self.offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            self.offspring = [self.toolbox.clone(ind) for ind in self.offspring]

            # Performing , crossover and mutation operations according to their probabilities
            for ind1, ind2 in zip(self.offspring[::2], self.offspring[1::2]):
                # Mating will happen 80% of time if cross_prob is 0.8
                if random.random() <= self.cross_prob:
                    # print("Mating happened")
                    self.toolbox.mate(ind1, ind2)

                    # If cross over happened to the individuals then we are deleting those individual
                    #   fitness values, This operations are being done on the offspring population.
                    del ind1.fitness.values, ind2.fitness.values
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)

            # Calculating fitness for all the invalid individuals in offspring
            self.invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.invalid_ind)
            for ind, fit in zip(self.invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            # Recalcuate the population with newly added offsprings and parents
            # We are using NSGA2 selection method, We have to select same population size
            self.pop = self.toolbox.select(self.pop + self.offspring, self.pop_size)

            # Recording stats in this generation
            #recordStat(self.invalid_ind, self.logbook, self.pop, self.stats, gen + 1)

        print(f"{20 * '#'} End of Generations {20 * '#'} ")


    def getBestInd(self):
        self.best_individual = tools.selBest(self.pop, 1)[0]

        # Printing the best after all generations
        print(f"Best individual is {self.best_individual}")
        print(f"Number of vechicles required are "
              f"{self.best_individual.fitness.values[0]}")
        print(f"Cost required for the transportation is "
              f"{self.best_individual.fitness.values[1]}")

        # Printing the route from the best individual
        printRoute(routeToSubroute(self.best_individual, self.json_instance))


    def runMain(self):
        self.generatingPopFitness()
        self.runGenerations()
        self.getBestInd()

#In the nsga2vrp function rename "A-n33-k5" json_instance = load_instance('./data/json/A-n33-k5.json')
#Rename "A-n33-k5" with another file name in the data/json folder 
def nsga2vrp():

    # Loading the instance
    #Change the name of the instance file as stated in the json data folder 
    json_instance = load_instance('./data/json/A-n80-k10.json')
    
    # Getting number of customers to get individual size
    ind_size = json_instance['Number_of_customers']

    # Setting variables
    pop_size = 500
    # Crossover probability
    cross_prob = 0.85
    # Mutation probability
    mut_prob = 0.02
    # Number of generations to run
    num_gen = 1000

    # Developing Deap algorithm from base problem    
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    # Registering toolbox
    toolbox = base.Toolbox()
    toolbox.register('indexes', random.sample, range(1,ind_size+1), ind_size)

    # Creating individual and population from that each individual
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    # Creating evaluate function using our custom fitness
    #   toolbox.register is partial, *args and **kwargs can be given here
    #   and the rest of args are supplied in code
    toolbox.register('evaluate', eval_indvidual_fitness, instance=json_instance, unit_cost = 1)

    # Selection method
    toolbox.register("select", tools.selNSGA2)

    # Crossover method
    toolbox.register("mate", cxOrderedVrp)

    # Mutation method
    toolbox.register("mutate", mutationShuffle, indpb = mut_prob)


    ### Starting ga process
    print(f"Generating population with size of {pop_size}")
    pop = toolbox.population(n=pop_size)


    # Getting all invalid individuals who don't have fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]   

    # Evaluate the population, making list for same size as population
    fitnesses = list(map(toolbox.evaluate, invalid_ind))

    # Assigning fitness attribute to each of the individual with the calculated one
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Assigning crowding distance using NSGA selection process, no selection is done here
    pop = toolbox.select(pop, len(pop))

    # Starting the generation process
    for gen in range(num_gen):
        #print(f"######## Currently Evaluating {gen} Generation ######## ")

        # Selecting individuals
        # Selecting offsprings from the population, about 1/2 of them
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]


        # Performing , crossover and mutation operations according to their probabilities
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # Mating will happen 80% of time if cross_prob is 0.8
            if random.random() <= cross_prob:
                # print("Mating happened")
                toolbox.mate(ind1, ind2)

                # If cross over happened to the individuals then we are deleting those individual
                #   fitness values, This operations are being done on the offspring population.
                del ind1.fitness.values, ind2.fitness.values                 
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)

        # Calculating fitness for all the invalid individuals in offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        

        # Recalcuate the population with newly added offsprings and parents
        # We are using NSGA2 selection method, We have to select same population size
        pop = toolbox.select(pop + offspring, pop_size)

    best_individual = tools.selBest(pop, 1)[0]

    # Printing the best after all generations
    print(f"Best individual is {best_individual}")
    print(f"Number of vechicles required are {best_individual.fitness.values[0]}")
    print(f"Cost required for the transportation is {best_individual.fitness.values[1]}")

    # Printing the route from the best individual
    printRoute(routeToSubroute(best_individual, json_instance))


##RUN PROGRAM
#1. Input data according to the name of the instance file in the data folder with the .json format in the nsgaAlgo class, def nsga2vrp, and def main
#2. Fill in the default arguments to run NSGA-II
#3. Please wait if using 1000 iterations as it will only produce the final result 
def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_name', type=str, default="./data/json/A-n80-k10.json", required=False,
                        help="Enter the input Json file name")
    parser.add_argument('--popSize', type=int, default=500, required=False,
                        help="Enter the population size")
    parser.add_argument('--crossProb', type=float, default=0.85, required=False,
                        help="Crossover Probability")
    parser.add_argument('--mutProb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--numGen', type=int, default=1000, required=False,
                        help="Number of generations to run")


    args = parser.parse_args()

    # Initializing instance
    nsgaObj = nsgaAlgo()

    # Setting internal variables
    nsgaObj.json_instance = load_instance(args.instance_name)
    nsgaObj.pop_size = args.popSize
    nsgaObj.cross_prob = args.crossProb
    nsgaObj.mut_prob = args.mutProb
    nsgaObj.num_gen = args.numGen

    # Running Algorithm
    nsgaObj.runMain()


if __name__ == '__main__':
    main()