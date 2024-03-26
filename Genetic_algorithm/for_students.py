from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


#obliczanie prawdopodobieństwa wyboru
def rouletteWheel(items, knsapsack_max_capacity, population):
    propabilities = []
    denominator = 0
    for j in range(len(population)):
        denominator += fitness(items, knapsack_max_capacity, population[j])
    for i in range(len(population)):
        propability = fitness(items, knapsack_max_capacity, population[i])/denominator
        propabilities.append(propability)
    return propabilities

# tworzenie potomków
def crossOver(population, items):
    tmpPopulation = []
    midPoint = len(items)//2
    midPopulation = len(population)//2
    for i in range(midPopulation):
        firstChild = population[i][:midPoint] + population[i+midPopulation][midPoint:]
        secondChild = population[i][midPoint:] + population[i + midPopulation][:midPoint]
        tmpPopulation.append(firstChild)
        tmpPopulation.append(secondChild)
    return tmpPopulation

# mutacja
def mutation(population, items):
    chance = len(items)
    for i in range(len(population)):
        random_number = random.randint(0,chance-1)
        if population[i][random_number] == True:
            population[i][random_number] = False
        else:
            population[i][random_number] = True


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 400
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
#print(population)
for _ in range(generations):
    population_history.append(population)
    # TODO: implement genetic algorithm
    # wybór rodziców
    probability = rouletteWheel(items, knapsack_max_capacity, population)
    tmp_population = []
    for i in range(len(population)):
        chosen_element = random.choices(population, probability)[0]
        tmp_population.append(chosen_element)
    tmp_population = crossOver(tmp_population, items)
    mutation(tmp_population, items)

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    tmp_population[0] = best_individual
    population = tmp_population
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
