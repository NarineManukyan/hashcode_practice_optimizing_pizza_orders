# Author: Narine Hall
# Purpose: Hashcode Competition Practice
# Credits: GA code used from https://github.com/ahmedfgad/GeneticAlgorithmPython

import numpy
import ga
import matplotlib.pyplot

VIS = False
VERBOSE = False


def run_ga(sol_per_pop=8, num_parents_mating=4):
    """
    The y=target is to maximize this equation:
        y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
        where (x1,x2,x3,x4,x5,x6)= pizza slices
        What are the best values for the 6 weights w1 to w6?
        Weights can only be 0 (absent) and 1(included)
        We are going to use the genetic algorithm for the best possible values after a number of generations.
        Genetic algorithm parameters:
        Mating pool size
        Population size
    """

    try:
        f = open("b_small.in", "r")
        if f.mode == 'r':
            f1 = f.readlines()
            for row_index in range(0, 2):
                if row_index == 0:
                    max_val = int(f1[row_index].split(' ')[0])
                elif row_index == 1:
                    eq_inputs = [int(val) for val in f1[row_index].split()]
    except FileNotFoundError:
        print('Can not find the file')
    finally:
        f.close()

    # Number of the weights we are looking to optimize.
    num_weights = len(eq_inputs)

    # Defining the population size.
    pop_size = (sol_per_pop,
                num_weights)
    # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    # Creating the initial population.
    new_population = numpy.random.randint(low=0, high=2, size=pop_size)
    if VERBOSE:
        print(new_population)

    best_outputs = []
    num_generations = 1000
    for generation in range(num_generations):
        if VERBOSE:
            print("Generation : ", generation)
        # Measuring the fitness of each chromosome in the population.
        fitness_val = ga.cal_pop_fitness(eq_inputs, new_population, max_val)
        if VERBOSE:
            print("Fitness")
            print(fitness_val)

        best_outputs.append(numpy.max(numpy.sum(new_population * eq_inputs, axis=1)))
        # The best result in the current iteration.
        if VERBOSE:
            print("Best result : ", numpy.max(numpy.sum(new_population * eq_inputs, axis=1)))

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness_val,
                                        num_parents_mating)
        if VERBOSE:
            print("Parents")
            print(parents)

        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents,
                                           offspring_size=(pop_size[0] - parents.shape[0], num_weights))
        if VERBOSE:
            print("Crossover")
            print(offspring_crossover)

        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
        if VERBOSE:
            print("Mutation")
            print(offspring_mutation)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    # Getting the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    fitness_val = ga.cal_pop_fitness(eq_inputs, new_population, max_val)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness_val == numpy.max(fitness_val))

    best_match_idx = best_match_idx[0][0]
    if VERBOSE:
        print("Best solution : ", new_population[best_match_idx, :])
        print("Best solution fitness : ", fitness_val[best_match_idx])
        print('Max is %s', max_val)
        print('Pizza Slices are %s ', str(eq_inputs))
    return new_population[best_match_idx, :], fitness_val[best_match_idx], max_val, str(eq_inputs)

    if VIS:
        matplotlib.pyplot.plot(best_outputs)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Fitness")
        matplotlib.pyplot.show()


fit = 0
for i in range(100):
    solution, fitness, max_iter_val, equation_inputs = run_ga()
    if fitness > fit:
        fit = fitness
        sol = solution
        m = max_iter_val
        eqs = equation_inputs
        print('Fitness %s fit, solution %s, max %s, eqs = %s ', (fit, sol, m, eqs))
print('Final Fitness %s fit, solution %s ', (fit, sol))
