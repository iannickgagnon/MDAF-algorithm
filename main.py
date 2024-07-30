
# External libraries
from random import randint, \
                   choice, \
                   random
from math import exp

# Internal libraries
from src.algorithms.subclasses.simulated_annealing import SimulatedAnnealingContext, \
                                                          SimulatedAnnealing


if __name__ == "__main__":

    def generate_neighbor(context):
        neighbor = context.current_solution[:]
        index = randint(0, len(context.current_solution) - 1)
        neighbor[index] += choice([-1, 1])
        return neighbor
    
    def accept(context):
        if context.neighbor_value < context.current_value:
            prob = 1.0
        else:
            prob = exp((context.current_value - context.neighbor_value) / context.current_temperature) 
        return prob > random()

    def terminate(context):
        return context.temperature_index == len(context.temperature_schedule) - 1

    def objective_function(x):
        return sum(x)

    def lower_temperature(context):
        if context.temperature_index < len(context.temperature_schedule) - 1:
            context.temperature_index += 1
            context.current_temperature = context.temperature_schedule[context.temperature_index]

    # Example usage
    context = SimulatedAnnealingContext(
        temperature_schedule=[100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1],
        initial_solution=[10, 10, 10],
        generate_neighbor=generate_neighbor,
        accept=accept,
        terminate=terminate,
        lower_temperature=lower_temperature
    )

    algo = SimulatedAnnealing(context)
    algo.run(objective_function=objective_function)

    print("Best solution:", algo.best_solution)
    print("Best solution value:", algo.best_value)
