
# External libraries
from random import random, \
                   gauss
from math import exp

# Internal libraries
from src.algorithms.subclasses.simulated_annealing import SimulatedAnnealingContext, \
                                                          SimulatedAnnealing


if __name__ == "__main__":

    def generate_neighbor(context, step_size=1):
        """
        Generates a neighbor solution by randomly moving the x-position of a point.
        """
        return context.current_solution + step_size * gauss()


    def accept(context):
        """
        Determines whether to accept the neighbor solution based on the Metropolis criterion.
        """

        if context.neighbor_value < context.current_value:
            # If the neighbor solution is better, always accept it
            prob = 1.0
        else:
            # If the neighbor solution is worse, calculate the acceptance probability
            prob = exp((context.current_value - context.neighbor_value) / context.current_temperature)

        # Return whether the neighbor solution is accepted
        is_accepted = prob > random()

        return is_accepted


    def terminate(context):
        """
        Determines whether to terminate the algorithm based on the temperature schedule.
        """
        return context.temperature_index == len(context.temperature_schedule) - 1


    def objective_function(x):
        """
        Example objective function to minimize a quadratic function.
        """
        return x ** 2


    def lower_temperature(context):
        """
        Lowers the temperature by incrementing the temperature index and updating the current temperature.
        """
        if context.temperature_index < len(context.temperature_schedule) - 1:
            context.temperature_index += 1
            context.current_temperature = \
                context.temperature_schedule[context.temperature_index]


    # Initialize context
    context = SimulatedAnnealingContext(
        temperature_schedule=list(range(1000, 0, -10)),
        initial_solution=10,
        objective_function=objective_function,
        generate_neighbor=generate_neighbor,
        accept=accept,
        terminate=terminate,
        lower_temperature=lower_temperature
    )

    # Instantiate the algorithm
    algo = SimulatedAnnealing(context)

    # Run the algorithm
    algo.run()

    print("Best solution:", algo.best_solution)
    print("Best solution value:", algo.best_value)
