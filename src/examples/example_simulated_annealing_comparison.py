
# External libraries
from random import random, gauss, randint
import matplotlib.pyplot as plt
from math import exp, pi
import seaborn as sns
import numpy as np

# Internal libraries
from src.algorithms.subclasses.simulated_annealing import SimulatedAnnealingContext, \
                                                          SimulatedAnnealing


if __name__ == "__main__":

    def generate_neighbor(context, step_size=1) -> float | np.ndarray:
        """
        Generates a neighbor solution by randomly moving the x-position of a point.
        """
        return context.current_solution + step_size * gauss()


    def accept_exp(context):
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
    

    def accept_pi(context):
        """
        Determines whether to accept the neighbor solution based on the Metropolis criterion.
        """

        if context.neighbor_value < context.current_value:

            # If the neighbor solution is better, always accept it
            prob = 1.0
        
        else:

            # If the neighbor solution is worse, calculate the acceptance probability
            prob = pi ** ((context.current_value - context.neighbor_value) / context.current_temperature)

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

    # Same starting position for both contexts
    starting_position = randint(0, 1000) / 100

    # Initialize contexts
    context_exp = SimulatedAnnealingContext(
        temperature_schedule=list(range(1000, 0, -10)),
        initial_solution=starting_position,
        objective=objective_function,
        generate_neighbor=generate_neighbor,
        accept=accept_exp,
        terminate=terminate,
        lower_temperature=lower_temperature
    )

    context_pi = SimulatedAnnealingContext(
        temperature_schedule=list(range(1000, 0, -10)),
        initial_solution=starting_position,
        objective=objective_function,
        generate_neighbor=generate_neighbor,
        accept=accept_pi,
        terminate=terminate,
        lower_temperature=lower_temperature
    )

    # Instantiate algorithm
    algo_exp = SimulatedAnnealing(context_exp)
    algo_pi = SimulatedAnnealing(context_pi)

    # Extract mean best values distributions
    algo_exp_mean_best = algo_exp.generate_statistic_distribution(np.mean, sample_size=100)
    algo_pi_mean_best = algo_pi.generate_statistic_distribution(np.mean, sample_size=100)

    # Plot results
    sns.histplot(algo_exp_mean_best, kde=False, color='blue', label='exp', alpha=0.5, stat='density')
    sns.kdeplot(algo_exp_mean_best, color='blue', linestyle=':', lw=2)

    sns.histplot(algo_pi_mean_best, kde=False, color='green', label='pi', alpha=0.5, stat='density')
    sns.kdeplot(algo_pi_mean_best, color='green', linestyle='--', lw=2)

    plt.legend(loc='upper right')
    plt.show()
