
# External libraries
import numpy as np
from random import gauss, randint

# Internal libraries
from src.algorithms.subclasses.random_search import RandomSearchContext, \
                                                    RandomSearch


if __name__ == "__main__":

    def generate_neighbor(context) -> None:
        """
        Generates a neighbor by taking a randomly weighted step from the current solution.
        """
        return context.current_solution + context.step_size * gauss()
    

    def objective_function(x):
        """
        Example objective function to minimize a quadratic function.
        """
        return x ** 2
    

    def terminate(context) -> bool:
        """
        Determines whether to terminate the algorithm based on the maximum number of iterations.
        """
        return context.iter >= context.max_iter

    # Initialize context
    context = RandomSearchContext(
        max_iter = 100,
        step_size=0.5,
        initial_solution=randint(0, 1000) / 100,
        objective=objective_function,
        generate_neighbor=generate_neighbor,
        terminate=terminate
    )

    # Instantiate algorithm
    algo = RandomSearch(context)

    # Plot the best value profile
    algo.plot_profile(metric=algo.profiles.BEST,
                      is_legend=True)

    # Plot the best value distribution
    algo.plot_distribution(metric=algo.metrics.BEST, 
                           foo_statistic=np.mean,
                           sample_size=100,
                           is_legend=True)
