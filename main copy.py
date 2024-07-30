
# External libraries
from random import random, \
                   gauss
from random import randint

# Internal libraries
from src.algorithms.subclasses.random_search import RandomSearchContext, \
                                                    RandomSearch


if __name__ == "__main__":

    def generate_neighbor(context) -> None:
        return context.current_solution + context.step_size * gauss()
    
    def objective_function(x):
        return x ** 2
    
    def terminate(context) -> bool:
        return context.iter >= context.max_iter

    context = RandomSearchContext(
        max_iter = 100,
        step_size=1,
        initial_solution=randint(0, 10),
        objective=objective_function,
        generate_neighbor=generate_neighbor,
        terminate=terminate
    )

    # Instantiate the algorithm
    algo = RandomSearch(context)

    # Run the algorithm
    algo.run()

    print("Best solution:", algo.best_solution)
    print("Best solution value:", algo.best_value)