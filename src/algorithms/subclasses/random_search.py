
# External libraries
from dataclasses import dataclass

# Internal libraries
from src.algorithms.abstract_context import AbstractContext
from src.algorithms.algorithm import Algorithm


@dataclass
class RandomSearchContext(AbstractContext):
    """
    Represents the context for the Random Search algorithm.

    Attributes:
        initial_solution (list): The initial solution for the algorithm.
        current_solution (list): The current solution being evaluated.
        current_value (float): The value of the current solution.
        best_solution (list): The best solution found so far.
        best_value (float): The value of the best solution found so far.
        objective (callable): The objective function used to evaluate solutions.
        generate_random_solution (callable): The function used to generate a random solution.
        terminate (callable): The function used to determine whether to terminate the algorithm.
    """
    iter: int = 0
    max_iter: int = None
    step_size: int | float = None
    initial_solution: list = None
    initial_value: list = None
    current_solution: list = None
    current_value: float = None
    best_solution: list = None
    best_value: float = None
    objective: callable = None
    generate_neighbor: callable = None
    terminate: callable = None


class RandomSearch(Algorithm):
    """
    Modular implementation of the Random Search algorithm.

    Args:
        context (RandomSearchContext): The context object containing the algorithm parameters.

    Attributes:
        current_solution (np.ndarray): The current solution being evaluated.
        current_value (float): The value of the current solution.
        best_solution (np.ndarray): The best solution found so far.
        best_value (float): The value of the best solution found so far.
    """

    def __init__(self, 
                 context: RandomSearchContext):

        # Initialize the context, solutions, and values
        self.initialize(context) 

    def run(self):
        """
        Run the Random Search algorithm.

        Args:
            objective (callable): The objective function to be optimized.

        Returns:
            None.
        """

        # Main loop
        while not self.terminate():

            # Generate neighbor solution
            self.current_solution = self.generate_neighbor()

            # Increment iteration counter
            self.iter += 1
