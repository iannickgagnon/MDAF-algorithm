from abc import ABC
from dataclasses import dataclass


@dataclass
class BaseContext(ABC):
    """
    Represents the context for an optimization algorithm.

    Attributes:
        solution_initializer (callable): Function to initialize the solution.
        initial_solution (list): The initial solution vector.
        current_solution (list): The current solution vector.
        current_value (float): The value (y) of the current solution.
        best_solution (list): The best solution found so far.
        best_value (float): The value of the best solution found so far.
        nb_agents (int): Number of agents in the algorithm.
        objective (callable): Objective function to be optimized.
    """

    solution_initializer: callable = None
    initial_solution: list = None
    current_solution: list = None
    current_value: float = None
    best_solution: list = None
    best_value: float = None
    nb_agents: int = None
    objective: callable = None

    def copy(self):
        """
        Create and return a shallow copy of the current instance.

        Args:
            None

        Returns:
            object: A new instance of the same class with attributes copied from the current instance.
        """

        # Create a new instance of the same class
        new_instance = type(self)()

        # Copy the attributes of the current instance to the new instance
        new_instance.__dict__.update(self.__dict__)

        return new_instance


@dataclass
class AbstractContext(BaseContext):
    """
    Abstract base class for optimization algorithm contexts.

    This class serves as a blueprint for creating specific algorithm contexts.
    It inherits from BaseContext and can be extended to include additional
    attributes or methods specific to different optimization algorithms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
