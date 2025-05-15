
# External libraries
from abc import ABC
from dataclasses import dataclass

@dataclass
class AbstractContext(ABC):
    """
    Represents the context for an optimization algorithm. 
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
        
        # Create a new instance of the same class
        new_instance = type(self)()
        
        # Copy the attributes of the current instance to the new instance
        new_instance.__dict__.update(self.__dict__)
        
        return new_instance

    pass
