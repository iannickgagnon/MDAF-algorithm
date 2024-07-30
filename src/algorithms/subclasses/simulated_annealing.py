
# External libraries
from dataclasses import dataclass

# Internal libraries
from src.algorithms.abstract_context import AbstractContext
from src.algorithms.algorithm import Algorithm


@dataclass
class SimulatedAnnealingContext(AbstractContext):
    """
    Represents the context for the Simulated Annealing algorithm.

    Attributes:
        temperature_schedule (list): The schedule of temperatures to be used during the algorithm.
        temperature_index (int): The index of the current temperature in the temperature_schedule.
        current_temperature (int): The current temperature being used.
        initial_solution (list): The initial solution for the algorithm.
        current_solution (list): The current solution being evaluated.
        current_value (float): The value of the current solution.
        best_solution (list): The best solution found so far.
        best_value (float): The value of the best solution found so far.
        objective_function (callable): The objective function used to evaluate solutions.
        generate_neighbor (callable): The function used to generate a neighbor solution.
        accept (callable): The function used to determine whether to accept a neighbor solution.
        terminate (callable): The function used to determine whether to terminate the algorithm.
        lower_temperature (callable): The function used to lower the temperature.
    """

    temperature_schedule: list
    temperature_index: int = 0
    current_temperature: int = None
    initial_solution: list = None
    current_solution: list = None
    current_value: float = None
    best_solution: list = None
    best_value: float = None
    objective_function: callable = None
    generate_neighbor: callable = None
    accept: callable = None
    terminate: callable = None
    lower_temperature: callable = None


class SimulatedAnnealing(Algorithm):
    """
    Modular implementation of the Simulated Annealing algorithm.

    Args:
        context (SimulatedAnnealingContext): The context object containing the algorithm parameters.

    Attributes:
        current_solution (np.ndarray): The current solution being evaluated.
        current_value (float): The value of the current solution.
        best_solution (np.ndarray): The best solution found so far.
        best_value (float): The value of the best solution found so far.
        current_temperature (int): The current temperature of the algorithm.
    """

    def __init__(self, 
                 context: SimulatedAnnealingContext):
        
        # Store the context
        super().__init__(context)

        # Initialize the algorithm
        self.current_solution = context.initial_solution
        self.current_value = context.objective_function(context.initial_solution)
        self.best_solution = context.initial_solution
        self.best_value = context.objective_function(context.initial_solution)
        self.current_temperature = context.temperature_schedule[context.temperature_index]


    def run(self):
        """
        Run the Simulated Annealing algorithm.

        Args:
            objective_function (callable): The objective function to be optimized.

        Returns:
            tuple[np.ndarray, float]: The best solution found by the algorithm and its value.
        """

        # Main loop
        while not self.terminate():

            # Generate a neighbor solution
            self.neighbor = self.generate_neighbor()

            # Calculate the value of the neighbor solution
            self.neighbor_value = self.objective_function(self.neighbor)

            # Probabilistic acceptance of the neighbor
            if self.accept():
                self.current_solution = self.neighbor
                self.current_value = self.neighbor_value

            # Update the best solution so far
            if self.objective_function(self.current_solution) < self.objective_function(self.best_solution):
                self.best_solution = self.current_solution
                self.best_value = self.current_value
            
            # Move to the next temperature
            self.lower_temperature()
