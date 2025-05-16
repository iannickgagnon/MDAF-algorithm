from dataclasses import dataclass

from algorithms.context import AlgorithmContext
from algorithms.algorithm import Algorithm


@dataclass
class SimulatedAnnealingContext(AlgorithmContext):
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

    temperature_schedule: list = None
    temperature_index: int = 0
    current_temperature: int = None
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

    def __init__(self, context: SimulatedAnnealingContext):

        # Initialize the context, solutions, and values
        self.initialize(context)

        # Initialize temperature
        self.current_temperature = self.context.temperature_schedule[
            self.context.temperature_index
        ]

    def run(self):
        """
        Run the Simulated Annealing algorithm.

        Args:
            None.

        Returns:
            tuple[np.ndarray, float]: The best solution found by the algorithm and its value.
        """

        # Main loop
        while not self.terminate():

            # Generate neighbor solution
            self.neighbor = self.generate_neighbor()

            # Evaluate neighbor solution
            self.neighbor_value = self.objective(self.neighbor)

            # Probabilistic acceptance
            if self.accept():
                self.current_solution = self.neighbor

            # Move along temperature schedule
            self.lower_temperature()
