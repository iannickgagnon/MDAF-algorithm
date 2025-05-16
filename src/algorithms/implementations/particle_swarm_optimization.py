from dataclasses import dataclass

from algorithms.context import AlgorithmContext
from algorithms.algorithm import Algorithm


@dataclass
class PSOContext(AlgorithmContext):
    """
    Represents the context for the Particle Swarm Optimization (PSO) algorithm.

    Attributes:
        nb_agents (int): The number of particles in the swarm.
        nb_dimensions (int): The number of dimensions in the search space.
        lower_bound (list): The lower bounds for each dimension.
        upper_bound (list): The upper bounds for each dimension.
        inertia_weight (float): The inertia weight used in the velocity update.
        cognitive_constant (float): The cognitive constant used in the velocity update.
        social_constant (float): The social constant used in the velocity update.
        max_iterations (int): The maximum number of iterations.
        positions (list): The current positions of the particles.
        velocities (list): The current velocities of the particles.
        personal_best_solution (list): The personal best positions of the particles.
        personal_best_value (list): The personal best values of the particles.
        global_best_solution (list): The global best position found by the swarm.
        global_best_value (float): The value of the global best position.
        objective (callable): The objective function used to evaluate solutions.
        initialize_particles (callable): The function used to initialize particles.
        update_velocities (callable): The function used to update particle velocities.
        update_position (callable): The function used to update particle positions.
        update_particles (callable): The function used to evaluate particles.
        terminate (callable): The function used to determine whether to terminate the algorithm.
    """

    iteration: int = None
    nb_agents: int = None
    nb_dimensions: int = None
    lower_bound: list = None
    upper_bound: list = None
    inertia_weight: float = None
    cognitive_constant: float = None
    social_constant: float = None
    initial_solution: list = None
    max_iterations: int = None
    positions: list = None
    velocities: list = None
    personal_best_solution: list = None
    personal_best_value: list = None
    global_best_solution: list = None
    global_best_value: float = float("inf")

    solution_initializer: callable = None
    objective: callable = None
    initialize_particles: callable = None
    update_velocities: callable = None
    update_position: callable = None
    update_particles: callable = None
    terminate: callable = None


class PSO(Algorithm):
    """
    Modular implementation of the Particle Swarm Optimization (PSO) algorithm.

    Args:
        context (PSOContext): The context object containing the algorithm parameters.

    Attributes:
        positions (np.ndarray): The current positions of the particles.
        velocities (np.ndarray): The current velocities of the particles.
        personal_best_solution (np.ndarray): The personal best positions of the particles.
        personal_best_value (np.ndarray): The personal best values of the particles.
        global_best_solution (np.ndarray): The global best position found by the swarm.
        global_best_value (float): The value of the global best position.
    """

    def __init__(self, context: PSOContext):
        self.initialize(context)

    def run(self):
        """
        Run the Particle Swarm Optimization (PSO) algorithm.

        Args:
            None.

        Returns:
            tuple[np.ndarray, float]: The best solution found by the algorithm and its value.
        """

        while not self.terminate():

            self.update_velocities()

            self.update_position()

            self.update_particles()

            self.context.iteration += 1
