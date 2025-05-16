from algorithms.implementations.particle_swarm_optimization import PSO, PSOContext
import numpy as np


if __name__ == "__main__":

    def solution_initializer(context):
        """
        Initializes the solution, velocities and personal / best positions and values.
        """

        # Initialize the positions of the particles
        context.initial_solution = np.random.uniform(
            context.lower_bound,
            context.upper_bound,
            (context.nb_agents, context.nb_dimensions),
        )

        # Initialize the velocities of the particles
        context.velocities = np.random.uniform(
            -1, 1, (context.nb_agents, context.nb_dimensions)
        )

        # Initialize personal / best positions and values
        context.personal_best_solution = context.initial_solution.copy()
        context.personal_best_value = context.objective(context.initial_solution)
        context.global_best_solution = context.personal_best_solution[
            np.argmin(context.personal_best_value)
        ]
        context.global_best_value = np.min(context.personal_best_value)

    def update_velocities(context):
        """
        Updates the velocity of the particles based on the canonical PSO equations.
        """

        # Generate random factors for cognitive and social velocities update
        r1 = np.random.rand(context.nb_agents, context.nb_dimensions)
        r2 = np.random.rand(context.nb_agents, context.nb_dimensions)

        # Update cognitive velocity
        cognitive_velocities = (
            context.cognitive_constant
            * r1
            * (context.personal_best_solution - context.current_solution)
        )

        # Update social velocity
        social_velocities = (
            context.social_constant
            * r2
            * (context.global_best_solution - context.current_solution)
        )

        # Update total velocity
        context.velocities = (
            context.inertia_weight * context.velocities
            + cognitive_velocities
            + social_velocities
        )

    def update_position(context):
        """
        Updates the positions and applies the boundaries.
        """

        # Update position
        context.current_solution += context.velocities

        # Enforce bounds
        context.current_solution = np.clip(
            context.current_solution, context.lower_bound, context.upper_bound
        )

        # Update values
        context.current_value = context.objective(context.current_solution)

    def update_particles(context):

        # Update particle histories
        personal_best_mask = context.current_value < context.personal_best_value
        context.personal_best_value[personal_best_mask] = context.current_value[
            personal_best_mask
        ]
        context.personal_best_solution[personal_best_mask] = context.current_solution[
            personal_best_mask
        ].copy()

        # Update global best
        global_best_mask = context.current_value < context.global_best_value
        if np.any(global_best_mask):
            context.global_best_value = np.min(context.current_value[global_best_mask])
            context.global_best_solution = context.current_solution[
                np.argmin(context.current_value)
            ]

    def terminate(context):
        """
        Indicates whether the algorithm should terminate based on number of iterations.
        """
        return context.iteration >= context.max_iterations

    def objective_function(x):
        """
        Multidimensional sphere function.
        """
        return np.sum(x**2, axis=1)

    # Example usage
    context = PSOContext(
        # Swarm size
        nb_agents=30,
        # Number of dimensions of the objective function
        nb_dimensions=2,
        # Bounds for each dimension
        lower_bound=[-10, -10],
        upper_bound=[10, 10],
        # PSO parameters
        inertia_weight=0.5,
        cognitive_constant=1.5,
        social_constant=1.5,
        # Maximum number of iterations
        iteration=0,
        max_iterations=100,
        # Algorithm components
        objective=objective_function,
        solution_initializer=solution_initializer,
        update_velocities=update_velocities,
        update_position=update_position,
        update_particles=update_particles,
        terminate=terminate,
    )

    # Instantiate PSO
    algo = PSO(context)

    # Run PSO on n-dimensional Sphere
    algo.run()

    algo.plot_profile(metric=algo.profiles.BEST, is_legend=True)

    # Print results
    print("Best solution:", algo.global_best_solution)
    print("Best solution value:", algo.global_best_value)
