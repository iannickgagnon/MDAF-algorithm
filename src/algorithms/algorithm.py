
# External libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from inspect import signature
from typing import Any
import numpy as np


# Internal libraries
from src.algorithms.abstract_context import BaseContext


@dataclass
class Metrics:
    """
    Dataclass for storing metrics.
    """
    BEST: str = 'best'
    FIRST_HITTING_TIME: str = 'first hitting time'


@dataclass
class Profiles:
    """
    Dataclass for storing profiles.
    """
    BEST: str = 'best'
    VALUE: str = 'value'


def plot_fonts(foo):
    """
    Decorator to set the default font for plots.

    Args:
        foo (function): The function to be decorated.
    
    Returns:
        Any: The result of the undecorated function.
    """

    def wrapper(*args, **kwargs):
        
        # Store the current rcParams
        rcParams = plt.rcParams.copy()

        # Set the default parameters
        plt.rcParams['font.family'] = \
            kwargs.get('font.family', 'Times New Roman')
        
        plt.rcParams['font.size'] = \
            kwargs.get('font.size', 16)

        # Call the original function
        result = foo(*args, **kwargs)

        # Restore the original rcParams
        plt.rcParams.update(rcParams)

        return result
    
    return wrapper


class Algorithm(ABC):
    """
    Abstract base class for metaheuristic algorithms.
    """

    def __init__(self, 
                 context: BaseContext):
        """
        Initializes the Algorithm object.

        Args:
            context (AbstractContext): The context object for the algorithm.
        """
        
        # Store context
        self.context = context

        # Internalize metrics and pofiles constants
        self.metrics = Metrics()
        self.profiles = Profiles()


    def __getattr__(self, 
                    attribute_name: str,
                    *args) -> BaseContext:
        """
        Retrieves the attribute from the context object.

        Args:
            attribute_name (str): The name of the attribute.

        Returns:
            Any: The value of the attribute.
        """
        try:

            # Get the attribute from the context object
            attribute = getattr(self.context, attribute_name)
            
            # If it is callable and has only one parameter named 'context', return wrapper with context injected
            if callable(attribute):  
                parameters = list(signature(attribute).parameters.values())
                if parameters[0].name == 'context':
                    return lambda: attribute(self.context, *args)

            # Return the attribute from the context object
            return getattr(self.context, attribute_name)
        
        except AttributeError:

            # Return the attribute from the Algorithm object if it does not exist in the context object
            return getattr(self, attribute_name)
        

    def __setattr__(self, 
                    attribute_name: str, 
                    value: Any) -> None:
        """
        Sets the attribute value.

        Args:
            attribute_name (str): The name of the attribute.
            value (Any): The value to be set.
        """

        match attribute_name:
            
            case 'current_value' | 'best_solution' | 'best_value' | 'initial_value' | 'initial_solution':

                # Raise an error if trying to set the value directly
                raise AttributeError('\033[91mInitial, current and best values are updated automatically'
                                     ' when setting the current solution (self.current_solution).\033[0m')

            case 'context':

                # Set the attribute in the Algorithm object to avoir recursion in the subclass __setattr__ method
                super().__setattr__(attribute_name, value)
            
            case 'current_solution':  
            
                # Update current value
                self.context.current_solution = value
                self.context.current_value = self.context.objective(value)

                # Update histories  
                self.context.solution_history.append(self.context.current_solution)
                self.context.value_history.append(self.context.objective(self.current_solution))

                # Update best
                if self.context.current_value < self.context.best_value:
                    
                    # Update best solution and value
                    self.context.best_solution = self.current_solution
                    self.context.best_value = self.current_value
                    self.context.first_hitting_time = len(self.context.value_history) - 1

                # Always Update best histories
                self.context.best_solution_history.append(self.context.best_solution)
                self.context.best_value_history.append(self.context.best_value)
    
            case _:
        
                # Try to set the attribute in the context object, if it does not exist, set it in the Algorithm object
                try:
                    setattr(self.context, attribute_name, value)
                except AttributeError:
                    setattr(self, attribute_name, value)


    def initialize_solutions_and_values(self) -> None:
        """
        Initializes the following values for the algorithm:
            
            - Solutions and their associated values
            - Histories for solutions and values

        Args:
            None

        Returns:
            None
        """

        # Use initializer if present
        if self.context.solution_initializer is not None:
            self.context.solution_initializer(self.context)
        elif self.context.initial_solution is None:
            raise AttributeError('\033[91mPlease provide an initial solution (initial_solution: list | np.ndarray)'
                                 ' or an initializer function (solution_initializer: callable).\033[0m')

        # Determine the number of agents (e.g., particles in PSO)
        if self.context.nb_agents is None:

            # Try to infer the number of agents from the initial solution
            if isinstance(self.context.initial_solution, (int, float, list)):
                self.context.nb_agents = 1
            elif isinstance(self.context.initial_solution, np.ndarray):
                self.context.nb_agents = self.context.initial_solution.shape[0]
            else:
                raise TypeError('\033[91mInitial solution (initial_solution) must be a list, an int, a float, or a NumPy array.\033[0m')

            # Warn user
            print(f'\033[93mNumber of agents not provided. Automatically set to {self.context.nb_agents} based on initial solution size.\033[0m')

        # Initialize solutions and their values
        self.context.initial_value = self.objective(self.initial_solution)
        self.context.current_solution = self.initial_solution
        self.context.current_value = self.objective(self.initial_solution)
        self.context.best_solution = self.initial_solution
        self.context.best_value = self.objective(self.initial_solution) 
        
        # Initialize histories
        self.context.solution_history = [self.initial_solution]
        self.context.value_history = [self.initial_value]
        self.context.best_solution_history = [self.best_solution]
        self.context.best_value_history = [self.best_value]
        self.context.first_hitting_time = 0


    def initialize(self, context) -> None:
        """
        Initializes the Algorithm object by storing the context and initializing the solutions and values.

        Args:
            context (AbstractContext): The context object for the algorithm.
        
        Returns:
            None.
        """

        # Store the context
        Algorithm.__init__(self, context)

        # Initialize the solutions and values
        self.initialize_solutions_and_values()


    @plot_fonts
    def plot_profile(self,
                     metric: str,
                     nb_runs: int = 100,
                     nb_bootstrap_samples: int = 1000,
                     is_legend=False,
                     is_export_data=False) -> None | np.ndarray:
        """
        Generates convergence profiles with 95% bootstrap confidence intervals using multiple runs of the algorithm.

        Args:
            metric (str): The metric to generate the profile for (e.g., 'best value').
            nb_runs (int): The number of runs to perform. Default is 100.
            nb_bootstrap_samples (int): The number of bootstrap samples to generate. Default is 1000.

        Returns:
            None
        """

        # Validate
        assert metric in self.profiles.__dict__.values(), \
            f'\033[91mThe supplied metric (\'{metric}\') is not in {tuple(self.profiles.__dict__.values())}.\033[0m'

        # Initialize histories
        histories = []

        # Collect histories
        for _ in range(nb_runs):
            
            # Instantiate new algorithm
            algo = self.__class__(self.context.copy())

            # Run
            algo.run()

            # Dispatch
            if metric == self.profiles.BEST:
                histories.append(algo.best_value_history)
            elif metric == self.profiles.VALUE:
                histories.append(algo.value_history)

        # Find the minimum length among all best value histories
        min_length = min(len(history) for history in histories)

        # Slice all best value histories to the minimum length
        histories = np.array([curve[:min_length] for curve in histories])

        # Calculate means
        means = np.mean(histories, axis=0)

        # Bootstrap resampling
        bootstrap_results = bootstrap((histories,),
                                    np.mean,
                                    n_resamples=nb_bootstrap_samples,
                                    method='percentile')

        # Calculate the 95% confidence intervals
        lower_bound, upper_bound = bootstrap_results.confidence_interval
        
        '''
        Plot results
        '''

        # Shade the confidence interval
        plt.fill_between(range(len(means)), 
                         lower_bound, 
                         upper_bound, 
                         color='b', 
                         alpha=0.5, 
                         label=f'95% CI (n = {nb_runs})')
        
        # Plot 
        plt.plot(means, color='b', label='Mean')
        plt.ylim([min(lower_bound) * 0.9, max(upper_bound) * 1.1])

        # Show legend
        if is_legend:
            plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=2)
        
        # Add grid
        plt.grid(linestyle='--', alpha=0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        plt.show()

        # Export data if required
        if is_export_data:
            return histories


    def generate_metric_distribution(self,
                                     metric: str = 'best',
                                     sample_size: int = 100):
        """
        Generates a sample of the first hitting times or best solution values.

        Args:
            metric (str): The metric to generate the sample for (e.g., 'best'). Default is 'best'.
            sample_size (int): The number of samples to generate. Default is 100.

        Returns:
            np.array: The sample of first hitting times or best solution values, depending on the chosen metric.
        """

        # Validate
        assert metric in self.metrics.__dict__.values(), \
            f'\033[91mThe supplied metric (\'{metric}\') is not in {tuple(self.metrics.__dict__.values())}.\033[0m'

        # Initialize output
        metrics = []

        # Collect metrics
        for _ in range(sample_size):
                
                # Instantiate new algorithm
                algo = self.__class__(self.context.copy())
    
                # Run
                algo.run()
    
                # Collect
                metrics.append(algo.best_value if metric == 'best' else algo.first_hitting_time)

        return metrics


    def generate_statistic_distribution(self,
                                        foo_statistic: callable,
                                        metric: str = 'best',
                                        sample_size: int = 100,
                                        sub_sample_size: int = 100):
        """
        Generates a sample of a statistic calculated on metric (e.g., best value) distributions.

        Args:
            foo_statistic (callable): A function that calculates a statistic on the metric distribution.
            metric (str, optional): The type of metric to generate. Must be either 'best' or 'first hitting time'. Defaults to 'best'.
            sample_size (int, optional): The number of samples to generate. Defaults to 100.
            sub_sample_size (int, optional): The size of each sub-sample used to generate the metric distribution. Defaults to 100.

        Returns:
            list: A list of statistics calculated on the metric distribution.
        """

        # Validate
        assert metric in self.metrics.__dict__.values(), \
            f'\033[91mThe supplied metric (\'{metric}\') is not in {tuple(self.metrics.__dict__.values())}.\033[0m'
        
        # Initialize output
        statistics = []

        for _ in range(sample_size):
                
            # Generate metric distribution
            sample = self.generate_metric_distribution(metric, sub_sample_size)
    
            # Collect
            statistics.append(foo_statistic(sample))

        return statistics


    def generate_distribution(self,
                              metric: str,
                              foo_statistic: callable = None,
                              sample_size: int = 100,
                              sub_sample_size: int = 100):
        """
        Generates a sample of the first hitting times of the best solution.

        Args:
            metric (str): The metric to use for generating the sample. Must be either "best" or "first hitting time".
            foo_statistic (callable, optional): A custom statistic function to use for generating the sample. Default is None.
            sample_size (int, optional): The number of metric samples to generate. Default is 100.
            sub_sample_size (int, optional): The number of sub-samples to generate when using a statistic function. Default is 100.

        Returns:
            np.array: The sample of first hitting times.
        """

        # Make sure that the user does not provide both a metric and a custom statistic function
        if metric and foo_statistic:
            raise AttributeError('\033[91mPlease provide either a metric or a statistic function, not both.\033[0m')

        # Validate
        assert metric in self.metrics.__dict__.values(), \
            f'\033[91mThe supplied metric (\'{metric}\') is not in {tuple(self.metrics.__dict__.values())}.\033[0m'
        
        # Generate sample
        if foo_statistic is None:
        
            sample = self.generate_metric_distribution(metric, sample_size)
        
        else:

            sample = self.generate_statistic_distribution(metric=metric,
                                                          foo_statistic=foo_statistic, 
                                                          sample_size=sample_size, 
                                                          sub_sample_size=sub_sample_size)

        return sample


    @plot_fonts
    def plot_distribution(self,
                        metric: str,
                        foo_statistic: callable = None,
                        sample_size: int = 100,
                        sub_sample_size: int = 100,
                        nb_bootstraps: int = 1000,
                        is_legend: bool = False):
        """
        Generates a histogram of the median first hitting times of the best solution.

        Args:
            metric (str): The metric to use for generating the distribution. Must be either "best" or "first hitting time".
            foo_statistic (callable, optional): A function to calculate a custom statistic on the samples. Default is None.
            sample_size (int, optional): The size of the overall sample to generate. Default is 100.
            sub_sample_size (int, optional): The size of each sub-sample to generate. Default is 100.
            n_bootstraps (int, optional): The number of bootstrap samples to use for estimating error bars. Default is 1000.

        Returns:
            None
        """

        # Validate
        assert metric in self.metrics.__dict__.values(), \
            f'\033[91mThe supplied metric (\'{metric}\') is not in {tuple(self.metrics.__dict__.values())}.\033[0m'
        
        # Generate the sample
        sample = self.generate_distribution(metric=metric,
                                            foo_statistic=foo_statistic,
                                            sample_size=sample_size,
                                            sub_sample_size=sub_sample_size)

        # Calculate the histogram
        hist, bins = np.histogram(sample, bins=10, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Bootstrap estimate of the error bars
        bootstrap_samples = np.random.choice(sample, (nb_bootstraps, sample_size), replace=True)
        bootstrap_hist = np.array([np.histogram(bs, bins=bins, density=True)[0] for bs in bootstrap_samples])
        
        # Calculate 95% CI
        lower_bound = np.percentile(bootstrap_hist, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_hist, 97.5, axis=0)
        error = [hist - lower_bound, upper_bound - hist]
        
        '''
        Plot results
        '''

        # Plot the histogram
        plt.bar(bin_centers, 
                hist, 
                width=(bins[1] - bins[0]), 
                color='blue', 
                edgecolor='k', 
                alpha=0.5, 
                label='Rel. Frequency')
        
        # Add error bars
        plt.errorbar(bin_centers, 
                     hist, 
                     yerr=error, 
                     fmt='.', 
                     color='k', 
                     ecolor='black', 
                     capsize=3, 
                     label=f'95% CI (n = {sample_size})')

        plt.ylabel('Relative Frequency')
        if is_legend:
            plt.legend(bbox_to_anchor=(0.5, 1.25), 
                       loc='upper center', 
                       ncol=2)

        # Add grid
        plt.grid(linestyle='--', 
                 alpha=0.5)

        # Bring grid to the background
        plt.gca().set_axisbelow(True)

        # Adjust layout
        plt.tight_layout()
        
        plt.show()


    @abstractmethod
    def run(self) -> None:
        """
        Abstract method for running the algorithm.
        """
        pass
