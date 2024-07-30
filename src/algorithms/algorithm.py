
# External libraries
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any

# Internal libraries
from src.algorithms.abstract_context import AbstractContext


class Algorithm(ABC):
    """
    Abstract base class for metaheuristic algorithms.
    """

    def __init__(self, 
                 context: AbstractContext):
        """
        Initializes the Algorithm object.

        Args:
            context (AbstractContext): The context object for the algorithm.
        """
        self.context = context

    def __getattr__(self, 
                    attribute_name: str) -> AbstractContext:
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
                if len(parameters) == 1 and parameters[0].name == 'context':
                    return lambda: attribute(self.context)

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


        if attribute_name == 'context':
            
            # To avoid infinite recursion, set the attribute in the Algorithm object to circumvent __setattr__ method
            super().__setattr__(attribute_name, value)
        
        else:
        
            # Try to set the attribute in the context object, if it does not exist, set it in the Algorithm object
            try:
                setattr(self.context, attribute_name, value)
            except AttributeError:
                setattr(self, attribute_name, value)

    @abstractmethod
    def run(self) -> None:
        """
        Abstract method for running the algorithm.
        """
        pass
