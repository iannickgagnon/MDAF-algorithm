
# External libraries
from abc import ABC


class AbstractContext(ABC):
    
    def copy(self):
        
        # Create a new instance of the same class
        new_instance = type(self)()
        
        # Copy the attributes of the current instance to the new instance
        new_instance.__dict__.update(self.__dict__)
        
        return new_instance
    
    pass
