import matplotlib.pyplot as plt


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
        plt.rcParams["font.family"] = kwargs.get("font.family", "Times New Roman")

        plt.rcParams["font.size"] = kwargs.get("font.size", 16)

        # Call the original function
        result = foo(*args, **kwargs)

        # Restore the original rcParams
        plt.rcParams.update(rcParams)

        return result

    return wrapper
