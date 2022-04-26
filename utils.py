import numpy as np
import math

# Utilities
def percent_reconstructed(data, results):
    """
    What percentage of entries in data match those in results.
    """
    n = len(data)
    return 1 - ((np.count_nonzero(data != results)) / n)


class ConstantFunction:
    # This should not be a class but it did something else before
    def __init__(self, val):
        self.val = val

    def __call__(self, _x):
        return self.val
