import numpy as np
import math
from scipy.optimize import linprog
from abc import ABC
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import pulp as pl
from utils import *


def generate_bit_data(n=1000):
    """
    Generate an array of n random bits.
    """
    return np.random.randint(0, 2, n)


def generate_int_data(n=1000, k=5):
    """
    Generate an array of n random numbers in [0...k].
    """
    return np.random.randint(0, k + 1, n)


class Mechanism(ABC):
    pass


class NoiseMechanism(Mechanism):
    """
    A mechanism whose queries return the true result plus noise generated by noise_generator.
    noise_generator takes 0 arguments and returns a value of the same type as query
    """

    def __init__(self, dataset, noise_generator):
        self.dataset = dataset
        self.length = len(dataset)
        self.noise_generator = noise_generator

    def get_len(self):
        return self.length

    def get_raw_data(self):
        return self.dataset

    def query(self, indices):
        result = np.sum(self.dataset[indices])
        noise = self.noise_generator()
        return result + noise


class TrivialMechanism(NoiseMechanism):
    """
    A noise mechanism that draws its noise from the Laplace(sensitivity/epsilon) distribution
    """

    def __init__(self, dataset):
        noise = lambda: 0
        super().__init__(dataset, noise)


class LaplaceMechanism(NoiseMechanism):
    """
    A noise mechanism that draws its noise from the Laplace(sensitivity/epsilon) distribution
    """

    def __init__(self, dataset, epsilon, sensitivity=1):
        self.scale = sensitivity / epsilon
        noise = lambda: np.random.laplace(loc=0, scale=self.scale, size=None)
        super().__init__(dataset, noise)


class GaussianMechanism(NoiseMechanism):
    """
    A noise mechanism that draws its noise from the Normal(2*(sensitivity^2)log(1/25/delta)/(epsilon^2))distribution
    """

    def __init__(self, dataset, epsilon, delta, sensitivity=1):
        self.scale = (2 * (sensitivity**2) * math.log(1.25 / delta)) / (epsilon**2)
        noise_generator = lambda: np.random.normal(loc=0, scale=self.scale, size=None)
        super().__init__(dataset, noise_generator)


def discover_accuracy(mech, num_queries=100):
    """
    Takes a mechanism and calculates the (nonnormalized) average error per randomly generated subset query.
    """
    baseline = TrivialMechanism(mech.get_raw_data())
    l = mech.get_len()

    def generate_query(n):
        # Generate a random subset of [n] by generating a list of bits and convert to indices
        q_j = np.random.randint(0, 2, n)
        indices2 = q_j.nonzero()
        return (q_j, indices2)

    running_total = 0
    for i in range(num_queries):
        q, _ = generate_query(l)
        true_value = baseline.query(q)
        mech_value = mech.query(q)
        running_total += abs(true_value - mech_value)

    return running_total / num_queries
