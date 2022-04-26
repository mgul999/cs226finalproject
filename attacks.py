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
import pulp as plt
from mechanisms import *
from utils import *


def dinur_nissim(
    mech,
    gen_t=lambda n: n * (math.log2(n) ** 2),
    epsilon=0.5,
    should_log=False,
    **kwargs,
):
    """
    Perform a dinur_nissim attack on the dataset, mediated by mech, a Mechanism object.

    epsilon and t are the two hyperparameters of the attack. gen_t generates int-valued t as a function of n, the size of the dataset.

    Wrap all logging behond `should_log`
    """

    n = mech.get_len()
    t = int(gen_t(n))

    def generate_query():
        # Generate a random subset of [n] by generating a list of bits and convert to indices
        q_j = np.random.randint(0, 2, n)
        indices2 = q_j.nonzero()
        return (q_j, indices2)

    # Make t queries to the mechanism. A_ub and b_ub are respectively the coefficient matrix
    #   and upper bound for the linear program
    A_ub = []
    b_ub = []
    for _ in range(int(t)):
        q, indices = generate_query()
        answer = mech.query(indices)
        A_ub.append(q)
        b_ub.append(answer + epsilon)
        A_ub.append(-1 * q)
        b_ub.append(epsilon - answer)
    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)
    # Rounding Phase: round the results of the LP into integers

    if should_log:
        print(f"Solving LP with {len(b_ub)} constraints")

    program_result = linprog(
        np.zeros(n),
        np.vstack(A_ub),
        b_ub,
        bounds=(0, 1),
        options={"maxiter": 3000000000},
    )

    if should_log:
        print(f"Done w/ LP")
    round_vector = np.vectorize(round, otypes=[int])
    return round_vector(program_result["x"])


def cn19(
    mech, gen_t=lambda n: n * (math.log2(n) ** 2), k=1, should_log=False, **kwargs
):
    """
    Performs the attack from CN19 on the dataset, mediated by mech, a Mechanism object.

    k and t are the two hyperparameters of the attack.

    gen_t generates int-valued t as a function of n, the size of the dataset.

    k is the number such that the data take values in [0...k].

    Wrap all logging behond `should_log`
    """

    n = mech.get_len()
    t = int(gen_t(n))

    def generate_query():
        # Generate a random subset of [n] by generating a list of bits and convert to indices
        q_j = np.random.randint(0, 2, n)
        indices2 = q_j.nonzero()
        return (q_j, indices2)

    # Set up the LP using pulp
    lp = pl.LpProblem("Decoding", pl.LpMinimize)

    # x_i contains the i'th entry in the candidate database
    xs = [pl.LpVariable(str(i), 0, k) for i in range(n)]
    # abs_e_qs stores the absolute values of the e_q, the difference between the answer and q(x')
    abs_e_qs = []
    # Generate variables based on the query
    for i in range(t):
        q, indices = generate_query()
        answer = mech.query(indices)
        # Get q(x') by evaluating our query on the candidate database
        q_x = [(xs[j], q[j]) for j in range(n)]
        abs_e_q = pl.LpVariable(("abs_e_q" + str(i)))
        abs_e_qs.append(abs_e_q)
        # Add a variable to represent the sum of a_q - q(x')
        temp_sum = pl.LpVariable("temp_sum" + str(i))
        # Add the constraints such that abs_e_qs will become the absolute value of a_q - q(x')
        lp += temp_sum == (answer - pl.LpAffineExpression(q_x))
        lp += abs_e_q >= temp_sum
        lp += abs_e_q >= -temp_sum

    # We want our objective to be the sum of |e_q| over all queries
    lp += pl.lpSum(abs_e_qs)

    if should_log:
        print(f"Solving LP with {len(b_ub)} constraints")
    status = lp.solve(pl.PULP_CBC_CMD(msg=False))

    if should_log:
        print(f"Done w/ LP")
    # Return the rounded values
    return [round(pl.value(x)) for x in xs]
