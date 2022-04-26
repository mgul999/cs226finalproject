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
from run_experiments import *

# Some preliminary specific plots


def plot_3d_eps_t(df, title="% Recon vs. T and Epsilon"):

    plot_3d(
        df,
        filter_cond=lambda x: x,
        x1="num_queries",
        x2="epsilon",
        y="percent_reconstructed",
        labels=["Num Queries", "Epsilon", "Proportion Reconstructed"],
        title=title,
    )


def plot_eps_multiple_t(df, ts, title="Success vs. Eps for various Ts"):
    plot_single_x_multiple_subsets(
        df=df,
        filter_cond=lambda x: x,
        x="epsilon",
        y="percent_reconstructed",
        divide_on="num_queries",
        dividers=ts,
        xlabel="Epsilon",
        ylabel="% Reconstructed",
        legend_formatter=lambda x: f"{x} queries",
        title=title,
    )


def plot_t_multiple_eps(df, epss, title="Success vs. T for various Epsilons"):
    plot_single_x_multiple_subsets(
        df=df,
        filter_cond=lambda x: x,
        x="num_queries",
        y="percent_reconstructed",
        divide_on="epsilon",
        dividers=epss,
        xlabel="# Queries",
        ylabel="% Reconstructed",
        legend_formatter=lambda x: f"Epsilon = {x}",
        title=title,
    )


def epsilon_thresshold(df, threshold=0.95, title="Queries vs. Epsilon"):
    """
    Plot # queries required to attain a propoertion reconstructed >- thresshold
    """
    tres = (
        df[df.percent_reconstructed >= threshold]
        .groupby(df.epsilon)
        .apply(lambda d: d[d.num_queries == d.num_queries.min()])
    )

    return plot_single_x_multiple_subsets(
        df=tres,
        filter_cond=lambda x: x,
        x="epsilon",
        y="num_queries",
        divide_on=None,
        xlabel="Epsilon",
        ylabel="Queries Required",
        title=title,
    )


def recon_accuracy_tradeoff(df, title="Accuracy and Percent Recon vs. Epsilon"):
    """
    Makes two plots:

    Both have epsilon on the x axis.
    First has mechanism error ("accuracy" proxy) on y axis
    Second has % reconstructed.

    Tweak the accuracy metric defined in mechanisms.py to improve this.
    """
    acc = plot_single_x_multiple_subsets(
        df=df,
        filter_cond=lambda x: x,
        x="epsilon",
        y="mechanism_error",
        divide_on=None,
        xlabel="Epsilon",
        ylabel="Mechanism Error",
        legend_formatter=lambda x: f"Epsilon = {x}",
        title=title,
    )

    perc_rec = plot_single_x_multiple_subsets(
        df=df,
        filter_cond=lambda x: x,
        x="epsilon",
        y="percent_reconstructed",
        divide_on=None,
        xlabel="Epsilon",
        ylabel="% Reconstructed",
        legend_formatter=lambda x: f"Epsilon = {x}",
        title="",
    )
