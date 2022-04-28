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
from mechanisms import *
from utils import *
from attacks import *

default_dn_args = {"epsilon": 0.5, "should_log": False}

# Run a set of experiments. See usage below.
def run_experiment(
    ns,
    mechanism_class,
    mech_args,
    dn_epsilon,
    t_generators,
    should_log=True,
    output_filename=None,
    attacker=dinur_nissim,
    data_maker=generate_bit_data,
    attacker_args=default_dn_args,
    trials = 1
):

    results_df = pd.DataFrame(
        columns=[
            "trial",
            "n",
            *list(mech_args[0].keys()),
            "dn_epsilon",
            "num_queries",
            "percent_reconstructed",
            "mechanism_error",
        ]
    )

    def make_entry(trial, mech_arg, n, dn_epsilon, t, pc_reconstructed, mechanism_error):
        res = {x: [mech_arg[x]] for x in mech_arg.keys()}
        res.update(
            {
                "trial" : trial,
                "n": [n],
                "dn_epsilon": [dn_epsilon],
                "num_queries": [t],
                "percent_reconstructed": pc_reconstructed,
                "mechanism_error": mechanism_error,
            }
        )
        return pd.DataFrame.from_dict(res)

    total = trials * len(ns) * len(mech_args) * len(t_generators)


    if should_log:
        print(f"Running {total} experiments")
    so_far = 0

    for trial in range(trials):
        for n in ns:
            for mech_arg in mech_args:
                # New dataset and mech for each size/mechanism params pair
                data = data_maker(n=n)
                mech = mechanism_class(data, **mech_arg)
                mech_acc = discover_accuracy(mech)
                for t_generator in t_generators:
                    attacker_args["epsilon"] = dn_epsilon
                    result = attacker(mech, t_generator, **attacker_args)
                    # Lambdas are hard to log so we log its application. We can probably abstract this better.
                    t = int(t_generator(n))

                    # Append new results
                    entry = make_entry(
                        trial,
                        mech_arg,
                        n,
                        dn_epsilon,
                        t,
                        percent_reconstructed(data, result),
                        mech_acc,
                    )
                    results_df = pd.concat([results_df, entry], ignore_index=True)
                    so_far += 1
                    if should_log:
                        print(f"Finished {so_far}/{total}")

    if output_filename:
        if should_log:
            print(f"Writing results to {output_filename}")
        results_df.to_csv("./data/" + output_filename, encoding="utf-8", header="true")
    return results_df


def read_results(filename):
    "Read results written by the function above"
    return pd.read_csv(filename, index_col=0)


def gensym():
    gensym.gensym_counter += 1
    return gensym.gensym_counter

gensym.gensym_counter = 0


# General plotting functions
def plot_3d(df, filter_cond, x1, x2, y, labels, title):
    """
    Make a 3d surface plot with y as a function of x1, x2.
    x1,x2,y should be df series / arrays
    """
    df = filter_cond(df)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(df[x1], df[x2], df[y], cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel(labels[0], fontsize=10)
    ax.set_ylabel(labels[1], fontsize=10)
    ax.set_zlabel(labels[2], fontsize=10)
    ax.set_title(title, fontsize=15)
    plt.savefig(f"newest{gensym()}.pdf")
    plt.show()
    return ax

def average_stats(df):
    cols = list(df.columns)
    cols.remove("trial")
    cols.remove("percent_reconstructed")
    cols.remove("mechanism_error")
    return df.groupby(cols).mean().reset_index()


def plot_single_x_multiple_subsets(
    df,
    filter_cond,
    x,
    y,
    divide_on,
    dividers=None,
    xlabel="x",
    ylabel="y",
    legend_formatter=lambda n: n,
    title="My graph",
):
    """
    Todo: Explain this
    """
    fig = plt.figure()
    ax = plt.gca()
    if divide_on is None:
        s = df
        ax.plot(s[x], s[y])
    else:
        for div in dividers:
            s = df[df[divide_on] == div]
            ax.plot(s[x], s[y], label=legend_formatter(div))
        ax.legend()
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f"newest{gensym()}.pdf")
    return ax
