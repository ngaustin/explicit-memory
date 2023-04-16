"""Matplotlib functions."""
import logging
import os
from copy import deepcopy
from glob import glob
from typing import List, Tuple

# import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import room_env
import torch
# from room_env.utils import get_handcrafted
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# from train import DQNLightning, RLAgent
# from utils import read_yaml

plot_dir = "./plotting_data/"
kinds = ["1", "2", "4", "8", "16", "32", "1f", "2f", "4f", "8f", "16f", "32f"] # capacity (with f means with filter 32f)
epoch = 16

num = 0
# cap_means = np.array([0] * int(len(kinds)/2))
# cap_stds = np.array([0] * int(len(kinds)/2))
# cap_means_f = np.array([0] * int(len(kinds)/2))
# cap_stds_f = np.array([0] * int(len(kinds)/2))
results = {}

for kind in kinds:
    reward_wrt_cap_path = glob(
            os.path.join(
                plot_dir,
                kind,
                "test_debug-mean=*"
            )
        )
    if len(reward_wrt_cap_path):
        reward_wrt_cap_path = reward_wrt_cap_path[0]
        num += 1
        cap_reward = float(reward_wrt_cap_path[reward_wrt_cap_path.find('-mean=') + 6 : reward_wrt_cap_path.find('-std')])   
        cap_std = float(reward_wrt_cap_path[reward_wrt_cap_path.find('-std') + 5 : -5]) 
        if kind[-1] == "f":
            results[kind[:-1]].append((cap_reward, cap_std))
        else:
            results[kind] = [(cap_reward, cap_std)]



    paths = glob(
            os.path.join(
                plot_dir,
                kind,
                "val_debug_episode=*"
            )
        )

    data_dict = {}

    if len(paths):
        steps = np.array([i * 128 for i in range(epoch)])
        means = np.array([0] * epoch)
        stds = np.array([0] * epoch)

        for path in paths:
            episode = int(path[path.find('episode') + 8 : path.find('-mean')])
            mean = float(path[path.find('-mean') + 6 : path.find('-std')])
            std = float(path[path.find('-std') + 5 : -5])
            means[episode] = mean
            stds[episode] = std

        title = "Avg. total rewards, validation."
        xlabel = "Step"
        ylabel = "Avg. total rewards"

        plt.figure()
        plt.plot(steps, means, color="pink")
        plt.fill_between(
            steps,
            means - stds,
            means + stds,
            alpha=0.2,
            edgecolor="pink",
            facecolor="pink",
            label="_nolegend_",
        )  

        plt.xticks()
        plt.yticks()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig("./new_figures/" + kind + "_reward.png")


if num == len(kinds):
    width = 0.17
    idx = np.asanyarray([i for i in range(len(results))])
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots()
    title = "Avg. total rewards, varying capacities, test."
    xlabel = kinds[:int(len(kinds)/2)]
    ylabel = "Avg. total rewards"

    idx = np.asanyarray([i for i in range(len(results))])

    fig, ax = plt.subplots()

    legend_order = [
        "Without filer",
        "With filter",
    ]

    color_order = ["orange", "dodgerblue"]

    for i, w, color in zip([0,1], [-0.5, 0.5], color_order):
        height = [results_[i][0] for _, results_ in results.items()]
        yerr = [results_[i][1] for _, results_ in results.items()]
        ax.bar(
            x=idx + w * width,
            height=height,
            yerr=yerr,
            width=width,
            color=color,
            capsize=4,
        )

    ax.set_xticks(idx)
    ax.set_xticklabels(list(results.keys()))
    plt.xticks(0)
    plt.yticks()
    ax.set_ylim([0, 128])
    ax.legend(legend_order)
    ax.set_xlabel("Memory capacity")
    ax.set_ylabel(ylabel)
    plt.title(title)

    plt.savefig("./new_figures/vary_capacity.png")    
    