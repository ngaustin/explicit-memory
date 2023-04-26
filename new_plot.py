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

plot_dir = "./plotting_data/"
kinds = ["4_bidirectional", "8_bidirectional", "16_bidirectional"]# ["2_filter", "4_filter", "8_filter", "16_filter", "32_filter"]# ["vary_steps_0", "vary_steps_512", "vary_steps_1024", "vary_steps_1536"]# ["1", "2", "4", "8", "16", "32"] # capacity (with f means with filter 32f)
epoch = 16

num = 0
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
        plt.ylim((0, 128))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig("./new_figures/" + kind + "_reward.png")


print(num)
if num == len(kinds):
    # width for with/without filter is 0.25, for without only is 0.4
    # width = 0.25
    width = 0.4
    idx = np.asanyarray([i for i in range(len(results))])
    fig, ax = plt.subplots()
    

    ############## EDIT HERE ##################
    title = "Avg. total rewards, varying memory size with new model, test."
    
    xlabel = [4, 8, 16]# kinds[:int(len(kinds)/2)]
    ylabel = "Avg. total rewards"
    ###########################################

    idx = np.asanyarray([i for i in range(len(results))])

    fig, ax = plt.subplots()

    legend_order = [
        "Without filer",
        "With filter",
    ]

    color_order = ["orange", "dodgerblue"]

    ## Code for with/without filter starts
    # for i, w, color in zip([0,1], [-0.53, 0.53], color_order):
    #     height = [results_[i][0] + 128 for _, results_ in results.items()]
    #     yerr = [results_[i][1] for _, results_ in results.items()]
    #     ax.bar(
    #         x=idx + w * width,
    #         height=height,
    #         yerr=yerr,
    #         width=width,
    #         color=color,
    #         capsize=4,
    #         bottom=-128
    #     )
    ## Code for with/without filter ends
    
    # Code for without filter only starts
    height = [results_[0][0] + 128 for _, results_ in results.items()]
    yerr = [results_[0][1] for _, results_ in results.items()]
    ax.bar(
        x=idx,
        height=height,
        yerr=yerr,
        width=width,
        color="orange",
        capsize=4,
        bottom=-128
    )
    # Code for without filter only ends


    ax.set_xticks(idx)
    ax.set_xticklabels(xlabel)
    plt.xticks()
    plt.yticks()
    ax.set_ylim([0, 128])
    # ax.legend(legend_order)
    ax.set_xlabel("Memory capacity")
    ax.set_ylabel(ylabel)
    plt.title(title)

    plt.savefig("./new_figures/vary_capacity.png")    
    