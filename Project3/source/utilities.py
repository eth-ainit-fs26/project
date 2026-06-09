
"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import logging
import os
import datetime
import pytz
import re

import numpy as np
import torch
from typing import List, Literal
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import ticker

########################################
# Get_Logger
########################################
tz = pytz.timezone("Europe/Zurich")

def timetz(*args):
    return datetime.datetime.now(tz).timetuple()

def Get_Logger(SAVE_FOLDER_NAME):
    # make_dir
    #######################################################
    prefix = datetime.datetime.now(pytz.timezone("Europe/Zurich")).strftime("%Y%m%d_%H%M__")
    result_folder_no_postfix = "./result/{}".format(prefix + SAVE_FOLDER_NAME)

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)

    os.makedirs(result_folder_path)

    # Logger
    #######################################################
    logger = logging.getLogger(result_folder_path) 

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('{}/log.txt'.format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)

    return logger, result_folder_path

def Extract_from_LogFile(result_folder_path, variable_name):
    logfile_path = '{}/log.txt'.format(result_folder_path)
    with open(logfile_path) as f:
        datafile = f.readlines()
    found = False  # This isn't really necessary
    for line in reversed(datafile):
        if variable_name in line:
            found = True
            m = re.search(variable_name + '[^\n]+', line)
            break
    exec_command = "Print(No such variable found !!)"
    if found:
        return m.group(0)
    else:
        return exec_command

########################################
# Average_Meter
########################################
class Average_Meter:
 
    def __init__(self, device):
        self.device = device
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.sum = torch.tensor(0.).to(self.device)
        self.count = 0

    def push(self, some_tensor, n_for_rank_0_tensor=None):
        assert not some_tensor.requires_grad # You get Memory error, if you keep tensors with grad history
        
        rank = len(some_tensor.shape)

        if rank == 0: # assuming "already averaged" Tensor was pushed
            self.sum += some_tensor * n_for_rank_0_tensor
            self.count += n_for_rank_0_tensor
            
        else:
            self.sum += some_tensor.sum()
            self.count += some_tensor.numel()

    def peek(self):
        average = (self.sum / self.count).tolist()
        return average

    def result(self):
        average = (self.sum / self.count).tolist()
        self.reset()
        return average

########################################
# View NN Parameters
########################################

def get_n_params1(model):
    pp = 0
    for p in list(model.parameters()):
        nn_count = 1
        for s in list(p.size()):
            nn_count = nn_count * s
        pp += nn_count
        print(nn_count)
        print(p.shape)
    print("Total: {:d}".format(pp))


def get_n_params2(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def get_n_params3(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_structure(model):
    print(model)



#########################################
# Visualizations
##########################################
from .parameters import MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS

def visualize_solver_performance(solver_names: List[str], x_vals: List[float], costs: List[List[float]], times: List[List[float]], 
                                 n_instances_per_size: int, eval_seed: int, figsize=(20,4)):
    ''' Visualize the performance of multiple solvers on CVRP instances.
        Parameters:
            solver_names (List[str]): List of solver names
            xvals (List[float]): List of x-axis values (number of nodes including depot)
            costs (List[List[float]]): List of average costs per solver
            times (List[List[float]]): List of average solving times per solver
            n_instances_per_size (int): Number of instances evaluated per problem size
            eval_seed (int): Seed used for dataset generation
    '''
    costs, times, x_vals = np.array(costs), np.array(times), np.array(x_vals)-1

    def plot_absolute_costs(ax: plt.Axes):
        ''' Plot average costs for multiple solvers '''
        for i, solver_name in enumerate(solver_names):
            ax.plot(x_vals, costs[i], marker='o', label=solver_name, color=['b','y','r'][i])
        ax.set_xlabel('Number of Customers'), ax.set_ylabel(f'Average Solution Cost')
        ax.set_xticks(range(MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS + 1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.set_title('Average Solution Cost', fontsize=16)
        ax.legend(), ax.grid(True)

    def plot_absolute_times(ax: plt.Axes):
        for i, solver_name in enumerate(solver_names):
            ax.plot(x_vals, times[i], marker='o', label=solver_name, color=['b','y','r'][i])
        ax.set_xlabel('Number of Customers'), ax.set_ylabel(f'Average Solving Time (seconds)')
        ax.set_xticks(x_vals), ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.set_title('Average Solving Time', fontsize=16)
        ax.legend(), ax.grid(True)

    def plot_relative_metrics(ax: plt.Axes, kind=Literal['time', 'cost']):
        ''' Plot relative solution cost or solving times compared to a baseline solver, assumed to be the first solver '''
        baseline_metric = times[0] if kind=='time' else costs[0]
        assert (baseline_metric > 0).all(), f"Baseline {kind}s must be positive."
        for i, solver_name in enumerate(solver_names[1:]):
            metric = times[i+1] if kind=='time' else costs[i+1]
            ax.plot(x_vals, metric / baseline_metric, marker='o', label=f'{kind.capitalize()}: {solver_name}', color=['y','r'][i])
        ax.plot(x_vals, np.ones(len(x_vals)), '--', color='b', label='Baseline = 1', alpha=0.7) 
        ax.set_xticks(x_vals), ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.set_xlabel('Number of Customers'), ax.set_ylabel(f'Relative {kind.capitalize()} Performance (Baseline = 1)')
        ax.legend()
        ax.set_title(f'Average {kind.capitalize()} Ratio against Baseline', fontsize=16)
        ax.grid(True)

    def plot_relative_times_and_costs(ax: plt.Axes):
        ''' Plot relative solution cost and solving times compared to a baseline solver, assumed to be the first solver '''
        baseline_cost, baseline_time = costs[0], times[0]
        assert (baseline_cost > 0).all(), "Baseline costs must be positive."
        assert (baseline_time > 0).all(), "Baseline times must be positive."
        for i, solver_name in enumerate(solver_names[1:]):
            ax.plot(x_vals, costs[i+1] / baseline_cost, marker='o', label=f'Cost: {solver_name}')
        # ax.plot(x_vals, np.ones(len(x_vals)), '--', color='gray', label='Baseline = 1')  
        ax.set_xticks(x_vals), ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.set_xlabel('Number of Customers'), ax.set_ylabel(f'Relative Cost Performance (Baseline = 1)')

        ax2 = ax.twinx()
        for i, solver_name in enumerate(solver_names[1:]):
            ax2.plot(x_vals, times[i+1] / baseline_time, marker='x', label=f'Time: {solver_name}')
        # ax2.plot(x_vals, np.ones(len(x_vals)), '--', color='gray', label='Baseline = 1')
        ax2.set_ylabel(f'Relative Time Performance (Baseline = 1)')
        ax.legend(loc='upper left'), ax2.legend(loc='upper right')
        ax.set_title('Relative Solution Cost and Solving Time (Ratio against Baseline)', fontsize=16)
        ax.grid(True)

    assert len(solver_names) == len(costs) == len(times), "Solver names, costs, and times length mismatch."
    if len(solver_names)==1: # only one solver, plot only absolute costs and times
        fig, ax = plt.subplots(1,2, figsize=figsize)
        plot_absolute_costs(ax[0]) # plot average costs
        plot_absolute_times(ax[1]) # plot average solving times
        plt.suptitle(f'Solver Performance on {n_instances_per_size} random CVRP instances (seed={eval_seed})', fontsize=22)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(2,2, figsize=figsize)
        plot_absolute_costs(ax[0][0]) # plot average costs
        plot_absolute_times(ax[0][1]) # plot average solving times
        plot_relative_metrics(ax[1][0], kind='cost') # plot relative costs
        plot_relative_metrics(ax[1][1], kind='time') # plot relative times  
        plt.suptitle(f'Solver Performance on {n_instances_per_size} random CVRP instances (seed={eval_seed})', fontsize=22)
        fig.tight_layout()
    
    return fig

#########################################
# Solution format conversion
##########################################
def convert_tour_to_routes(nodes: torch.LongTensor) -> List[List[int]]:
    """
    Convert a sequence of visited nodes (including depots) into a list of routes.
    Each route is a list of nodes starting and ending with the depot (0).
    Consecutive zeros indicate the end of one route and the start of another.
    Trailing zeros that do not indicate additional routes are removed.
    Parameters:
    nodes: list of node indices representing the sequence of visited nodes in the solution, e.g. [0, 3, 2, 0, 1, 0, 0]
    Returns:
    routes: list of routes, where each route is a list of node indices starting and ending with 0, e.g. [[0, 3, 2, 0], [0, 1, 0]]
    """
    nodes = nodes.tolist()
    while len(nodes) > 1 and nodes[-1] == 0 and nodes[-2] == 0:
        nodes.pop()
    routes = []
    current_route = []

    for node in nodes:
        current_route.append(node)
        if node == 0:
            # If we hit a depot and the current route has more than just the starting 0, close it
            if len(current_route) > 1:
                routes.append(current_route)
            current_route = [0] # Start the next route with the depot

    return routes