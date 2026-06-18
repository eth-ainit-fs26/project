"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon
Copyright (c) 2026 Department of Computer Science, ETH Zurich

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

import torch
import numpy as np
from tqdm import tqdm
import time
from IPython.core.debugger import set_trace # for debugging

from source.utilities import Average_Meter
from source.cvrp import DATALOADER, GROUP_ENVIRONMENT, compute_batched_features
from source.grouped_actor import ACTOR
DEVICE = None # to be set in the notebook before using

########################################
# EVAL
########################################

eval_result = []

def EVAL(grouped_actor: ACTOR,
         generator, 
         epoch: int, 
         timer_start, 
         logger,
         TEST_DATASET_SIZE: int,
         TEST_BATCH_SIZE: int,
         problem_sizes_mean=None,
         problem_sizes_std=None,
         rng=None):

    global eval_result
    grouped_actor.eval()
    device = grouped_actor.device

    eval_AM = Average_Meter(device)
    test_loader = DATALOADER(generator=generator,
                             num_sample=TEST_DATASET_SIZE,
                             batch_size=TEST_BATCH_SIZE,
                             problem_sizes_mean=problem_sizes_mean,
                             problem_sizes_std=problem_sizes_std,
                             rng=rng)

    with torch.no_grad():
        for demands, cost_matrix in test_loader:
            # demands.shape = (batch, problem+1, 1)
            # cost_matrix.shape = (batch, problem+1, problem+1)
            batch_s = demands.size(0)
            group_s = demands.size(1) - 1  # = problem size
            # move data to device
            demands = demands.to(device)
            cost_matrix = cost_matrix.to(device)
            features = compute_batched_features(generator, cost_matrix, demands) # already on device

            env = GROUP_ENVIRONMENT(demands, features, cost_matrix)
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state, env)

            # First Move is given
            first_action = torch.zeros((batch_s, group_s), dtype=torch.long, device=device)
            group_state, reward, done = env.step(first_action)

            # Second Move is given
            second_action = torch.arange(1, group_s+1, dtype=torch.long, device=device)[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(second_action)

            while not done:
                action_probs = grouped_actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem+1)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)

            max_reward, _ = reward.max(dim=1)
            eval_AM.push(-max_reward)  # reward was given as negative cost


    # LOGGING
    cost_avg = eval_AM.result()
    eval_result.append(cost_avg)


    logger.info('--------------------------------------------------------------------------')
    log_str = '  <<< EVAL after Epoch:{:03d} >>>   AvgCost:{:f}'.format(epoch, cost_avg)
    logger.info(log_str)
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')


def evaluate_actor(grouped_actor: ACTOR, dataloader, generator):
    ''' Evaluate the trained grouped actor on CVRP test dataset.
    Parameters:
        grouped_actor: trained ACTOR instance to be evaluated (defined in notebook)
        dataloader: DATALOADER instance that provides the test dataset
        generator: CVRPGenerator instance used to create the test dataset
    Returns:
        cost: costs per batch, shape = (num_batches, batch_size)
        total_time: total solving times per batch, shape = (num_batches,)
    '''
    cost = []
    total_time = [] 
    device = grouped_actor.device
    grouped_actor.eval() # set the actor to evaluation mode

    with torch.no_grad():
        for demands, cost_matrix in tqdm(dataloader, desc="Evaluating POMO Actor"):
            start_time = time.time()
            batch_s = demands.size(0) # batch size
            group_s = demands.size(1)-1  # problem size = n_cust

            # Move tensors to the correct device for evaluation
            demands = demands.to(device)
            cost_matrix = cost_matrix.to(device)
            # Compute features, shape = (batch, problem+1, NODE_FEATURE_DIM)
            features = compute_batched_features(generator, cost_matrix, demands) # already on device
            
            # Step 0
            env = GROUP_ENVIRONMENT(demands, features, cost_matrix)
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state, env)
            # Steps 1 and 2
            first_action = torch.zeros((batch_s, group_s), dtype=torch.long, device=device)
            group_state, reward, done = env.step(first_action)
            second_action = torch.arange(1, group_s+1, dtype=torch.long, device=device)[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(second_action)
            # Subsequent Steps
            while not done:
                action_probs = grouped_actor.get_action_probabilities(group_state) # shape = (batch, group, problem+1)
                action = action_probs.argmax(dim=2) # shape = (batch, group)
                action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action) # reward shape = (batch, group)
            
            max_reward, _ = reward.max(dim=1) # best rollouts; shape = (batch,)
            min_cost = -max_reward.to('cpu')
            cost.append(min_cost) # cost per batch
            end_time = time.time()
            total_time.append(end_time - start_time) # time lapse for this batch
    return np.array(cost), np.array(total_time)

# Benchmark actor performance against OR-Tools solver
from .utils_or_tools import evaluate_baseline_solver_using_dataloader

def evaluate_both_solvers(actor, n_instances_per_size, p_sizes, generator, seed):
    '''Solve instances using both the actor and OR tools with the same dataloader.
    Parameters:
        actor: ACTOR instance to be evaluated
        n_instances_per_size (int): Number of instances to solve per problem size
        p_sizes (list): List of problem sizes to evaluate
        generator: CVRPGenerator instance used to create the test dataset
        seed: random seed for reproducibility
    Returns:
        ortools_stats: Array of shape (4, p_sizes) containing average costs, first quartiles, third quartiles, and solving times for the OR-Tools solver
        actor_stats: Array of shape (4, p_sizes) containing average costs, first quartiles, third quartiles, and solving times for the actor
    '''
    num_samples = n_instances_per_size * len(p_sizes)

    # Evaluate the OR-Tools solver on the test dataset
    test_loader = DATALOADER(
        generator=generator,
        num_sample=num_samples,
        batch_size=n_instances_per_size,
        problem_sizes_mean=p_sizes,
        problem_sizes_std=0,
        rng=np.random.default_rng(seed)
    )
    or_costs, or_times = evaluate_baseline_solver_using_dataloader(test_loader)
    # shapes: (num_batches, batch_size) for costs, (num_batches,) for times

    # Evaluate the actor on the test dataset
    test_loader = DATALOADER(
        generator=generator,
        num_sample=num_samples,
        batch_size=n_instances_per_size,
        problem_sizes_mean=p_sizes,
        problem_sizes_std=0,
        rng=np.random.default_rng(seed)
    )
    actor_costs, actor_times = evaluate_actor(actor, test_loader, generator)
    # shapes: (num_batches, batch_size) for costs, (num_batches,) for times

    return or_costs, or_times, actor_costs, actor_times


from source.utilities import convert_tour_to_routes
from source import map_utils
from source.utils_or_tools import solve_with_ortools
def compare_POMO_with_baseline(solve_with_POMO_actor: callable, trained_actor, p_size: int, seed: int, 
                               G_utm, generator, fname='pomo_vs_baseline.html'):

    test_loader = DATALOADER(generator=generator,
                             num_sample=1, batch_size=1,
                             problem_sizes_mean=p_size, problem_sizes_std=0, 
                             rng=np.random.default_rng(seed), return_edges=True
    )
    demands, cost_matrix, edge_index, t = list(test_loader)[0]
    pomo_sol = solve_with_POMO_actor(trained_actor, demands, cost_matrix)[1]
    pomo_routes = convert_tour_to_routes(pomo_sol)

    or_mat = cost_matrix.squeeze(0).to('cpu').numpy()
    or_dem = demands.to('cpu').squeeze().numpy()
    or_sol = solve_with_ortools(or_mat, or_dem)
    or_routes = or_sol['routes']
    or_eid = edge_index.to('cpu').squeeze().numpy()
    or_t = t.to('cpu').squeeze().numpy()

    test_instance = map_utils.convert_node_info_to_dataframe(or_eid, or_t, or_dem)
    
    map_utils.visualize_two_cvrp_solutions(G_utm, generator, test_instance, or_mat, 
                                           or_routes, pomo_routes,
                                           panel_title_1="ORTOOLS Solution",
                                           panel_title_2="POMO Solution",
                                           fname='pomo_vs_baseline.html'
    )