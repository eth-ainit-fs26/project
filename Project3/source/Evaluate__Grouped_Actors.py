"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon
Copyright (c) 2025 Department of Computer Science, ETH Zurich

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

from source.utilities import Average_Meter, augment_xy_data_by_8_fold
from source.cvrp import (
    CVRP_DATA_LOADER__RANDOM, 
    GROUP_ENVIRONMENT,
    MAX_NUM_CUSTOMERS,
    MIN_NUM_CUSTOMERS
)
from source.grouped_actors import ACTOR
DEVICE = None # to be set in the notebook before using

########################################
# EVAL
########################################

eval_result = []

def EVAL(grouped_actor: ACTOR, epoch: int, timer_start, logger,
         TEST_DATASET_SIZE: int,
         TEST_BATCH_SIZE: int,
         problem_sizes_mean=None,
         problem_sizes_std=None,
         rng=None):

    global eval_result
    grouped_actor.eval()
    DEVICE = grouped_actor.device

    eval_AM = Average_Meter(DEVICE)
    test_loader = CVRP_DATA_LOADER__RANDOM(num_sample=TEST_DATASET_SIZE,
                                           batch_size=TEST_BATCH_SIZE,
                                           problem_sizes_mean=problem_sizes_mean,
                                           problem_sizes_std=problem_sizes_std,
                                           rng=rng)

    with torch.no_grad():
        for depot_xy, node_xy, node_demand, _ in test_loader:
            # depot_xy.shape = (batch, 1, 2)
            # node_xy.shape = (batch, problem, 2)
            # node_demand.shape = (batch, problem, 1)
            # remark: demand_scalers is not used in evaluation

            batch_s = depot_xy.size(0)

            env = GROUP_ENVIRONMENT(depot_xy, node_xy, node_demand)
            group_s = node_xy.size(1)  # problem size
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)

            # First Move is given
            first_action = torch.LongTensor(np.zeros((batch_s, group_s))).to(DEVICE)  # start from node_0-depot
            group_state, reward, done = env.step(first_action)

            # Second Move is given
            second_action = torch.LongTensor(np.arange(group_s)+1)[None, :].expand(batch_s, group_s).to(DEVICE)
            group_state, reward, done = env.step(second_action)

            while not done:
                action_probs = grouped_actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem+1)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)

            max_reward, _ = reward.max(dim=1)
            eval_AM.push(-max_reward)  # reward was given as negative dist


    # LOGGING
    dist_avg = eval_AM.result()
    eval_result.append(dist_avg)


    logger.info('--------------------------------------------------------------------------')
    log_str = '  <<< EVAL after Epoch:{:03d} >>>   Avg.dist:{:f}'.format(epoch, dist_avg)
    logger.info(log_str)
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')


def evaluate_actor(grouped_actor: ACTOR, n_instances_per_size: int, augment: bool=False, seed: int=None):
    ''' Evaluate the trained grouped actor on CVRP test dataset.
    Parameters:
        grouped_actor: trained ACTOR instance to be evaluated (defined in notebook)
        n_instance_per_size: number of instances per problem size for evaluation
        augment: whether to use data augmentation for evaluation
        seed: random seed for eval data generation
    Returns:
        eval_result_dist: list of average tour lengths per problem size
        eval_result_time: list of average solving times per problem size
    '''
    eval_result_dist = [] # to store eval results per epoch
    eval_result_time = [] # to store solving times per epoch
    rng = np.random.default_rng(seed)
    DEVICE = grouped_actor.device
    grouped_actor.eval() # set the actor to evaluation mode

    with torch.no_grad():    
        for n_customers in tqdm(range(MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS + 1)):
            test_loader = CVRP_DATA_LOADER__RANDOM(num_sample=n_instances_per_size, 
                                                   batch_size=1, # for fair comparison
                                                   problem_sizes_mean=n_customers, 
                                                   rng=rng)
            total_time, total_dist = 0, 0 # for this problem size
            for depot_xy, node_xy, node_demand, _ in test_loader:
                # shape = (batch, 1, 2), (batch, n_cust, 2), (batch, n_cust, 1)
                start_time = time.time()

                batch_s = depot_xy.size(0) # batch size
                group_s = node_xy.size(1)  # problem size = n_cust
                
                if augment: # 8 fold Augmented
                    depot_xy = augment_xy_data_by_8_fold(depot_xy) # shape = (8*batch, 1, 2)
                    node_xy = augment_xy_data_by_8_fold(node_xy) # shape = (8*batch, n_cust, 2)
                    node_demand = node_demand.repeat(8, 1, 1) # shape = (8*batch, n_cust, 2)
                    batch_s = 8*batch_s
                
                # Step 0
                env = GROUP_ENVIRONMENT(depot_xy, node_xy, node_demand)
                group_state, reward, done = env.reset(group_size=group_s)
                grouped_actor.reset(group_state)
                # Steps 1 and 2
                first_action = torch.LongTensor(np.zeros((batch_s, group_s))).to(DEVICE) 
                group_state, reward, done = env.step(first_action)
                second_action = torch.LongTensor(np.arange(group_s)+1)[None, :].expand(batch_s, group_s).to(DEVICE)
                group_state, reward, done = env.step(second_action)
                # Subsequent Steps
                while not done:
                    action_probs = grouped_actor.get_action_probabilities(group_state) # shape = (batch, group, problem+1)
                    action = action_probs.argmax(dim=2) # shape = (batch, group)
                    action[group_state.finished] = 0  # stay at depot, if you are finished
                    group_state, reward, done = env.step(action) # reward shape = (batch, group)
                if not augment:
                    max_reward, _ = reward.max(dim=1) # best rollouts; shape = (batch,)
                else:
                    reward = reward.reshape(8, round(batch_s/8), group_s) # shape = (8, batch, group)
                    reward, _ = reward.max(dim=2) # best rollouts among groups; shape = (8, batch)
                    max_reward, _ = reward.max(dim=0) # best rollouts among augmentations; shape = (batch,)
                total_dist += -max_reward.mean().to('cpu') # reward was given as negative distance
                
                end_time = time.time()
                total_time += end_time - start_time
            eval_result_dist.append(total_dist/n_instances_per_size) # average dist per instance
            eval_result_time.append(total_time/n_instances_per_size) # average time per instance

    assert len(eval_result_dist) == (MAX_NUM_CUSTOMERS - MIN_NUM_CUSTOMERS + 1), \
            "Evaluation result length mismatch with number of problem sizes evaluated."
    
    return eval_result_dist, eval_result_time
