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

from source.utilities import Average_Meter
from source.cvrp import DATALOADER, GROUP_ENVIRONMENT
from .parameters import MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS
from source.grouped_actor import ACTOR
DEVICE = None # to be set in the notebook before using

########################################
# EVAL
########################################

eval_result = []

def EVAL(grouped_actor: ACTOR,
         generator, 
         n2v_embeddings: np.ndarray,   
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
    DEVICE = grouped_actor.device

    eval_AM = Average_Meter(DEVICE)
    test_loader = DATALOADER(generator=generator,
                             n2v_embeddings=n2v_embeddings,
                             num_sample=TEST_DATASET_SIZE,
                             batch_size=TEST_BATCH_SIZE,
                             problem_sizes_mean=problem_sizes_mean,
                             problem_sizes_std=problem_sizes_std,
                             rng=rng)

    with torch.no_grad():
        for depot_features, node_features, node_demand, cost_matrix in test_loader:
            # depot_features.shape = (batch, 1, 2+N2V_DIM)
            # node_features.shape = (batch, problem, 2+N2V_DIM)
            # node_demand.shape = (batch, problem, 1)
            # cost_matrix.shape = (batch, problem+1, problem+1)

            batch_s = depot_features.size(0)
            group_s = node_features.size(1)  # = problem size

            env = GROUP_ENVIRONMENT(depot_features, node_features, node_demand, cost_matrix)
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
            eval_AM.push(-max_reward)  # reward was given as negative cost


    # LOGGING
    cost_avg = eval_AM.result()
    eval_result.append(cost_avg)


    logger.info('--------------------------------------------------------------------------')
    log_str = '  <<< EVAL after Epoch:{:03d} >>>   AvgCost:{:f}'.format(epoch, cost_avg)
    logger.info(log_str)
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')


def evaluate_actor(grouped_actor: ACTOR, 
                   generator,
                   n2v_embeddings,
                   n_instances_per_size: int, 
                   seed: int=None):
    ''' Evaluate the trained grouped actor on CVRP test dataset.
    Parameters:
        grouped_actor: trained ACTOR instance to be evaluated (defined in notebook)
        generator: CVRPGenerator instance for generating evaluation data
        n2v_embeddings: numpy array of Node2Vec embeddings (defined in notebook)
        n_instance_per_size: number of instances per problem size for evaluation
        seed: random seed for eval data generation
    Returns:
        eval_result_cost: list of average tour costs per problem size
        eval_result_time: list of average solving times per problem size
    '''
    eval_result_cost = [] # to store eval results per epoch
    eval_result_time = [] # to store solving times per epoch
    rng = np.random.default_rng(seed)
    DEVICE = grouped_actor.device
    grouped_actor.eval() # set the actor to evaluation mode

    with torch.no_grad():    
        for n_customers in tqdm(range(MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS + 1)):
            test_loader = DATALOADER(generator=generator,
                                     n2v_embeddings=n2v_embeddings,
                                     num_sample=n_instances_per_size, 
                                     batch_size=1, # for fair comparison
                                     problem_sizes_mean=n_customers, 
                                     rng=rng)
            total_time, total_cost = 0, 0 # for this problem size
            for depot_features, node_features, node_demand, cost_matrix in test_loader:
                start_time = time.time()
                batch_s = depot_features.size(0) # batch size
                group_s = node_features.size(1)  # problem size = n_cust

                # Step 0
                env = GROUP_ENVIRONMENT(depot_features, node_features, node_demand, cost_matrix)
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
                
                max_reward, _ = reward.max(dim=1) # best rollouts; shape = (batch,)
                total_cost += -max_reward.mean().to('cpu') # reward was given as negative cost
                
                end_time = time.time()
                total_time += end_time - start_time
            eval_result_cost.append(total_cost/n_instances_per_size) # average cost per instance
            eval_result_time.append(total_time/n_instances_per_size) # average time per instance

    assert len(eval_result_cost) == (MAX_NUM_CUSTOMERS - MIN_NUM_CUSTOMERS + 1), \
            "Evaluation result length mismatch with number of problem sizes evaluated."
    
    return eval_result_cost, eval_result_time
