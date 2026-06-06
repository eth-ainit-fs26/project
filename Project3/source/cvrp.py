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



# EXTERNAL LIBRARY
import numpy as np
from numpy.random._generator import Generator
from torch.utils.data import Dataset, DataLoader
from IPython.core.debugger import set_trace
import torch
from typing import List, Optional, Union

from .parameters import MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS, MAX_DEMAND, VEHICLE_CAPACITY

from .cvrp_generator import CVRPGenerator
NumberType = np.generic | int | float
DEVICE = None # to be set in the notebook before using

def CVRP_DATA_LOADER(
    generator: CVRPGenerator,
    num_sample: int, 
    batch_size: int, 
    problem_sizes_mean: Optional[Union[List[NumberType], NumberType]] = None, 
    problem_sizes_std: Optional[Union[List[NumberType], NumberType]] = None, 
    normalized_coordinates: bool = True,
    rng: Optional[np.random.Generator] = None
):
    '''
    Dataloader that generates random CVRP instances on-the-fly.
    The problem sizes (number of customers) are fixed per batch and sampled 
    from a gaussian with specified mean and std before rounding
    Parameters:
        generator: an instance of CVRPGenerator that provides the logic for sampling CVRP instances
        num_sample: total number of instances to generate
        batch_size: batch size for the data loader
        problem_sizes_mean: the means of problem size per batch; if None, set to 25
        problem_sizes_std: the stds of problem size per batch; if None, set to 0
        rng: random number generator for reproducibility
     '''
    n_batches = int(np.ceil(num_sample / batch_size))
    
    # Input sanity checks matching your template
    if isinstance(problem_sizes_mean, list):
        assert len(problem_sizes_mean) == n_batches, \
            "Length of problem_sizes_mean must match the number of batches"
        if isinstance(problem_sizes_std, list):
            assert len(problem_sizes_mean) == len(problem_sizes_std), \
                "problem_sizes_mean and problem_sizes_std must have the same length"
                
    problem_sizes_mean = 50 if problem_sizes_mean is None else problem_sizes_mean
    problem_sizes_mean = [problem_sizes_mean] * n_batches if not isinstance(problem_sizes_mean, list) else problem_sizes_mean
    
    problem_sizes_std = 0 if problem_sizes_std is None else problem_sizes_std
    problem_sizes_std = [problem_sizes_std] * n_batches if not isinstance(problem_sizes_std, list) else problem_sizes_std
    
    assert np.all(np.array(problem_sizes_std) >= 0), "problem_sizes_std must be nonnegative"
    
    rng = np.random.default_rng() if rng is None else rng
    
    dataset = CVRP_Dataset(
        generator=generator,
        num_sample=num_sample, 
        batch_size=batch_size, 
        problem_sizes_mean=problem_sizes_mean,
        problem_sizes_std=problem_sizes_std,
        rng=rng
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CVRP_collate_fn
    )
    return data_loader



class CVRP_Dataset(Dataset):
    def __init__(self, 
                 generator: CVRPGenerator, 
                 num_sample: int, 
                 batch_size: int, 
                 problem_sizes_mean: List[NumberType], 
                 problem_sizes_std: List[NumberType],
                 rng=None):
        '''Constructs a dataset of CVRP instances by pre-generating all the data in batches.
        Parameters:
            generator: an instance of CVRPGenerator that provides the logic for sampling CVRP instances
            num_sample: total number of instances to generate
            batch_size: batch size for the dataset
            problem_sizes_mean: the means of problem size per batch
            problem_sizes_std: the stds of problem size per batch
            rng: random number generator for reproducibility
        '''
        self.generator = generator
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_sample / batch_size))
        self.rng = np.random.default_rng() if rng is None else rng

        # 1. Generate dynamic problem sizes per batch
        self.problem_size_list = np.round(self.rng.normal(problem_sizes_mean, problem_sizes_std))
        self.problem_size_list = np.clip(self.problem_size_list, MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS).astype(int)
        
        # 2. Pre-allocate storage lists for the batched data
        self.batched_edge_ids = []
        self.batched_fractions = []
        self.batched_demands = []
        self.batched_cost_matrix = []

        # 3. Use sample_batch to generate all data up front
        for batch_idx in range(self.num_batches):
            prob_size = self.problem_size_list[batch_idx]
            num_locations = prob_size + 1  # 1 depot + customers

            # Vectorized generation of the entire batch at once
            edge_idx, ts, demands, cost_matrices = self.generator.sample_batch(
                batch_size=self.batch_size, 
                num_locations=num_locations
            )
            self.batched_edge_ids.append(edge_idx)
            self.batched_fractions.append(ts)
            self.batched_demands.append(demands)
            self.batched_cost_matrix.append(cost_matrices)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        # Determine which batch the index belongs to, and its local position inside that batch
        batch_number = index // self.batch_size
        item_number = index % self.batch_size

        # Extract the single item from our pre-allocated batch tensors
        edge_ids = self.batched_edge_ids[batch_number][item_number]
        fractions = self.batched_fractions[batch_number][item_number]
        demands = self.batched_demands[batch_number][item_number]
        cost_matrix = self.batched_cost_matrix[batch_number][item_number]

        return edge_ids, fractions, demands, cost_matrix
    
def CVRP_collate_fn(batch):
    edge_ids_tuples, fractions_tuples, demands_tuples, cost_matrix_tuples = zip(*batch)
    
    # edge_ids must be LongTensor to serve as index tensors later
    edge_ids = torch.LongTensor(np.array(edge_ids_tuples)).to(DEVICE)
    fractions = torch.FloatTensor(np.array(fractions_tuples)).to(DEVICE)
    demands = torch.FloatTensor(np.array(demands_tuples)).to(DEVICE)
    cost_matrix = torch.FloatTensor(np.array(cost_matrix_tuples)).to(DEVICE)
    
    return edge_ids, fractions, demands, cost_matrix


DATALOADER = CVRP_DATA_LOADER # short name



# ####################################
# # DATA LOADER
# ####################################
# def CVRP_DATA_LOADER__RANDOM(num_sample: int, batch_size: int, 
#                              problem_sizes_mean: Optional[List[NumberType]|NumberType]=None, 
#                              problem_sizes_std: Optional[List[NumberType]|NumberType]=None, 
#                              rng: Generator=None):
#     ''' Data loader that generates random CVRP instances on-the-fly.
#         The problem sizes (number of customers) are fixed per batch and sampled 
#         from a gaussian with specified mean and std before rounding
#         Parameters:
#             num_sample: total number of data samples to generate
#             batch_size: batch size for the data loader
#             problem_sizes_mean: the means of problem size per batch; if None, set to 25
#             problem_sizes_std: the stds of problem size per batch; if None, set to 0
#             rng: random number generator for reproducibility
#     '''
#     # Input sanity checks
#     n_batches = int(np.ceil(num_sample/batch_size)) # number of batches
#     if isinstance(problem_sizes_mean, List):
#         assert len(problem_sizes_mean) == n_batches, \
#                 "Length of problem_sizes_mean must match the number of batches = ceil(num_sample/batch_size)"
#         if isinstance(problem_sizes_std, List):
#             assert len(problem_sizes_mean) == len(problem_sizes_std), \
#                 "problem_sizes_mean and problem_sizes_std must have the same length if provided as lists"
#     # Convert means and stds to list of length n_batches
#     problem_sizes_mean = 25 if problem_sizes_mean is None else problem_sizes_mean
#     problem_sizes_mean = [problem_sizes_mean]*n_batches if isinstance(problem_sizes_mean, NumberType) \
#                             else problem_sizes_mean
#     problem_sizes_std = 0 if problem_sizes_std is None else problem_sizes_std
#     problem_sizes_std = [problem_sizes_std]*n_batches if isinstance(problem_sizes_std, NumberType) \
#                             else problem_sizes_std
#     assert np.all(np.array(problem_sizes_std)>=0), "problem_sizes_std must be nonnegative or contain nonnegative entries"
    
    
#     rng = np.random.default_rng() if rng is None else rng
#     dataset = CVRP_Dataset__Random(num_sample=num_sample, 
#                                    batch_size=batch_size, 
#                                    problem_sizes_mean=problem_sizes_mean,
#                                    problem_sizes_std=problem_sizes_std,
#                                    rng=rng)
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=batch_size,
#                              shuffle=False,
#                              num_workers=0,
#                              collate_fn=CVRP_collate_fn)
#     return data_loader

# class CVRP_Dataset__Random(Dataset):
#     def __init__(self, num_sample, batch_size, problem_sizes_mean, problem_sizes_std, rng=None):
#         self.num_sample = num_sample
#         self.batch_size = batch_size
#         self.num_batches = int(np.ceil(num_sample / batch_size))
#         self.rng = np.random.default_rng() if rng is None else rng

#         # Generate dynamic problem size (possible number of customers) per batch
#         # via guassian + clipping + rounding
#         self.problem_size_list = np.round(self.rng.normal(problem_sizes_mean, problem_sizes_std))
#         self.problem_size_list = np.clip(self.problem_size_list, MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS).astype(int)

#         self.demand_scalers_list = [VEHICLE_CAPACITY] * len(self.problem_size_list)

#     def __getitem__(self, index):
#         # determine the problem size and demand scaler for this index
#         batch_number = index // self.batch_size
#         problem_size = self.problem_size_list[batch_number]
#         demand_scaler = self.demand_scalers_list[batch_number]

#         # generate random data
#         depot_xy_data = self.rng.random((1, 2))
#         node_xy_data = self.rng.random((problem_size, 2))
#         node_demand_data = self.rng.integers(1, MAX_DEMAND_PER_CUSTOMER + 1, problem_size) / demand_scaler

#         return depot_xy_data, node_xy_data, node_demand_data, demand_scaler

#     def __len__(self):
#         return self.num_sample


# def CVRP_collate_fn(batch):
#     depot_xy_data_tuples, node_xy_data_tuples, node_demand_data_tuples, demand_scalers = zip(*batch)
#     depot_xy_data = np.array(depot_xy_data_tuples)
#     node_xy_data = np.array(node_xy_data_tuples)
#     node_demand_data = np.array(node_demand_data_tuples)
#     depot_xy = torch.FloatTensor(depot_xy_data).to(DEVICE)
#     node_xy = torch.FloatTensor(node_xy_data).to(DEVICE)
#     node_demand = torch.FloatTensor(node_demand_data)[:, :, None].to(DEVICE)  # unsqeeeze to match the shape of node_xy
#     return depot_xy, node_xy, node_demand, demand_scalers[0]

# DATALOADER = CVRP_DATA_LOADER__RANDOM # short name

####################################
# STATE
####################################

class GROUP_STATE:
    def __init__(self, group_size, data):
        raise NotImplementedError("Implement the GROUP_STATE class in the notebook before using it.")

    def move_to(self, selected_idx_mat):
        raise NotImplementedError("Implement the 'move_to' method in the notebook before using it.")


####################################
# ENVIRONMENT
####################################

class GROUP_ENVIRONMENT:
    def __init__(self, depot_xy, node_xy, node_demand):
        raise NotImplementedError("Implement the GROUP_ENVIRONMENT class in the notebook before using it.")

    def reset(self, group_size):
        raise NotImplementedError("Implement the 'reset' method in the notebook before using it.")

    def step(self, selected_idx_mat):
        raise NotImplementedError("Implement the 'step' method in the notebook before using it.")

    def _get_travel_distance(self):
        raise NotImplementedError("Implement the '_get_travel_distance' method in the notebook before using it.")

