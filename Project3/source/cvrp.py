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

# EXTERNAL LIBRARY
import numpy as np
from numpy.random._generator import Generator
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union
from IPython.core.debugger import set_trace

from .parameters import MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS
from .cvrp_generator import CVRPGenerator
NumberType = np.generic | int | float
DEVICE = None # to be set in the notebook before using

def CVRP_DATA_LOADER(
    generator: CVRPGenerator,
    num_sample: int, 
    batch_size: int, 
    problem_sizes_mean: Optional[Union[List[NumberType], NumberType]] = None, 
    problem_sizes_std: Optional[Union[List[NumberType], NumberType]] = None, 
    return_edges: bool = False,
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
        return_edges: whether to return the raw edge indices and interpolation factors
        rng: an optional numpy random generator for reproducibility; if None, use the generator from CVRPGenerator
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

    
    dataset = CVRP_Dataset(
        generator=generator,
        num_sample=num_sample, 
        batch_size=batch_size, 
        problem_sizes_mean=problem_sizes_mean,
        problem_sizes_std=problem_sizes_std,
        return_edges=return_edges,
        rng=generator.rng if rng is None else rng
    )
    
    collate_fn = CVRP_collate_fn_with_edges if return_edges else CVRP_collate_fn
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return data_loader



class CVRP_Dataset(Dataset):
    def __init__(self, 
                 generator: CVRPGenerator,
                 num_sample: int, 
                 batch_size: int, 
                 problem_sizes_mean: List[NumberType], 
                 problem_sizes_std: List[NumberType],
                 normalize: bool = True,
                 return_edges: bool = False,
                 rng: Optional[np.random.Generator] = None):
        '''Constructs a dataset of CVRP instances by pre-generating all the data in batches.
        Parameters:
            generator: an instance of CVRPGenerator that provides the logic for sampling CVRP instances
            num_sample: total number of instances to generate
            batch_size: batch size for the dataset
            problem_sizes_mean: the means of problem size per batch
            problem_sizes_std: the stds of problem size per batch
            normalize: whether to normalize the cost matrix by the global max cost across the map
            return_edges: whether to return the raw edge indices and interpolation factors
        '''
        self.generator = generator
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_sample / batch_size))
        self.normalize = normalize
        self.return_edges = return_edges
        if rng is None:
            self.rng = generator.rng
        else:
            self.rng = rng
            self.generator.rng = rng # Set the generator's random number generator to ensure reproducibility

        # 1. Generate dynamic problem sizes per batch
        self.problem_size_list = np.round(self.rng.normal(problem_sizes_mean, problem_sizes_std))
        self.problem_size_list = np.clip(self.problem_size_list, MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS).astype(int)
        
        # 2. Pre-allocate storage lists for the batched data
        self.batches_demands = []
        self.batches_features = []
        self.batches_cost_matrix = []
        if self.return_edges:
            self.batches_edge_idx = []
            self.batches_t = []

        # 3. Use sample_batch to generate all data up front
        for batch_idx in range(self.num_batches):
            prob_size = self.problem_size_list[batch_idx]
            num_locations = prob_size + 1  # 1 depot + customers

            # Batch-generated random instances
            edge_idx, ts, demands, cost_matrices = self.generator.sample_batch(
                batch_size=self.batch_size, 
                num_locations=num_locations
            )

            # Feature extraction
            # 1. Normalized demand
            normalized_demands = demands / self.generator.max_demand

            # 2. Mask the diagonal with NaN so self-loops don't skew the statistics
            masked_costs = cost_matrices / self.generator.global_max_cost
            diag_idx = np.arange(num_locations)
            masked_costs[:, diag_idx, diag_idx] = np.nan
            
            # 3. Compute Outbound Stats (Row-wise: axis=1)
            min_out = np.nanmin(masked_costs, axis=1)
            mean_out = np.nanmean(masked_costs, axis=1)
            std_out = np.nanstd(masked_costs, axis=1)
            p10_out, p50_out, p90_out = np.nanpercentile(masked_costs, [10, 50, 90], axis=1)

            # 4. Compute Inbound Stats (Column-wise: axis=2)
            min_in = np.nanmin(masked_costs, axis=2)
            mean_in = np.nanmean(masked_costs, axis=2)
            std_in = np.nanstd(masked_costs, axis=2)
            p10_in, p50_in, p90_in = np.nanpercentile(masked_costs, [10, 50, 90], axis=2)

            # 5. Stack into the final Node Feature Tensor
            # Shape before stack: All are (batch_size, num_locations)
            # Shape after stack: (batch_size, num_locations, NODE_FEATURE_DIM)
            features = np.stack([
                normalized_demands, min_in, min_out, 
                mean_in, mean_out, std_in, std_out, 
                p10_in, p10_out, p50_in, p50_out, p90_in, p90_out
            ], axis=-1)

            # Append the arrays to our dataset storage
            self.batches_demands.append(demands)
            self.batches_features.append(features)
            self.batches_cost_matrix.append(cost_matrices)
            if self.return_edges:
                self.batches_edge_idx.append(edge_idx)
                self.batches_t.append(ts)
                
    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        # Determine which batch the index belongs to, and its local position inside that batch
        batch_number = index // self.batch_size
        item_number = index % self.batch_size

        # Extract the single item from our pre-allocated batch tensors
        demands = self.batches_demands[batch_number][item_number]
        features = self.batches_features[batch_number][item_number]
        cost_matrix = self.batches_cost_matrix[batch_number][item_number] # Shape: (num_locations, num_locations)        
        if self.return_edges:
            edge_idx = self.batches_edge_idx[batch_number][item_number]
            t = self.batches_t[batch_number][item_number]
            return demands, features, cost_matrix, edge_idx, t
        return demands, features, cost_matrix

def CVRP_collate_fn(batch):
    demands_tuples, features_tuples, cost_matrix_tuples = zip(*batch)
    
    demands = torch.LongTensor(np.array(demands_tuples))[:,:,None].to(DEVICE)
    features = torch.FloatTensor(np.array(features_tuples)).to(DEVICE)
    cost_matrix = torch.FloatTensor(np.array(cost_matrix_tuples)).to(DEVICE)

    return demands, features, cost_matrix

def CVRP_collate_fn_with_edges(batch):
    demands_tuples, features_tuples, cost_matrix_tuples, edge_idx_tuples, t_tuples = zip(*batch)
    
    demands = torch.LongTensor(np.array(demands_tuples))[:,:,None].to(DEVICE)
    features = torch.FloatTensor(np.array(features_tuples)).to(DEVICE)
    cost_matrix = torch.FloatTensor(np.array(cost_matrix_tuples)).to(DEVICE)
    edge_idx = torch.LongTensor(np.array(edge_idx_tuples)).to(DEVICE)
    t = torch.FloatTensor(np.array(t_tuples)).to(DEVICE)

    return demands, features, cost_matrix, edge_idx, t

DATALOADER = CVRP_DATA_LOADER # short name



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
    def __init__(self, depot_features, node_features, node_demand, cost_matrix):
        raise NotImplementedError("Implement the GROUP_ENVIRONMENT class in the notebook before using it.")

    def reset(self, group_size):
        raise NotImplementedError("Implement the 'reset' method in the notebook before using it.")

    def step(self, selected_idx_mat):
        raise NotImplementedError("Implement the 'step' method in the notebook before using it.")

    def _get_travel_distance(self):
        raise NotImplementedError("Implement the '_get_travel_distance' method in the notebook before using it.")

