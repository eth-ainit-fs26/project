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

from .parameters import MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS

from .cvrp_generator import CVRPGenerator
NumberType = np.generic | int | float
DEVICE = None # to be set in the notebook before using

def CVRP_DATA_LOADER(
    generator: CVRPGenerator,
    n2v_embeddings: np.ndarray,
    num_sample: int, 
    batch_size: int, 
    problem_sizes_mean: Optional[Union[List[NumberType], NumberType]] = None, 
    problem_sizes_std: Optional[Union[List[NumberType], NumberType]] = None, 
    rng: Optional[np.random.Generator] = None
):
    '''
    Dataloader that generates random CVRP instances on-the-fly.
    The problem sizes (number of customers) are fixed per batch and sampled 
    from a gaussian with specified mean and std before rounding
    Parameters:
        generator: an instance of CVRPGenerator that provides the logic for sampling CVRP instances
        n2v_embeddings: node2vec embeddings for the base graph
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
        n2v_embeddings=n2v_embeddings,
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
                 n2v_embeddings: np.ndarray, 
                 num_sample: int, 
                 batch_size: int, 
                 problem_sizes_mean: List[NumberType], 
                 problem_sizes_std: List[NumberType],
                 normalize: bool = True,
                 rng: Optional[np.random.Generator] = None):
        '''Constructs a dataset of CVRP instances by pre-generating all the data in batches.
        Parameters:
            generator: an instance of CVRPGenerator that provides the logic for sampling CVRP instances
            n2v_embeddings: node2vec embeddings for the base graph
            num_sample: total number of instances to generate
            batch_size: batch size for the dataset
            problem_sizes_mean: the means of problem size per batch
            problem_sizes_std: the stds of problem size per batch
            normalize: whether to normalize the coordinates
            rng: random number generator for reproducibility
        '''
        self.generator = generator
        self.n2v_embeddings = n2v_embeddings
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_sample / batch_size))
        self.rng = np.random.default_rng() if rng is None else rng
        self.normalize = normalize

        # 1. Generate dynamic problem sizes per batch
        self.problem_size_list = np.round(self.rng.normal(problem_sizes_mean, problem_sizes_std))
        self.problem_size_list = np.clip(self.problem_size_list, MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS).astype(int)
        
        # 2. Pre-allocate storage lists for the batched data
        self.batches_depot_xy = []
        self.batches_depot_n2v = []
        self.batches_node_xy = []
        self.batches_node_demand = []
        self.batches_node_n2v = []
        self.batches_cost_matrix = []

        # 3. Use sample_batch to generate all data up front
        for batch_idx in range(self.num_batches):
            prob_size = self.problem_size_list[batch_idx]
            num_locations = prob_size + 1  # 1 depot + customers

            # Vectorized generation of the entire batch at once
            edge_idx, ts, demands, cost_matrices = self.generator.sample_batch(
                batch_size=self.batch_size, 
                num_locations=num_locations
            )
            # Compute coordinates
            px, py = self.generator.compute_interpolated_coordinates(edge_idx, ts, normalized=self.normalize)
            coordinates = np.stack([px, py], axis=-1)  # Shape: (batch_size, num_locations, 2)
            
            # Slice out the depot (idx 0) from customers (idx 1:) across the batch dimension
            depot_xy = coordinates[:, 0:1, :]   # Shape: (batch_size, 1, 2)
            node_xy = coordinates[:, 1:, :]     # Shape: (batch_size, prob_size, 2)
            node_demand = demands[:, 1:]        # Shape: (batch_size, prob_size)
            
            # Get node2vec embeddings for depot and customers
            # First get the serialized indices of source/target nodes
            u_idx, v_idx = self.generator.u_idx[edge_idx], self.generator.v_idx[edge_idx] 
            # Then get the node2vec embeddings for source/target nodes and interpolate them
            u_n2v = self.n2v_embeddings[u_idx]  # Shape: (batch_size, num_locations, embedding_dim)
            v_n2v = self.n2v_embeddings[v_idx]  # Shape: (batch_size, num_locations, embedding_dim)
            all_n2v = u_n2v + ts[:,:,None] * (v_n2v - u_n2v)  # Shape: (batch_size, num_locations, embedding_dim)
            depot_n2v = all_n2v[:, 0:1, :]  # Shape: (batch_size, 1, embedding_dim)
            node_n2v = all_n2v[:, 1:, :]  # Shape: (batch_size, prob_size, embedding_dim)

            # Append the arrays to our dataset storage
            self.batches_depot_xy.append(depot_xy)
            self.batches_depot_n2v.append(depot_n2v)
            self.batches_node_xy.append(node_xy)
            self.batches_node_demand.append(node_demand)
            self.batches_node_n2v.append(node_n2v)
            self.batches_cost_matrix.append(cost_matrices)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        # Determine which batch the index belongs to, and its local position inside that batch
        batch_number = index // self.batch_size
        item_number = index % self.batch_size

        # Extract the single item from our pre-allocated batch tensors
        depot_xy = self.batches_depot_xy[batch_number][item_number]
        depot_n2v = self.batches_depot_n2v[batch_number][item_number]
        node_xy = self.batches_node_xy[batch_number][item_number]
        node_demands = self.batches_node_demand[batch_number][item_number]
        node_n2v = self.batches_node_n2v[batch_number][item_number]
        cost_matrix = self.batches_cost_matrix[batch_number][item_number] # Shape: (num_locations, num_locations)

        # Combine depot features and customer features
        depot_features = np.concatenate([depot_xy, depot_n2v], axis=-1)  # Shape: (1, 2+embedding_dim)
        node_features = np.concatenate([node_xy, node_n2v], axis=-1)  # Shape: (num_customers, 2+embedding_dim)
        return depot_features, node_features, node_demands, cost_matrix

def CVRP_collate_fn(batch):
    depot_features_tuples, node_features_tuples, node_demands_tuples, cost_matrix_tuples = zip(*batch)
    
    depot_features = torch.FloatTensor(np.array(depot_features_tuples)).to(DEVICE)
    node_features = torch.FloatTensor(np.array(node_features_tuples)).to(DEVICE)
    node_demands = torch.LongTensor(np.array(node_demands_tuples))[:, :, None].to(DEVICE)
    cost_matrix = torch.FloatTensor(np.array(cost_matrix_tuples)).to(DEVICE)
    
    return depot_features, node_features, node_demands, cost_matrix


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

