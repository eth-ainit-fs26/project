import numpy as np
from .parameters import MAX_DEMAND
import networkx as nx

class CVRPGenerator:
    def __init__(self, G: nx.MultiDiGraph, apsp_matrix: np.ndarray, 
                 node_to_idx: dict, rng=np.random.default_rng()):
        self.apsp_matrix = apsp_matrix
        self.num_edges = len(G.edges)
        self.rng = rng
        self.max_demand = MAX_DEMAND

        # Pre-allocate numpy arrays for rapid vectorization
        # u_idx, v_idx: integer indices of the source/target node
        self.u_idx = np.zeros(self.num_edges, dtype=int)
        self.v_idx = np.zeros(self.num_edges, dtype=int)
        self.u_x = np.zeros(self.num_edges)
        self.u_y = np.zeros(self.num_edges)
        self.v_x = np.zeros(self.num_edges)
        self.v_y = np.zeros(self.num_edges)
        self.u_x_norm = np.zeros(self.num_edges)
        self.u_y_norm = np.zeros(self.num_edges)
        self.v_x_norm = np.zeros(self.num_edges)
        self.v_y_norm = np.zeros(self.num_edges)
        self.edge_costs = np.zeros(self.num_edges)
        
        for i, (u, v, k, d) in enumerate(G.edges(keys=True, data=True)):
            # u and v are OSM node IDs, i is the edge index
            self.u_idx[i], self.v_idx[i] = node_to_idx[u], node_to_idx[v]
            self.u_x[i], self.u_y[i] = G.nodes[u]['x'], G.nodes[u]['y']
            self.v_x[i], self.v_y[i] = G.nodes[v]['x'], G.nodes[v]['y']
            self.u_x_norm[i], self.u_y_norm[i] = G.nodes[u]['x_norm'], G.nodes[u]['y_norm']
            self.v_x_norm[i], self.v_y_norm[i] = G.nodes[v]['x_norm'], G.nodes[v]['y_norm']
            self.edge_costs[i] = d['fuel']
    
    
    def compute_interpolated_coordinates(self, edge_indices: np.ndarray, t: np.ndarray, normalized: bool):
        '''Given edge indices and interpolation factors t, computes the absolute coordinates of the sampled points.'''
        if normalized:
            u_x, u_y = self.u_x_norm[edge_indices], self.u_y_norm[edge_indices]
            v_x, v_y = self.v_x_norm[edge_indices], self.v_y_norm[edge_indices]
        else:
            u_x, u_y = self.u_x[edge_indices], self.u_y[edge_indices]
            v_x, v_y = self.v_x[edge_indices], self.v_y[edge_indices]
        
        px = u_x + t * (v_x - u_x)
        py = u_y + t * (v_y - u_y)

        return px, py

    def sample_instance(self, num_locations: int):
        '''Generates a single CVRP instance with the specified number of locations.
        Input:
            num_locations: The number of locations (including the depot) to sample for the CVRP
        Outputs:
            edge_indices: Shape (num_locations,) -> The edge index for each sampled location
            t: Shape (num_locations,) -> The fractional position along the edge for each sampled location
            demands: Shape (num_locations,) -> Random demand values for each location
            cost_matrix: Shape (num_locations, num_locations) -> Asymmetric routing costs
        '''
        # 1. Sample points p=u+t(v-u) for random edges u->v and fractions t
        edge_indices = self.rng.choice(self.num_edges, size=num_locations, replace=True)
        t = self.rng.random(num_locations)
        
        # 2. Build Cost Matrix
        u = self.u_idx[edge_indices] # source node indices
        v = self.v_idx[edge_indices] # target node indices
        costs = self.edge_costs[edge_indices] # edge costs

        # To compute the cost from p=u+t(v-u) to p'=u'+t'(v'-u') for each pair (p,p')
        # of the sampled nodes, we need to traverse the path p -> v -> u' -> p'.
        cost_to_v = (1 - t) * costs # the costs p -> v
        cost_from_u = t * costs     # the costs u' -> p'
        
        # Matrix of node-to-node costs from endpoint v_i to startpoint u_j
        node_to_node = self.apsp_matrix[np.ix_(v, u)] # the costs v -> u'
        
        # Base matrix computation using broadcasting
        cost_matrix = cost_to_v[:, None] + node_to_node + cost_from_u[None, :]
        # shape: (num_locations, num_locations) with cost_matrix[p,p'] = cost(p -> v -> u' -> p')
        # cost_to_v[:, None]: broadcasted across columns
        # cost_from_u[None, :]: broadcasted across rows

        # 3. Handle identical edge overlaps safely
        # If u->v and u'->v' are the same edge and t <= t' (i.e., p is before p' on the same edge), 
        # we can directly compute the cost as (t'-t)*cost(u->v) without routing through the network 
        same_edge_mask = edge_indices[:, None] == edge_indices[None, :]
        t_diff = t[None, :] - t[:, None]
        valid_direct_mask = same_edge_mask & (t_diff >= 0)
        direct_costs = t_diff * costs[:, None]
        # Overwrite valid direct routes; otherwise, it relies on wrapping around the network 
        cost_matrix = np.where(valid_direct_mask, direct_costs, cost_matrix)
        # Component-wise: cost_matrix[p,p'] = direct_costs[p,p'] if valid_direct_mask[p,p'] else cost_matrix[p,p']

        # 4. Zero out diagonal
        np.fill_diagonal(cost_matrix, 0.0)
        
        # 5. Generate demand vector
        demands = self.rng.integers(1, self.max_demand + 1, size=num_locations)
        demands[0] = 0 # depot has zero demand
        
        return edge_indices, t, demands, cost_matrix 
    
    def sample_batch(self, batch_size: int, num_locations: int):
        """
        Generates a heavily vectorized batch of independent CVRP instances.
        Inputs:
            batch_size: The number of independent CVRP instances to generate in the batch
            num_locations: The number of locations (including the depot) to sample for each CVRP instance
        Outputs:
            edge_indices: Shape (batch_size, num_locations) -> The edge index for each sampled location
            ts: Shape (batch_size, num_locations) -> The fractional position along the edge for each sampled location
            demands: Shape (batch_size, num_locations) -> Random demand values for each location
            cost_matrices: Shape (batch_size, num_locations, num_locations) -> Asymmetric routing costs
        """
        # 1. Multi-dimensional sampling: shape (batch_size, num_locations)
        edge_indices = self.rng.choice(self.num_edges, size=(batch_size, num_locations), replace=True)
        ts = self.rng.random((batch_size, num_locations))
        
        # 2. Build cost matrix
        u = self.u_idx[edge_indices]
        v = self.v_idx[edge_indices]
        costs = self.edge_costs[edge_indices]
        
        # Compute Sub-Edge Exit & Entry Costs
        cost_to_v = (1 - ts) * costs   # Shape: (batch_size, num_locations)
        cost_from_u = ts * costs       # Shape: (batch_size, num_locations)
        
        # Extract Macro-routing Sub-matrices from the APSP Oracle
        # Because APSP is 2D and v,u are 2D batch matrices, we use advanced integer broadcasting
        node_to_node = self.apsp_matrix[v[:, :, None], u[:, None, :]] # Shape: (batch_size, num_locations, num_locations)
        
        # Assemble 3D Cost Matrix via Broadcasting
        # cost_to_v expands to (B, N, 1) and cost_from_u expands to (B, 1, N)
        cost_matrices = cost_to_v[:, :, None] + node_to_node + cost_from_u[:, None, :]
        
        # 3. Multi-Dimensional Same-Edge Masking
        # Check if locations share the same edge within their respective batch index
        same_edge_mask = edge_indices[:, :, None] == edge_indices[:, None, :]
        t_diff = ts[:, None, :] - ts[:, :, None] # Shape: (batch_size, num_locations, num_locations)
        
        # Valid shortcut condition: same edge, and destination is further down the line
        valid_direct_mask = same_edge_mask & (t_diff >= 0)
        direct_costs = t_diff * costs[:, :, None]
        
        # Overwrite macro paths with direct same-edge values where applicable
        cost_matrices = np.where(valid_direct_mask, direct_costs, cost_matrices)
        
        # 4. Zero out diagonals safely across the batch
        diag_idx = np.arange(num_locations)
        cost_matrices[:, diag_idx, diag_idx] = 0.0
        
        # 5. Generate demand vectors for the entire batch
        demands = self.rng.integers(1, self.max_demand + 1, size=(batch_size, num_locations))
        demands[:, 0] = 0 # depot has zero demand

        return edge_indices, ts, demands, cost_matrices 

