# CVRP parameters
MIN_NUM_CUSTOMERS = 20
MAX_NUM_CUSTOMERS = 50
MAX_DEMAND = 10 # per customer
VEHICLE_CAPACITY = 50
ALPHA = 1e-4 # coefficient for distance (L/m)
BETA = 1e-3  # coefficient for time (L/s)


# OR-Tools parameters
COST_SCALER = 1e8 # OR tool works with integers, so we scale up the cost for precision


# Node2Vec parameters
N2V_DIM = 128  # Dimensionality of the node embeddings 
P = 1.0  # Return parameter (p); smaller = more likely to backtrack
Q = 2.0  # In-out parameter (q); smaller = more likely to explore outward
NUM_WALKS = 100  # Number of random walks per node
WALK_LENGTH = 80  # Length of each random walk


