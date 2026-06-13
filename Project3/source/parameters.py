# CVRP parameters
MIN_NUM_CUSTOMERS = 30
MAX_NUM_CUSTOMERS = 60
MAX_DEMAND = 10 # per customer
VEHICLE_CAPACITY = 50
ALPHA = 1e-4 # coefficient for distance (L/m)
BETA = 1e-3  # coefficient for time (L/s)


# OR-Tools parameters
COST_SCALER = 1e8 # OR tool works with integers, so we scale up the cost for precision

# Node embedding parameters
NODE_FEATURE_DIM = 13
