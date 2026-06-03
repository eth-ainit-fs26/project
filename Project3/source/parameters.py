# CVRP parameters
MIN_NUM_CUSTOMERS = 20
MAX_NUM_CUSTOMERS = 50
MAX_DEMAND = 10 # per customer
VEHICLE_CAPACITY = 50
ALPHA = 1e-4 # coefficient for distance (L/m)
BETA = 1e-3  # coefficient for time (L/s)


# OR-Tools parameters
COST_SCALER = 1e8 # OR tool works with integers, so we scale up the cost for precision
