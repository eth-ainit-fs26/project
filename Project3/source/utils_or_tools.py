from .parameters import (
    VEHICLE_CAPACITY, 
    COST_SCALER,
    MIN_NUM_CUSTOMERS,
    MAX_NUM_CUSTOMERS,
)
from tqdm import tqdm
import time
import numpy as np
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from .cvrp_generator import CVRPGenerator

def reformat_cvrp_instance(cost_matrix: np.ndarray, demands: np.ndarray) -> dict:
    """
    Reformat CVRP instance data into dict format required by OR-Tools solver.
    Parameters:
        cost_matrix (np.ndarray): pairwise cost between locations (shape: [num_locations, num_locations])
        demands (np.ndarray): demands of the customer nodes (shape: [num_locations])
    Returns:
        instance (dict): dictionary containing the reformatted instance data
    """
    assert len(cost_matrix) == len(demands), "Cost matrix and demands length mismatch"
    assert np.all(cost_matrix >= 0), "Cost matrix must have non-negative values"
    assert np.all(np.diag(cost_matrix) == 0), "Cost matrix diagonal must be zero (no self-loops)"
    assert np.all(demands.astype(int) == demands), "All demands must be integers for OR-Tools"
    assert np.all(demands[1:] > 0), "Customer demands must be positive"
    assert demands[0] == 0, "Depot demand must be zero"

    n_locations = len(demands)  # including depot
    return {
        "demands": demands,
        "cost_matrix": cost_matrix,
        "num_locations": n_locations,
        "num_vehicles": n_locations - 1,
        "depot_index": 0,
        "vehicle_capacity": VEHICLE_CAPACITY,
    }


def solve_with_ortools(cost_matrix: np.ndarray, demands: np.ndarray) -> dict:
    '''Solve a CVRP instance using OR-Tools and return the solution details.
    Parameters:
        cost_matrix (np.ndarray): pairwise cost between locations (shape: [num_locations, num_locations])
        demands (np.ndarray): demands of the customer nodes (shape: [num_locations])
    Returns:
        solution_details (dict): dictionary containing the solution details, including:
            - routes: list of routes, where each route is a list of node indices (including depot at start and end)
            - objective_cost: total cost of the solution (scaled back to original cost)
            - total_cost: total cost calculated from the routes (for verification, should match objective_cost)
    '''
    instance_data = reformat_cvrp_instance(cost_matrix, demands)
    # 1. Convert cost matrix to integers for OR-Tools internal solver
    cost_matrix_scaled = (instance_data["cost_matrix"] * COST_SCALER).astype(int)
    demands = instance_data["demands"].astype(int)
    capacity = int(instance_data["vehicle_capacity"])

    manager = pywrapcp.RoutingIndexManager(
        instance_data["num_locations"], instance_data["num_vehicles"], instance_data["depot_index"]
    )
    routing = pywrapcp.RoutingModel(manager)

    # 2. Distance Callback to lookup values directly from the matrix
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(cost_matrix_scaled[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 
        0, 
        [capacity] * instance_data["num_vehicles"], 
        True, 
        "Capacity"
    )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None

    solution_details = {"routes": []}
    total_cost = 0
    for vehicle_id in range(instance_data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_nodes = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_cost += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        route_nodes.append(manager.IndexToNode(index))

        if len(route_nodes) > 2:
            solution_details["routes"].append(route_nodes)
    
    solution_details["objective_cost"] = solution.ObjectiveValue() / COST_SCALER
    solution_details["total_cost"] = total_cost / COST_SCALER
    return solution_details



# EVALUATION
def evaluate_baseline_solver(n_instances_per_size: int, 
                             generator: CVRPGenerator,
                             n_cust_min=MIN_NUM_CUSTOMERS,
                             n_cust_max=MAX_NUM_CUSTOMERS,
                             step=1,
                             seed=None):
    ''' Evaluate the baseline OR-Tools solver on random CVRP instances
        for problem sizes from MIN_NUM_CUSTOMERS to MAX_NUM_CUSTOMERS.
        Returns the average distance per problem size.
        Parameters:
            n_instances_per_size (int): Number of instances to evaluate per problem size
            generator (CVRPGenerator): The generator for creating CVRP instances
            seed (int): seed for reproducible dataset generation
            n_cust_min (int): Minimum number of customers (inclusive)
            n_cust_max (int): Maximum number of customers (inclusive)
            step (int): Step size for iterating through problem sizes
        Returns:
            avg_dist_per_size (list): List of average distances per problem size
            avg_time_per_size (list): List of average solving times per problem size
            size_range (list): List of problem sizes evaluated
    '''
    avg_dist_per_size = [] # to store average distances per problem size
    avg_time_per_size = [] # to store average solving times per problem size
    rng = np.random.default_rng(seed) # Initialize random number generator with seed
    generator.rng = rng # Set the generator's random number generator to ensure reproducibility
    n_cust_range = np.array(range(n_cust_min, n_cust_max + 1, step))
    
    for n_customers in tqdm(n_cust_range, desc="Evaluating OR-Tools Solver"):
        avg_dist = []
        avg_time = []
        _, _, demands, cost_matrices, = generator.sample_batch(
            batch_size = n_instances_per_size, 
            num_locations = n_customers+1 # +1 for depot
        ) 
        for cost_matrix, demand in zip(cost_matrices, demands):
            start_time = time.time()
            solution = solve_with_ortools(cost_matrix, demand)
            end_time = time.time()
            avg_time.append(end_time - start_time)
            avg_dist.append(solution["objective_cost"])
        avg_dist_per_size.append(np.mean(avg_dist))
        avg_time_per_size.append(np.mean(avg_time))
    
    return avg_dist_per_size, avg_time_per_size, (n_cust_range+1).tolist()

