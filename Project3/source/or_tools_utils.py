"""
The MIT License

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

import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import Optional, List
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from tqdm import tqdm
import time

# OR tools parameters
from source.cvrp import (
    MIN_NUM_CUSTOMERS, 
    MAX_NUM_CUSTOMERS,
    VEHICLE_CAPACITY,
    DATALOADER
)

COORD_SCALER = 100000 # OR-Tools works with integers
TIME_LIMIT = 1 # seconds, for OR-Tools solver

# For printing colors
class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Helper function to reformat the instance for OR-Tools
def reformat_cvrp_instance(depot_xy: np.ndarray, 
                           node_xy: np.ndarray, 
                           node_demand: np.ndarray,
                           demand_scaler: float) -> dict:
    """
    Reformat CVRP instance data into dict format required by OR-Tools solver.
    Parameters:
        depot_xy (np.ndarray): coordinates of the depot (shape: [1, 2])
        node_xy (np.ndarray): coordinates of the customer nodes (shape: [num_nodes, 2])
        node_demand (np.ndarray): SCALED demands of the customer nodes (shape: [num_nodes])
        demand_scaler (float): factor used to scale down the demands
    Returns:
        instance (dict): dictionary containing the reformatted instance data
    """
    # Combine depot and node data
    # The depot (index 0) has 0 demand and a wide open time window [0, HORIZON]
    all_xy = np.vstack([depot_xy, node_xy])
    all_demands = np.concatenate([[0], node_demand])
    # multiply all demands by demand_scaler and round to the nearest integer
    all_demands = np.round(all_demands * demand_scaler).astype(int)

    n_cust = len(all_xy) -1 # PROBLEM_SIZE
    return {
        "locations": all_xy,
        "demands": all_demands,
        "num_locations": n_cust+1,
        "num_vehicles": n_cust,
        "depot_index": 0,
        "vehicle_capacity": VEHICLE_CAPACITY(n_cust),
    }


def print_cvrp_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print('='*30 + " SOLUTION DETAILS (Routes formatted as \033[1m\033[91mNodeID\033[0m(\033[94mLoad\033[0m)) " + '='*30)
    print(f"Method: PATH_CHEAPEST_ARC + GREEDY_DESCENT (greedy local search) | Time limit: {TIME_LIMIT} seconds\n")

    total_distance = 0
    total_load = 0
    route_id = 1 # use this instead of vehicle_id for since some vehicles may be unused
    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        plan_output = f"Vehicle {route_id}: "
        route_id += 1
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f"{Color.BOLD+Color.RED}{node_index}{Color.END}({Color.BLUE}{int(route_load)}{Color.END}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {Color.BOLD+Color.RED}{manager.IndexToNode(index)}{Color.END}({Color.BLUE}{int(route_load)}{Color.END})\n"
        plan_output += f"Distance = {route_distance/COORD_SCALER}, Load = {int(route_load)}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance/COORD_SCALER}")
    print(f"Total load of all routes: {int(total_load)}")
    # print(f"Objective: {solution.ObjectiveValue()/COORD_SCALER}")

def solve_cvrp_instance_with_ortools(instance_data: dict, print_solution: bool = False, time_limit: int = None):
    """
    Solves the CVRP using Google OR-Tools.
    Parameters:
        instance_data (dict): dictionary containing the CVRP instance data returned by reformat_instance()
    Returns:
        solution_details (dict or None): dictionary containing the solution details, or None if no solution found
    """
    locations_scaled = (instance_data["locations"] * COORD_SCALER).astype(int)
    demands = (instance_data["demands"]).astype(int)
    capacity = int(instance_data["vehicle_capacity"])

    manager = pywrapcp.RoutingIndexManager(
        len(locations_scaled), instance_data["num_vehicles"], instance_data["depot_index"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(np.linalg.norm(locations_scaled[from_node] - locations_scaled[to_node]))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 
        0, # null capacity slack
        [capacity] * instance_data["num_vehicles"], # vehicle maximum capacities
        True, # start cumul to zero
        "Capacity"
    )
    
    # Set up search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Sample different first solution strategies and local search methods to add variety to tw generation

    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT

    search_parameters.time_limit.seconds = int(TIME_LIMIT) if time_limit is None else time_limit

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None
    if print_solution:
        print_cvrp_solution(instance_data, manager, routing, solution)

    solution_details = {"routes": []}
    total_distance = 0
    for vehicle_id in range(instance_data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_times = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        # Add final depot stop
        node_index = manager.IndexToNode(index)
        route_nodes.append(node_index)

        if len(route_nodes) > 2:
            solution_details["routes"].append(route_nodes)
    
    solution_details["objective_cost"] = solution.ObjectiveValue() / COORD_SCALER
    solution_details["total_distance"] = total_distance / COORD_SCALER
    return solution_details


# VISUALIZATION
def visualize_solution_cvrp(instance_data: dict, solution: dict, figsize=(10,8)) -> None:
    """Visualizes the CVRP solution as a map with routes
    Parameters:
        instance_data (dict): dictionary containing the CVRPTW instance data returned by reformat_instance()
        solution (dict): dictionary containing the solution details returned by solve_with_ortools()
    """
    locations = instance_data["locations"]
    depot_idx = instance_data["depot_index"]
    
    plt.figure(figsize=figsize)
    
    customer_locs = np.delete(locations, depot_idx, axis=0)
    plt.scatter(customer_locs[:, 0], customer_locs[:, 1], c='cyan', label='Customers', s=50, zorder=2)
    plt.scatter(locations[depot_idx, 0], locations[depot_idx, 1], c='red', s=150, label='Depot', zorder=3)
    
    # Augmented labels for all nodes
    for i, loc in enumerate(locations):
        if i == depot_idx:
            label = "" # f"{i} (Depot)"
        else:
            demand = instance_data["demands"][i]
            label = f'{i} ({int(demand)})'
        plt.text(loc[0], loc[1] + 0.01, label, fontsize=10)
        
    colors = plt.cm.jet(np.linspace(0, 1, len(solution["routes"])))
    for i, route in enumerate(solution["routes"]):
        color = colors[i]
        for j in range(len(route) - 1):
            start_node = locations[route[j]]
            end_node = locations[route[j+1]]
            plt.arrow(
                start_node[0], start_node[1],
                end_node[0] - start_node[0], end_node[1] - start_node[1],
                color=color, head_width=0.01, head_length=0.01,
                length_includes_head=True, zorder=1
            )
    
    plt.title(f"CVRP Solution Map | Total Distance: {solution['total_distance']:.2f}", fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



# EVALUATION
def evaluate_baseline_solver(n_instances_per_size: int=20, seed=None):
    ''' Evaluate the baseline OR-Tools solver on random CVRP instances
        for problem sizes from MIN_NUM_CUSTOMERS to MAX_NUM_CUSTOMERS.
        Returns the average distance per problem size.
        Parameters:
            n_instances_per_size (int): Number of instances to evaluate per problem size
            seed: seed for reproducible dataset generation
        Returns:
            avg_dist_per_size (list): List of average distances per problem size
            avg_time_per_size (list): List of average solving times per problem size
    '''
    avg_dist_per_size = [] # to store average distances per problem size
    avg_time_per_size = [] # to store average solving times per problem size
    rng = np.random.default_rng(seed) # Initialize random number generator with seed
    for n_customers in tqdm(range(MIN_NUM_CUSTOMERS, MAX_NUM_CUSTOMERS + 1)):
        avg_dist = []
        avg_time = []
        train_loader = DATALOADER(num_sample=n_instances_per_size, batch_size=1, problem_sizes_mean=n_customers, rng=rng)
        for depot_xy, node_xy, node_demand, demand_scaler in train_loader:
            depot_xy, node_xy, node_demand = depot_xy[0].to('cpu'), node_xy[0].to('cpu'), node_demand[0,:,0].to('cpu')
            cvrp_data_or = reformat_cvrp_instance(depot_xy=depot_xy, node_xy=node_xy, node_demand=node_demand, demand_scaler=demand_scaler)
            start_time = time.time()
            solution = solve_cvrp_instance_with_ortools(cvrp_data_or, print_solution=False)
            end_time = time.time()
            avg_time.append(end_time - start_time)
            avg_dist.append(solution["objective_cost"])
        avg_dist_per_size.append(np.mean(avg_dist))
        avg_time_per_size.append(np.mean(avg_time))
    
    return avg_dist_per_size, avg_time_per_size

