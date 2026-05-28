# ===================== VISUALIZE ACTOR SOLVERS ALONGSIDE OR TOOLS =====================
import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from source.utilities import augment_xy_data_by_8_fold
from source.cvrp import GROUP_ENVIRONMENT, DATALOADER
from source.or_tools_utils import (
    reformat_cvrp_instance,
    solve_cvrp_instance_with_ortools,
)

def _visualize_solution_cvrp(ax: plt.Axes, solver_name: str, instance_data: dict, solution: dict) -> None:
    locations = instance_data["locations"]
    depot_idx = instance_data["depot_index"]
    customer_locs = np.delete(locations, depot_idx, axis=0)
    ax.scatter(customer_locs[:, 0], customer_locs[:, 1], c='cyan', label='Customers', s=50, zorder=2)
    ax.scatter(locations[depot_idx, 0], locations[depot_idx, 1], c='red', s=150, label='Depot', zorder=3)
    
    # Augmented labels for all nodes
    for i, loc in enumerate(locations):
        if i == depot_idx:
            label = "" # f"{i} (Depot)"
        else:
            demand = instance_data["demands"][i]
            label = f'{i} ({int(demand)})'
        ax.text(loc[0], loc[1] + 0.01, label, fontsize=10)
        
    colors = plt.cm.jet(np.linspace(0, 1, len(solution["routes"])))
    for i, route in enumerate(solution["routes"]):
        color = colors[i]
        for j in range(len(route) - 1):
            start_node = locations[route[j]]
            end_node = locations[route[j+1]]
            ax.arrow(
                start_node[0], start_node[1],
                end_node[0] - start_node[0], end_node[1] - start_node[1],
                color=color, head_width=0.01, head_length=0.01,
                length_includes_head=True, zorder=1
            )
    
    ax.set_title(f"{solver_name} | Total Distance: {solution['total_distance']:.4f}")
    ax.grid(True, linestyle='--', alpha=0.6)

def visualize_all_solutions(instance_data: dict, solver_names: List[str], solutions: List[dict], figsize=(10,8), 
                            title: str="Solutions from All Solvers on a Random CVRP Instance", title_font_size: int=18) -> None:
    num_solvers = len(solver_names)
    assert num_solvers == len(solutions), "Number of solvers must equal to the number of solutions provided"
    
    fig, ax = plt.subplots(1, num_solvers, figsize=figsize, sharex=True, sharey=True)
    for i, (sol_name, sol) in enumerate(zip(solver_names, solutions)):
        _visualize_solution_cvrp(ax[i], sol_name, instance_data, sol)
    
    fig.suptitle(title, fontsize=title_font_size)
    plt.tight_layout()
    plt.show()

def actor_solve(grouped_actor, depot_xy, node_xy, node_demand, augment=True):
    DEVICE = grouped_actor.device
    batch_s = depot_xy.size(0) # batch size
    group_s = node_xy.size(1)  # problem size = n_cust
    
    if augment: # 8 fold Augmented
        depot_xy = augment_xy_data_by_8_fold(depot_xy) # shape = (8*batch, 1, 2)
        node_xy = augment_xy_data_by_8_fold(node_xy) # shape = (8*batch, n_cust, 2)
        node_demand = node_demand.repeat(8, 1, 1) # shape = (8*batch, n_cust, 2)
        batch_s = 8*batch_s
    
    # Step 0
    env = GROUP_ENVIRONMENT(depot_xy, node_xy, node_demand)
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
    
    if not augment:
        max_reward, argmax_group_idx = reward.max(dim=1) # best rollouts; shape = (batch,), (batch,)
        best_sols = group_state.selected_node_list[torch.arange(batch_s), argmax_group_idx, :] # shape = (batch, selected_count)
    else:
        batch_s = round(batch_s/8)
        reward = reward.reshape(8, batch_s, group_s) # shape = (8, batch, group)
        reward, argmax_group_idx = reward.max(dim=2) # best rollouts among groups; shape = (8, batch), (8, batch)
        max_reward, argmax_aug_idx = reward.max(dim=0) # best rollouts among augmentations; shape = (batch,), (batch,)
        
        
        selected = group_state.selected_node_list.reshape(8, batch_s, group_s, -1) # shape = (8, batch, group, selected_count)
        selected = selected[argmax_aug_idx, torch.arange(batch_s), :, :] # shape = (batch, group, selected_count)
        argmax_group_idx_final = argmax_group_idx[argmax_aug_idx, torch.arange(batch_s)] # shape = (batch,)
        best_sols = selected[torch.arange(batch_s), argmax_group_idx_final, :] # shape = (batch, selected_count)
        
    return -max_reward.to('cpu')[0], best_sols.to('cpu')[0]

def flat_to_routes(flat_solution: List[int] | np.ndarray | torch.Tensor) -> List[List[int]]:
    """Converts a "flat" VRP solution array into a list of individual routes."""
    if isinstance(flat_solution, np.ndarray) or isinstance(flat_solution, torch.Tensor):
        solution_list = flat_solution.tolist()
    else:
        solution_list = flat_solution
    routes: List[List[int]] = []
    current_route: List[int] = []

    while len(solution_list)>0 and solution_list[0] == 0:
        solution_list.pop(0)
    for node in solution_list:
        if node != 0:
            current_route.append(node)
        else:
            if current_route:
                routes.append([0]+current_route+[0])
            current_route = []
    return routes


def solve_one_random_instance_using_all_solvers_and_visualize(actor , problem_size: int, seed: int, figsize=(16,4)):
    def check_solution_feasbility(or_data, or_sol, tol=1e-4):
        "Check the feasibility of a solution, assuming data and solution are given in or tool format"
        total_distance = 0
        for route in or_sol['routes']:
            # check capacity constraints
            assert sum(or_data['demands'][route]) <= or_data['vehicle_capacity'], "Capacity constraint violated!"
            # check tour length correctness
            route_coords = or_data['locations'][route]
            total_distance += sum(np.sqrt(np.sum((route_coords[1:] - route_coords[:-1])**2, axis=1)))
        assert np.isclose(total_distance, or_sol['total_distance'], atol=tol)
    
    assert seed >= 0, "Random seed must be nonnegative"
    loader = DATALOADER(num_sample=1, batch_size=1, problem_sizes_mean=problem_size, rng=np.random.default_rng(seed))
    instance = next(iter(loader))
    depot_xy, node_xy, node_demand, demand_scaler = instance
    
    # Do some reformatting for OR-Tools
    depot_xy, node_xy, node_demand = depot_xy[0].to('cpu'), node_xy[0].to('cpu'), node_demand[0,:,0].to('cpu') 
    cvrp_data_or = reformat_cvrp_instance(depot_xy=depot_xy, node_xy=node_xy, 
                                                         node_demand=node_demand, demand_scaler=demand_scaler)
    # Call OR solver
    or_solution = solve_cvrp_instance_with_ortools(cvrp_data_or, print_solution=False)
    # Call actor
    pomo_dist, pomo_sol = actor_solve(actor, *instance[:-1], augment=False)
    pomo_aug_dist, pomo_aug_sol = actor_solve(actor, *instance[:-1], augment=True)
    
    # Convert POMO solutions to OR formal
    pomo_or = {
        'routes': flat_to_routes(pomo_sol),
        'objective_cost': float(pomo_dist),
        'total_distance': float(pomo_dist)
    }
    pomo_aug_or = {
        'routes': flat_to_routes(pomo_aug_sol),
        'objective_cost': float(pomo_aug_dist),
        'total_distance': float(pomo_aug_dist)
    }
    check_solution_feasbility(cvrp_data_or, pomo_or)
    check_solution_feasbility(cvrp_data_or, pomo_aug_or)
    # plot visualization
    all_solvers = ['OR Tools', 'POMO', 'POMO+8xAug']
    scores = [or_solution['total_distance'], pomo_or['total_distance'], pomo_aug_or['total_distance']]
    ranking = np.argsort(scores)
    leaderboard = np.array(all_solvers)[np.argsort(scores)].tolist()
    comp1 = ' == ' if np.isclose(scores[ranking[0]], scores[ranking[1]], atol=1e-4) else ' --> '
    comp2 = ' == ' if np.isclose(scores[ranking[1]], scores[ranking[2]], atol=1e-4) else ' --> '
    final_ranking = leaderboard[0] + comp1 + leaderboard[1] + comp2 + leaderboard[2]
    
    title = "Solutions from All Solvers on a Random CVRP Instance\n"
    title += f'Ranking: {final_ranking}'
    visualize_all_solutions(cvrp_data_or, all_solvers, [or_solution, pomo_or, pomo_aug_or], 
                            title = title, figsize=figsize)
    
    return cvrp_data_or, or_solution, pomo_or, pomo_aug_or
