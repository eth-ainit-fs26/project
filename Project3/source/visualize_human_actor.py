import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import GridspecLayout, HBox, Button, GridspecLayout, Layout
import numpy as np
import torch
import re
from .cvrp import (
    MIN_NUM_CUSTOMERS,
    MAX_NUM_CUSTOMERS,
    VEHICLE_CAPACITY,
    DATALOADER
)
def create_expanded_button(description, button_style):
    return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))
    
def initialize_cvrp_data_for_actor_visualization(problem_size: int, seed: int):
    assert seed>=0, "Random seed must be a nonnegative integer"
    assert MIN_NUM_CUSTOMERS<=problem_size<=MAX_NUM_CUSTOMERS, f"Problem size must be within [{MIN_NUM_CUSTOMERS}, {MAX_NUM_CUSTOMERS}]"
    cvrp_instance = next(iter(DATALOADER(num_sample=1, batch_size=1, rng=np.random.default_rng(seed), problem_sizes_mean=problem_size)))
    depot_xy_plot = cvrp_instance[0][0][0].cpu().numpy()
    nodes_xy_plot = cvrp_instance[1][0].cpu().numpy()
    demands = torch.round(cvrp_instance[2][0].flatten()*cvrp_instance[3]).cpu().numpy().astype(int)
    psize = cvrp_instance[1][0].size(0)
    cap = VEHICLE_CAPACITY(problem_size)
    plot_info = {
        "customers": cvrp_instance[1][0].to('cpu').numpy(),
        "depot": cvrp_instance[0][0][0].to('cpu').numpy(), 
        "path": [0], 
        "next_node_probs": np.zeros(psize+1), 
        "total_dist": 0, 
        "remaining_capacity": cap, 
        "demands": demands,
        "done": False
    }
    return plot_info
def plot_single_actor_status(ax, customers, depot, demands, path, next_node_probs, total_dist, remaining_capacity, done):
    psize = customers.shape[0]
    nodes = np.vstack([depot, customers])
    visited = list(set(path).difference([0])) # visited customers
    not_visited = list(set(range(1, psize+1)).difference(visited))
    ax.scatter(nodes[visited][:, 0], nodes[visited][:, 1], marker='o', c='k') 
    ax.scatter(nodes[not_visited][:, 0], nodes[not_visited][:, 1], marker='o', c='b')
    ax.scatter(depot[0], depot[1], c='g', marker='s')
    for i, c in enumerate(customers):
        ax.text(c[0]+0.012, c[1] + 0.012, f'{i+1} ({demands[i]})', fontsize=8)
    
    # Plot path
    if len(path) > 1:
        path_coords = nodes[path]
        for ii, (start_pos, end_pos) in enumerate(zip(path_coords[:-1], path_coords[1:])):
            ax.arrow(
                start_pos[0], start_pos[1],
                end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
                color='k' if ii<len(path)-2 else 'r', head_width=0.02, head_length=0.02,
                length_includes_head=True, zorder=1, linewidth=0.1
            )
    buffer = 0.1
    ax.set_xlim([nodes[:,0].min()-buffer, nodes[:,0].max()+buffer]), ax.set_ylim([nodes[:,1].min()-buffer, nodes[:,1].max()+buffer])
    ax.set_xticks([]), ax.set_yticks([])
    title = f'Total Dist: {total_dist:.3f} | Remaining Capacity: {remaining_capacity} | Done: {done}'
    ax.set_title(title, fontsize=16)
    return ax
    
def plot_human_actor_interactive(problem_size=6, seed=21, figsize=(10,8)):    
    def find_button_node_id(button):
        # Fallback if regex fails (e.g. for "Depot Node")
        numbers = re.findall(r'[0-9]+', button.description)
        return int(numbers[0]) if numbers else 0

    def create_button_grid(legal):
        all_buttons = []
        # Descriptions logic
        descriptions = ['Go to Depot Node 0'] +  [f'Go to {ii+1}' for ii in range(len(legal)-1)]
        statuses = np.array(['danger', 'info'])
        button_colors = statuses[legal.astype(int)] # Ensure indices are integers
        
        grid = GridspecLayout(11, 4)
        
        # Setup Depot Button
        depot_button = create_expanded_button('Go to Depot Node 0', button_colors[0])
        all_buttons.append(depot_button)
        grid[0, :] = depot_button 
        
        # Setup Customer Buttons
        for ii in range(len(legal)-1):
            r, c = ii//4 + 1, ii%4
            # Fix indexing: descriptions and colors are offset by 1 for customers
            cb = create_expanded_button(descriptions[ii+1], button_colors[ii+1])
            grid[r, c] = cb
            all_buttons.append(cb)
            
        return grid, all_buttons

    def update_button_styles(all_buttons, legal):
        statuses = np.array(['danger', 'info'])
        # Ensure boolean legal array is converted to int indices (0 or 1)
        new_colors = statuses[legal.astype(int)] 
        for i, button in enumerate(all_buttons):
            button.button_style = new_colors[i]
    
    # --- 1. SETUP DATA ---
    legal = np.array([0] + [1]*problem_size) 
    CAP = VEHICLE_CAPACITY(problem_size)
    progress = initialize_cvrp_data_for_actor_visualization(problem_size, seed)
    
    # FIX A: Convert path to list to allow .append() operations
    if isinstance(progress['path'], np.ndarray):
        progress['path'] = progress['path'].tolist()

    grid, all_buttons = create_button_grid(legal)

    # --- 2. SETUP PLOTTING ---
    output = widgets.Output()
    
    # FIX B: Create the figure *inside* the function scope, fix name 'fix' -> 'fig'
    # We use io.off() to prevent double display of the plot in the notebook
    plt.ioff() 
    fig, ax = plt.subplots(1, figsize=figsize)
    plt.close(fig) # to avoid duplication
    plt.ion()
    
    # Initial Plot
    plot_single_actor_status(ax=ax, **progress)
    
    # Display the container
    display(HBox([output, grid]))
    
    # Render the initial plot into the output widget
    with output:
        display(fig)
        print(f'Current State: {progress["path"]}')

    def on_button_clicked(b):
        # Use nonlocal to modify variables from the outer scope
        nonlocal legal, progress, CAP, ax, grid, all_buttons, fig
        
        node_id = find_button_node_id(b)
        
        # Check legality
        if not legal[node_id]:
            with output:
                # We clear output to prevent stacking error messages
                output.clear_output(wait=True)
                # IMPORTANT: We must re-display the figure even when showing an error
                # otherwise the plot disappears when clear_output is called.
                display(fig) 
                print(f"Action 'Go to Node {node_id}' is illegal! Choose a legal action (blue button)!")
                print(f'Current State: {progress["path"]}')
            return
        
        # Check if done
        if progress.get('done', False):
            with output:
                output.clear_output(wait=True)
                display(fig)
                print("All steps completed. Final Distance: {:.2f}".format(progress['total_dist']))
                print(f'Final Solution: {progress["path"]}')
            return
        
        # --- 3. UPDATE LOGIC ---
        prev_node_id = progress['path'][-1]
        
        # Determine coordinates
        prev_node_xy = progress['depot'] if prev_node_id == 0 else progress['customers'][prev_node_id-1]
        node_xy = progress['depot'] if node_id == 0 else progress['customers'][node_id-1]
        
        dist = np.sqrt(np.sum((prev_node_xy - node_xy)**2))
        
        # Update progress
        progress['path'].append(node_id) # Now works because we converted to list
        progress['total_dist'] += dist
        
        if node_id == 0:
            progress['remaining_capacity'] = CAP
        else:
            progress['remaining_capacity'] -= progress['demands'][node_id-1]
            
        # Check if all customers visited (unique nodes in path minus depot must equal problem size)
        unique_visits = set(progress['path'])
        if 0 in unique_visits:
            unique_visits.remove(0)
        progress['done'] = (len(unique_visits) == problem_size) and progress['path'][-1]==0
            
        # --- 4. UPDATE CONSTRAINTS ---
        # Reset customer legality based on capacity
        # Note: We use .flatten() or reshape if demands shape mismatches legal shape
        legal[1:] = (progress['demands'] <= progress['remaining_capacity'])
        
        # Mark visited customers as illegal (cannot visit twice)
        for visited_node in progress['path']:
            if visited_node != 0: # We can always revisit depot (conditionally)
                legal[visited_node] = 0
        
        # Depot logic: Can go to depot if we are NOT currently at depot
        legal[0] = 1 if node_id > 0 else 0
        
        # --- 5. RE-RENDER ---
        update_button_styles(all_buttons, legal)
        
        with output:
            output.clear_output(wait=True)
            ax.clear() # Clear internal axes
            plot_single_actor_status(ax=ax, **progress) # Redraw plot on axes
            display(fig) # Explicitly re-send the figure to the widget
            print(f'Current State: {progress["path"]}')

    # Link events
    for b in all_buttons:
        b.on_click(on_button_clicked)