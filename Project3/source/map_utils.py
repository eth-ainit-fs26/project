import colorsys, collections
from typing import List, Optional
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph 
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import folium
from folium import plugins
from pyproj import Transformer
import shapely.geometry as sg
from shapely.ops import substring
import html
import branca

from .cvrp_generator import CVRPGenerator
from .parameters import ALPHA, BETA

def generate_diverse_colors(n: int) -> List[str]:
    """
    Generates a list of n visually distinct hex colors by distributing 
    hues evenly across the color wheel.
    """
    if n <= 0:
        return []
    
    colors = []
    for i in range(n):
        # Evenly divide the hue circle (0.0 to 1.0)
        hue = i / n
        
        # Keep Saturation and Value high for bright, vivid map lines
        # 0.85 saturation avoids faded colors; 0.90 value keeps them sharp
        saturation = 0.85
        value = 0.90
        
        # Convert HSV coordinates to RGB floating fractions
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert the RGB fractions to standard hex codes (#RRGGBB)
        hex_color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        colors.append(hex_color)
        
    return colors


def convert_node_info_to_dataframe(edge_indices, t, demands):
    num_locations = demands.shape[-1]
    df = pd.DataFrame({
        'location_id': np.arange(num_locations),
        'edge_index': edge_indices,
        'frac': t,
        'demand': demands
    })
    return df

def prepare_zurich_environment(alpha=ALPHA, beta=BETA):
    '''Prepares the road network environment for Zürich, Switzerland by downloading the graph, processing it, and computing the all-pairs shortest path matrix.
    Args:
        alpha (float): Weight for distance in fuel calculation.
        beta (float): Weight for travel time in fuel calculation.
    Returns:
        G_utm (nx.MultiDiGraph): The processed road network graph in UTM coordinates.
        apsp (np.ndarray): The all-pairs shortest path matrix.
        node_map (dict): A mapping from OSM node IDs to integer indices.
    '''
    print("===== Preparing the road network for Zürich, Switzerland =====")
    
    # 1. Download Zurich drive network
    print("Downloading Zürich driving network from OSM...")
    G = ox.graph_from_place("Zurich, Switzerland", network_type="drive")

    # 2. Restrict to largest strongly connected component to avoid 'inf' routing costs
    print("Restricting to largest strongly connected component...")
    G = ox.truncate.largest_component(G, strongly=True)

    # 3. Project to local UTM CRS for accurate planar coordinates (meters)
    print("Projecting graph to local UTM CRS (planar coordinates)...")
    G = ox.projection.project_graph(G)

    # 4. Compute edge speeds and impute missing values
    print("Adding road speed limits...")
    G = ox.routing.add_edge_speeds(G)

    # 5. Define congestion factors to simulate realistic traffic conditions
    print("Applying congestion factors to estimate travel time...")
    congestion_factors = {
        'motorway': 1.45, 'trunk': 1.60, 'primary': 1.90,    
        'secondary': 1.70, 'tertiary': 1.40, 'residential': 1.10 
    }

    for u, v, k, data in G.edges(data=True, keys=True):
        road_type = data.get('highway', 'residential')
        if isinstance(road_type, list): # Handle cases where 'highway' is a list
            road_type = [t for t in road_type if t!='unclassified'] # Remove 'unclassified' if present
            road_type = road_type[0]
        factor = congestion_factors.get(road_type, 1.3)
        if 'speed_kph' in data: # Reduce the speed based on the congestion factor
            data['speed_kph'] = data['speed_kph'] / factor
    
    # 6. Calculate edge travel times  
    print("Calculating edge travel times (in seconds) for all edges...")     
    G = ox.routing.add_edge_travel_times(G)
    
    # 7. Define custom cost weight
    print("Calculating fuel consumption (in liters) for all edges...")
    for u, v, k, data in G.edges(data=True, keys=True):
        distance = data['length']          # meters
        total_time = data['travel_time']   # penalized seconds
        data['fuel'] = alpha * distance + beta * total_time # add fuel attribute to each edge


    # 8. Eliminate parallel edges (keep only the one with the lowest fuel cost)
    print("Removing parallel edges (keeping only the lowest fuel cost edge for each (u, v) pair)...")
    edges_to_remove = []
    for u, v in set(G.edges()):
        # If there's more than one edge between node u and node v
        if len(G[u][v]) > 1:
            # Find the key (k) that has the absolute lowest fuel cost
            best_key = min(G[u][v].keys(), key=lambda k: G[u][v][k].get('fuel', float('inf')))
            # Add all other keys for this specific (u, v) pair to the removal list
            for k in G[u][v].keys():
                if k != best_key:
                    edges_to_remove.append((u, v, k))

    # Safely batch-delete the sub-optimal parallel edges
    G.remove_edges_from(edges_to_remove)

    print("Computing All-Pairs Shortest Path (APSP) Matrix...")
    # 9. Map nodes to integers for SciPy
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # 10. Build sparse adjacency matrix (handle parallel edges by taking the minimum cost)
    row, col, data_list = [], [], [] 
    for u, v, k, d in G.edges(keys=True, data=True): 
        i, j = node_to_idx[u], node_to_idx[v]
        row.append(i)
        col.append(j)
        data_list.append(d['fuel'])
        
    adj_matrix = sp.coo_matrix((data_list, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    # coo_matrix: create a sparse matrix in coordinate (COO) format from the edge list
    # .to_csr(): convert to compressed sparse row format for efficient computation

    # 10. Compute All-Pairs Shortest Path (APSP) using Dijkstra's algorithm
    # We do the computation only once and reuse the resulting matrix for all routing queries
    apsp_matrix = csgraph.shortest_path(adj_matrix, directed=True, method='D')
    
    # 11. Normalize node coordinates to [-1, 1] range for better numerical stability
    print("Adding normalized coordinates to node attributes...")
    x_coords = [data['x'] for node, data in G.nodes(data=True)]
    y_coords = [data['y'] for node, data in G.nodes(data=True)]
        
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_denom = x_max - x_min
    y_denom = y_max - y_min
    # Scale strictly to [-1, 1]
    for _, data in G.nodes(data=True):    
        data['x_norm'] = 2*(data['x'] - x_min) / x_denom - 1
        data['y_norm'] = 2*(data['y'] - y_min) / y_denom - 1

    print("Environment ready.")

    return G, apsp_matrix, node_to_idx

def augment_instance_with_solution(cvrp_instance: pd.DataFrame, cvrp_solution: List[List[int]]) -> pd.DataFrame:
    '''Given a CVRP instance and its solution, this function maps each stop to its assigned vehicle in the solution.
    
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon' and 'lat' columns.
        - cvrp_solution: A list of lists, where each inner list represents a vehicle route.
    Returns:
        - cvrp_instance_aug: The input DataFrame augmented with an 'assigned_vehicle' column indicating the vehicle assigned to each stop.
    '''

    # Target Stops featuring hoverable demand variables
    stop_to_vehicle = {stop: vid for vid, stops in enumerate(cvrp_solution) for stop in stops if stop != 0}
    stop_to_vehicle[0] = "depot"
    cvrp_instance_aug = cvrp_instance.copy(deep=True)
    cvrp_instance_aug['assigned_vehicle'] = cvrp_instance_aug.index.map(stop_to_vehicle)
    return cvrp_instance_aug

def visualize_cvrp_solution(G_utm: nx.MultiDiGraph,
                            generator: CVRPGenerator,
                            cvrp_instance: pd.DataFrame, 
                            cost_matrix: np.ndarray,
                            cvrp_solution: Optional[List[List[int]]], 
                            panel_title: str = "CVRP Optimization Solution",
                            fname: str = None) -> folium.Map:
    '''Generates an interactive map visualization of the CVRP instance and its solution.
        - Draws the base road network, original VRP nodes and the vehicle routes in the CVRP solution.
        - Computes route cost metrics (fuel consumption) and displays them in a floating
            information panel on the map.
    Args:
        - G_utm: The road network graph from OSMnx in utm
        - generator: The CVRP generator used to create the instance
        - cvrp_instance: DataFrame containing the CVRP locations with 'edge_index', 'frac', and 'demand' columns.
        - cost_matrix: A 2D numpy array containing the pre-computed costs between each pair of locations.
        - cvrp_solution: A list of lists, where each inner list represents a vehicle route. 
                         If None, only the base map and stops will be visualized.
        - panel_title: Title for the floating information panel on the map.
        - fname: Filename to save the generated interactive map HTML (skip if None)
    Returns:
        - folium.Map: The generated interactive map.
    '''
    if fname is not None:
        assert fname.endswith(".html"), "Output filename must end with '.html'."
    assert 'demand' in cvrp_instance.columns, "cvrp_instance must have a 'demand' column."
    assert 'edge_index' in cvrp_instance.columns, "cvrp_instance must have an 'edge_index' column."
    assert 'frac' in cvrp_instance.columns, "cvrp_instance must have a 'frac' column."

    edge_indices = cvrp_instance['edge_index'].values
    t = cvrp_instance['frac'].values

    # Add vehicle assignment to the instance for visualization
    if cvrp_solution is not None:
        instance = augment_instance_with_solution(cvrp_instance, cvrp_solution)
        # Assign colors to routes (black for depot, unique colors for each vehicle route)
        route_colors = generate_diverse_colors(len(cvrp_solution))
    else:
        instance = cvrp_instance.copy(deep=True)

    # Extract graph information for visualization
    _, edges_gdf = ox.graph_to_gdfs(G_utm, nodes=True, edges=True)
    all_edges = list(G_utm.edges(keys=True))

    # Initialize transformer from UTM to WGS84 for mapping
    transformer = Transformer.from_crs(G_utm.graph['crs'], "epsg:4326", always_xy=True)

    # Convert UTM coordinates of CVRP locations to lat/lon for mapping
    x_utm, y_utm = generator.compute_interpolated_coordinates(edge_indices, t, normalized=False)
    lons, lats = transformer.transform(x_utm, y_utm)
    wgs_points = [sg.Point(lon, lat) for lon, lat in zip(lons, lats)]
    points_gdf = gpd.GeoDataFrame(instance, geometry=wgs_points, crs="EPSG:4326")

    # 1. Initialize Canvas and draw background driving network
    m = edges_gdf.to_crs("EPSG:4326").explore(
        color="#555555",
        tiles="CartoDB positron", 
        style_kwds={"weight": 1.0, "opacity": 0.2},
        zoom_start=12,
        name="Base Street Network",
        show=False
    )
    
    # 2. Route Tracing and Visualization
    if cvrp_solution is not None:
        route_summaries_html = ""
        total_cvrp_cost = 0
        
        for vehicle_id, stop_sequence in enumerate(cvrp_solution):
            vehicle_color = route_colors[vehicle_id]
            route_cost = 0
            route_demand = sum(instance.loc[stop_sequence, 'demand'])
            geom_points = []
            
            # Initialize a unique Layer Group for this individual vehicle setup
            vehicle_layer = folium.FeatureGroup(name=f"Vehicle {vehicle_id} Stops").add_to(m)

            for idx in range(len(stop_sequence) - 1):
                # Extract the current and next VRP stops in the sequence
                loc_from = stop_sequence[idx]
                loc_to = stop_sequence[idx + 1]
                
                # Extract the pre-computed cost from the cost matrix for this leg of the route
                leg_cost = cost_matrix[loc_from, loc_to]
                route_cost += leg_cost

                # Extract the source and target nodes for the edges sampled to produce the two stops
                u_from, v_from, k_from = all_edges[edge_indices[loc_from]]
                u_to, v_to, k_to = all_edges[edge_indices[loc_to]]
                d_from = G_utm.edges[u_from, v_from, k_from]
                d_to = G_utm.edges[u_to, v_to, k_to]

                # --- GET TRUE GEOMETRY OR CREATE ONE IF STRAIGHT ---
                # If OSM stored a shortcut straight line, it lacks a geometry attribute. 
                # We create a fallback LineString using the bounding intersections.
                if 'geometry' in d_from:
                    geom_from = d_from['geometry']
                else:
                    geom_from = sg.LineString([(G_utm.nodes[u_from]['x'], G_utm.nodes[u_from]['y']), 
                                            (G_utm.nodes[v_from]['x'], G_utm.nodes[v_from]['y'])])
                    
                if 'geometry' in d_to:
                    geom_to = d_to['geometry']
                else:
                    geom_to = sg.LineString([(G_utm.nodes[u_to]['x'], G_utm.nodes[u_to]['y']), 
                                            (G_utm.nodes[v_to]['x'], G_utm.nodes[v_to]['y'])])

                # --- CASE A: Same-Edge Shortcut ---
                if edge_indices[loc_from] == edge_indices[loc_to] and t[loc_from] <= t[loc_to]:
                    # Cut the exact segment curve out of the main edge line using normalized fractions
                    sub_curve = substring(geom_from, t[loc_from], t[loc_to], normalized=True)
                    
                    # Transform coordinates to lat/lon and add to path
                    lons_c, lats_c = transformer.transform(*sub_curve.xy)
                    geom_points.extend(zip(lats_c, lons_c))
                    
                # --- CASE B: Macro Routing across Intersections ---
                else:
                    # Part 1: True curved exit segment (from current fraction t_from to end of street 1.0)
                    exit_curve = substring(geom_from, t[loc_from], 1.0, normalized=True)
                    lons_c, lats_c = transformer.transform(*exit_curve.xy)
                    geom_points.extend(zip(lats_c, lons_c))
                    
                    # Part 2: Macro Network Path Execution
                    # If v_from and u_to are the same node, we are already at the correct intersection and can skip routing
                    if v_from == u_to:
                        continue
                    try:
                        osm_path = nx.shortest_path(G_utm, source=v_from, target=u_to, weight="fuel")                    
                        path_geom = ox.routing.route_to_gdf(G_utm, osm_path)
                        for _, row in path_geom.iterrows():
                            if 'geometry' in row and row['geometry'] is not None:
                                # Convert network road geometries cleanly back to WGS84 degrees
                                geom_wgs84 = ox.projection.project_geometry(row['geometry'], crs=G_utm.graph['crs'], to_latlong=True)[0]
                                geom_points.extend([(c[1], c[0]) for c in list(geom_wgs84.coords)])
                    except nx.NetworkXNoPath:
                        raise ValueError(f"No path found in the road network from node {v_from} to node {u_to}. This should not happen in a strongly connected graph. Please check the graph connectivity and edge weights.")
                    
                    # Part 3: True curved entry segment (from start of target street 0.0 to destination fraction t_to)
                    entry_curve = substring(geom_to, 0.0, t[loc_to], normalized=True)
                    lons_c, lats_c = transformer.transform(*entry_curve.xy)
                    geom_points.extend(zip(lats_c, lons_c))
                    

            if geom_points:
                route_line = folium.PolyLine(geom_points, color=vehicle_color, weight=4.5, opacity=0.85, 
                                            name=f"Vehicle Route {vehicle_id}").add_to(vehicle_layer)
                # 2. Bind directional arrows along the path
                plugins.PolyLineTextPath(
                    route_line,
                    '      >      ',       # > as arrow symbol
                    repeat=True,    # Repeat the arrow along the entire route
                    attributes={
                        'fill': vehicle_color, 
                        'font-weight': 'bold', 
                        'font-size': '18px'
                    }
                ).add_to(vehicle_layer)

            # Render stops directly into the layer instance
            points_gdf.loc[list(set(stop_sequence).difference({0}))].explore(
                m=vehicle_layer,               
                color=vehicle_color, 
                marker_kwds={"radius": 8}, 
                tooltip=["location_id", "demand", "assigned_vehicle"],
                highlight=True
            )
            
            total_cvrp_cost += route_cost
            route_summaries_html += f"""
            <div style="margin-bottom: 8px; border-left: 4px solid {vehicle_color}; padding-left: 8px;">
                <b style="color: {vehicle_color};">Vehicle {vehicle_id}</b><br/>
                <b>Path:</b> {' → '.join(map(str, stop_sequence))}<br/>
                <b>Total Loaded Demand:</b> {route_demand} units<br/>
                <b>Route Cost:</b> {round(route_cost, 4)} (L)
            </div>
            """
    else:
        # Render all customer stops cleanly on the map base layer (excluding the depot at index 0)
        points_gdf.iloc[1:].explore(
            m=m,               
            color="crimson", 
            marker_kwds={"radius": 8}, 
            tooltip=["location_id", "demand"] ,
            highlight=True,
            name="Unassigned Customer Stops"
        )
    
    # 3. Draw depot
    folium.Marker(
        location=[lats[0], lons[0]],
        popup="Depot",
        icon=folium.Icon(color="black", icon="home")
    ).add_to(m)

    # 4. Collapsible Floating Information Panel
    if cvrp_solution is not None:
        floating_panel_html = f"""
        <div id="cvrp-floating-card" style="position: fixed; bottom: 30px; left: 30px; width: 320px;
            background-color: rgba(255, 255, 255, 0.95); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 8px;
            padding: 15px; font-family: sans-serif; font-size: 13px; color: #333333; z-index: 9999; line-height: 1.4;
            display: flex; flex-direction: column;">
            
            <h4 style="margin: 0; font-size: 15px; border-bottom: 2px solid #ddd; padding-bottom: 6px; cursor: pointer; user-select: none; display: flex; justify-content: space-between; align-items: center;" 
                onclick="togglePanel(this)">
                <span>{panel_title} ({len(cvrp_instance)} nodes)</span>
                <span id="panel-chevron" style="font-size: 11px; color: #666; transition: transform 0.2s;">▼</span>
            </h4>
            
            <div id="panel-main-content" style="max-height: 350px; overflow-y: auto; margin-top: 10px; display: block;">
                <div style="margin-bottom: 15px; font-weight: bold; background: #f0f0f0; padding: 6px; border-radius: 4px;">
                    Overall Solution Cost: <span style="color: #2b2b2b;">{round(total_cvrp_cost, 4)} (L)</span>
                </div>
                {route_summaries_html}
            </div>
        </div>
        
        <script>
            function togglePanel(headerElement) {{
                var content = document.getElementById('panel-main-content');
                var chevron = document.getElementById('panel-chevron');
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    chevron.style.transform = 'rotate(0deg)';
                }} else {{
                    content.style.display = 'none';
                    chevron.style.transform = 'rotate(-90deg)';
                }}
            }}
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(floating_panel_html))
    
    folium.LayerControl(collapsed=True).add_to(m)

    # 5. Injection Mechanism for Inter-Map Sync Engine
    sync_macro = folium.MacroElement()
    sync_macro._template = branca.element.Template("""
        {% macro script(this, kwargs) %}
        var leafletMap = {{this._parent.get_name()}};
        
        function broadCastMovement() {
            window.parent.postMessage({
                type: 'LEAFLET_SYNC_EVENT',
                center: leafletMap.getCenter(),
                zoom: leafletMap.getZoom()
            }, '*');
        }
        
        // Listen to map events and forward parameters up to parent wrapper
        leafletMap.on('move', broadCastMovement);
        
        // Recieve sync commands coming downward from parent wrapper
        window.addEventListener('message', function(event) {
            if (event.data.type === 'LEAFLET_SYNC_EVENT') {
                leafletMap.off('move', broadCastMovement); // Temporarily detach listener to break loop
                leafletMap.setView(event.data.center, event.data.zoom, {animate: false});
                leafletMap.on('move', broadCastMovement);  // Reattach listener
            }
        });
        {% endmacro %}
    """)
    m.add_child(sync_macro)


    if fname is not None:
        m.save('visualizations/' + fname)
        print(f"Map visualization saved to visualizations/{fname}.\nOpen this file in a web browser to interact with the map.")

    return m

def visualize_two_cvrp_solutions(G_utm: nx.Graph,
                                 generator: CVRPGenerator,
                                 cvrp_instance: pd.DataFrame, 
                                 cost_matrix: np.ndarray,
                                 cvrp_solution_1: List[List[int]], 
                                 cvrp_solution_2: List[List[int]], 
                                 panel_title_1: str = "CVRP Solution 1",
                                 panel_title_2: str = "CVRP Solution 2",
                                 fname: str = "interactive_map.html"):
    """
    Generates a split-screen HTML visualization to compare two CVRP solutions side-by-side.
    """
    # 1. Generate both maps using copies of the dataframe to prevent column collisions
    m1 = visualize_cvrp_solution(
        G_utm, generator, cvrp_instance, cost_matrix, cvrp_solution_1,
        fname=None, panel_title=panel_title_1
    )
    
    m2 = visualize_cvrp_solution(
        G_utm, generator, cvrp_instance, cost_matrix, cvrp_solution_2,
        fname=None, panel_title=panel_title_2
    )
    
    # 2. Extract and protect source strings
    srcdoc1 = html.escape(m1.get_root().render())
    srcdoc2 = html.escape(m2.get_root().render())
    
    # 3. Construct master DOM wrapper with cross-iframe cross-talk script
    split_screen_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>CVRP Dual-View Optimization Comparison Panel</title>
        <style>
            body, html {{
                margin: 0; padding: 0; height: 100%; width: 100%; 
                display: flex; font-family: sans-serif; overflow: hidden;
                background-color: #222;
            }}
            .map-wrapper {{
                flex: 1; height: 100%; position: relative;
            }}
            .center-axis-divider {{
                width: 5px; background-color: #1a1a1a; z-index: 10000;
                box-shadow: 0 0 10px rgba(0,0,0,0.7);
            }}
            iframe {{
                width: 100%; height: 100%; border: none; display: block;
            }}
            
            /* Floating Control Widget Container */
            .control-hub {{
                position: fixed;
                top: 20px;
                left: 12.5%;
                transform: translateX(-50%);
                z-index: 20000;
                background: rgba(255, 255, 255, 0.95);
                padding: 10px 20px;
                border-radius: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                gap: 12px;
                user-select: none;
            }}
            .control-label {{
                font-size: 13px;
                font-weight: bold;
                color: #333;
            }}
            
            /* Sleek CSS Toggle Switch Styling */
            .switch {{
                position: relative;
                display: inline-block;
                width: 46px;
                height: 24px;
            }}
            .switch input {{
                opacity: 0; width: 0; height: 0;
            }}
            .slider {{
                position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
                background-color: #ccc; transition: .3s; border-radius: 24px;
            }}
            .slider:before {{
                position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
                background-color: white; transition: .3s; border-radius: 50%;
            }}
            input:checked + .slider {{
                background-color: #2196F3;
            }}
            input:checked + .slider:before {{
                transform: translateX(22px);
            }}
        </style>
    </head>
    <body>

        <div class="control-hub">
            <span class="control-label" id="statusLabel">Synchronize Views</span>
            <label class="switch">
                <input type="checkbox" id="syncToggle" checked onchange="updateToggleUI()">
                <span class="slider"></span>
            </label>
        </div>

        <div class="map-wrapper">
            <iframe id="mapFrameLeft" srcdoc="{srcdoc1}"></iframe>
        </div>
        
        <div class="center-axis-divider"></div>
        
        <div class="map-wrapper">
            <iframe id="mapFrameRight" srcdoc="{srcdoc2}"></iframe>
        </div>

        <script>
            var leftIframe = document.getElementById('mapFrameLeft');
            var rightIframe = document.getElementById('mapFrameRight');
            var syncToggle = document.getElementById('syncToggle');
            var statusLabel = document.getElementById('statusLabel');
            
            // Simple UI feedback update
            function updateToggleUI() {{
                if (syncToggle.checked) {{
                    statusLabel.style.color = '#2196F3';
                    statusLabel.innerText = "Views Synchronized";
                }} else {{
                    statusLabel.style.color = '#666';
                    statusLabel.innerText = "Views Independent";
                }}
            }}
            
            // Initialize UI Text Color on load
            updateToggleUI();
            
            // Central hub capturing synchronization flags and conditionally routing them
            window.addEventListener('message', function(event) {{
                // CRITICAL STEP: Only pass data if the checkbox is actively selected!
                if (syncToggle.checked && event.data && event.data.type === 'LEAFLET_SYNC_EVENT') {{
                    if (event.source === leftIframe.contentWindow) {{
                        rightIframe.contentWindow.postMessage(event.data, '*');
                    }} else if (event.source === rightIframe.contentWindow) {{
                        leftIframe.contentWindow.postMessage(event.data, '*');
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
        
    # 4. Save compilation file
        
    with open(fname, "w", encoding="utf-8") as f:
        f.write(split_screen_html)
    
    print(f"Dual-view comparison map saved to visualizations/{fname}.\nOpen this file in a web browser to interact with the synchronized maps.")