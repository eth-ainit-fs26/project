import colorsys, collections
from typing import List
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


def convert_node_info_to_dataframe(coords_utm, demands, edge_indices, t):
    num_locations = coords_utm.shape[0]
    df = pd.DataFrame({
        'location_id': np.arange(num_locations),
        'demand': demands,
        'x_utm': coords_utm[:, 0],
        'y_utm': coords_utm[:, 1],
        'edge_index': edge_indices,
        'frac': t,
    })
    return df

def prepare_zurich_environment(alpha=1e-4, beta=1e-3):
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

    print("Computing All-Pairs Shortest Path (APSP) Matrix...")
    # 8. Map nodes to integers for SciPy
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # 9. Build sparse adjacency matrix (handle parallel edges by taking the minimum cost)
    edges_dict = collections.defaultdict(lambda: float('inf'))
    for u, v, k, d in G.edges(keys=True, data=True): 
        # (u,v,k) is the edge index in OSM, where u is the source node id,
        # v is the target node id, and k is the key for parallel edges
        i, j = node_to_idx[u], node_to_idx[v]
        if d['fuel'] < edges_dict[(i, j)]:
            edges_dict[(i, j)] = d['fuel']
            
    row, col, data_list = [], [], [] # rows, columns, and data for COO format
    for (i, j), cost in edges_dict.items():
        row.append(i)
        col.append(j)
        data_list.append(cost)
        
    adj_matrix = sp.coo_matrix((data_list, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    # coo_matrix: create a sparse matrix in coordinate (COO) format from the edge list
    # .to_csr(): convert to compressed sparse row format for efficient computation

    # 10. Compute All-Pairs Shortest Path (APSP) using Dijkstra's algorithm
    # We do the computation only once and reuse the resulting matrix for all routing queries
    apsp_matrix = csgraph.shortest_path(adj_matrix, directed=True, method='D')
    
    print("Environment ready.")
    return G, apsp_matrix, node_to_idx

def augment_instance_with_solution(cvrp_instance: pd.DataFrame, cvrp_solution: List[List[int]]) -> None:
    '''Given a CVRP instance and its solution, this function maps each stop to its assigned vehicle in the solution.
    
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon' and 'lat' columns.
        - cvrp_solution: A list of lists, where each inner list represents a vehicle route.
    Returns:
        - None (the function modifies the cvrp_instance DataFrame in place).
    '''
    

    # Target Stops featuring hoverable demand variables
    stop_to_vehicle = {stop: vid for vid, stops in enumerate(cvrp_solution) for stop in stops if stop != 0}
    stop_to_vehicle[0] = "depot"
    cvrp_instance['assigned_vehicle'] = cvrp_instance.index.map(stop_to_vehicle)


def visualize_cvrp_solution(cvrp_instance: pd.DataFrame, 
                            cost_matrix: np.ndarray,
                            cvrp_solution: List[List[int]], 
                            G_utm: nx.Graph,
                            fname: str = "interactive_map.html") -> folium.Map:
    '''Generates an interactive map visualization of the CVRP instance and its solution.
        - Draws the base road network, original VRP nodes and the vehicle routes in the CVRP solution.
        - Computes route cost metrics (fuel consumption) and displays them in a floating
            information panel on the map.
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon', 'lat', and 'demand' columns.
        - cvrp_solution: A list of lists, where each inner list represents a vehicle route.
        - cost_matrix: A 2D numpy array containing the pre-computed costs between each pair of locations.
        - G_utm: The road network graph from OSMnx in utm
        - fname: Filename to save the generated interactive map HTML.
    Returns:
        - None (the function saves the interactive map as an HTML file and prints a message with instructions to view it).
    '''
    assert fname[-5:] == ".html", "Output filename must end with '.html'."
    assert 'demand' in cvrp_instance.columns, "cvrp_instance must have a 'demand' column."
    assert 'edge_index' in cvrp_instance.columns, "cvrp_instance must have an 'edge_index' column."
    assert 'frac' in cvrp_instance.columns, "cvrp_instance must have a 'frac' column."
    assert 'x_utm' in cvrp_instance.columns and 'y_utm' in cvrp_instance.columns, "cvrp_instance must have 'x_utm' and 'y_utm' columns for UTM coordinates."

    edge_indices = cvrp_instance['edge_index'].values
    t = cvrp_instance['frac'].values

    # Add vehicle assignment to the instance if not already present (for visualization purposes)
    if 'assigned_vehicle' not in cvrp_instance.columns:
        augment_instance_with_solution(cvrp_instance, cvrp_solution)
    
    # Assign colors to routes (black for depot, unique colors for each vehicle route)
    route_colors = generate_diverse_colors(len(cvrp_solution))
    
    # Extract graph information for visualization
    _, edges_gdf = ox.graph_to_gdfs(G_utm, nodes=True, edges=True)
    all_edges = list(G_utm.edges(keys=True))

    # Initialize transformer from UTM to WGS84 for mapping
    transformer = Transformer.from_crs(G_utm.graph['crs'], "epsg:4326", always_xy=True)

    # Convert UTM coordinates of CVRP locations to lat/lon for mapping
    lons, lats = transformer.transform(cvrp_instance.x_utm.values, cvrp_instance.y_utm.values)
    wgs_points = [sg.Point(lon, lat) for lon, lat in zip(lons, lats)]
    points_gdf = gpd.GeoDataFrame(cvrp_instance, geometry=wgs_points, crs="EPSG:4326")

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
    route_summaries_html = ""
    total_cvrp_cost = 0
    
    for vehicle_id, stop_sequence in enumerate(cvrp_solution):
        vehicle_color = route_colors[vehicle_id]
        route_cost = 0
        route_demand = sum(cvrp_instance.loc[stop_sequence, 'demand'])
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
                try:
                    osm_path = nx.shortest_path(G_utm, source=v_from, target=u_to, weight="fuel")                    
                    path_geom = ox.routing.route_to_gdf(G_utm, osm_path)
                    for _, row in path_geom.iterrows():
                        if 'geometry' in row and row['geometry'] is not None:
                            # Convert network road geometries cleanly back to WGS84 degrees
                            geom_wgs84 = ox.projection.project_geometry(row['geometry'], crs=G_utm.graph['crs'], to_latlong=True)[0]
                            geom_points.extend([(c[1], c[0]) for c in list(geom_wgs84.coords)])
                except nx.NetworkXNoPath:
                    pass
                
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
    
    # 3. Draw depot
    folium.Marker(
        location=[lats[0], lons[0]],
        popup="Depot",
        icon=folium.Icon(color="black", icon="home")
    ).add_to(m)

    # 4. Floating Information Panel Output
    floating_panel_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 320px; max-height: 400px; overflow-y: auto;
        background-color: rgba(255, 255, 255, 0.95); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 8px;
        padding: 15px; font-family: sans-serif; font-size: 13px; color: #333333; z-index: 9999; line-height: 1.4;">
        <h4 style="margin: 0 0 10px 0; font-size: 16px; border-bottom: 2px solid #ddd; padding-bottom: 5px;">CVRP Optimization Solution ({len(cvrp_instance)} nodes)</h4>
        <div style="margin-bottom: 15px; font-weight: bold; background: #f0f0f0; padding: 6px; border-radius: 4px;">
            Overall Solution Cost: <span style="color: #2b2b2b;">{round(total_cvrp_cost, 4)} (L)</span>
        </div>
        {route_summaries_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(floating_panel_html))
    folium.LayerControl(collapsed=True).add_to(m)
    m.save(fname)
