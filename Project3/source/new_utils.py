import colorsys
from typing import List
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as sg
import osmnx as ox
import networkx as nx
import folium


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

def augment_instance_with_nearest_nodes(cvrp_instance: pd.DataFrame, G: nx.Graph) -> None:
    '''Maps each location in the CVRP instance to its nearest graph node for accurate routing.
        - Uses OSMnx's nearest_nodes function to find the closest graph node for each location based on longitude and latitude.
        - Adds a new column 'nearest_node' to the cvrp_instance DataFrame that contains the ID of the nearest graph node for each location.
    
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon' and 'lat' columns.
        - G: The road network graph from OSMnx.
    Returns:
        - None (the function modifies the cvrp_instance DataFrame in place)
    '''
    assert 'lon' in cvrp_instance.columns and 'lat' in cvrp_instance.columns, "cvrp_instance must have 'lon' and 'lat' columns"
    cvrp_instance['nearest_node'] = ox.nearest_nodes(G, X=cvrp_instance.lon, Y=cvrp_instance.lat)

def compute_instance_geometries(cvrp_instance: pd.DataFrame, G: nx.Graph, nodes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    '''Computes geometry data for the CVRP instance for visualization.
        - Creates Point geometries for each location and LineString geometries connecting them to their nearest nodes.
    
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon' and 'lat' columns.
        - G: The road network graph from OSMnx.
        - nodes_gdf: GeoDataFrame of graph nodes for visualization.
    Returns:
        - pd.DataFrame: A DataFrame containing the geometry data for the CVRP instance.
    '''

    assert 'nearest_node' in cvrp_instance.columns, \
        "The variable `cvrp_instance` must have 'nearest_node' column for geometry calculations. Run `augment_instance_with_nearest_nodes()` first."

    # Augment the VRP instance with geometries for visualization
    df_geom = cvrp_instance[['node_id']]
    df_geom['node_geom'] = cvrp_instance.apply(lambda row: sg.Point(row.lon, row.lat), axis=1)
    df_geom['nn_geom'] = nodes_gdf.loc[cvrp_instance.nearest_node, 'geometry'].values
    df_geom['connector_line'] = df_geom.apply(lambda row: sg.LineString([row.node_geom, row.nn_geom]), axis=1)

    return df_geom


def augment_instance_with_solution(cvrp_instance: pd.DataFrame, cvrp_solution: List[List[int]], G: nx.Graph) -> None:
    '''Given a CVRP instance and its solution, this function maps each stop to its assigned vehicle in the solution.
    
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon' and 'lat' columns.
        - cvrp_solution: A list of lists, where each inner list represents a vehicle route.
        - G: The road network graph from OSMnx.
    Returns:
        - None (the function modifies the cvrp_instance DataFrame in place).
    '''
    

    # Target Stops featuring hoverable demand variables
    stop_to_vehicle = {stop: vid for vid, stops in enumerate(cvrp_solution) for stop in stops if stop != 0}
    stop_to_vehicle[0] = "depot"
    cvrp_instance['assigned_vehicle'] = cvrp_instance.index.map(stop_to_vehicle)




def generate_pairwise_distance_matrix(cvrp_instance: pd.DataFrame, G: nx.Graph) -> np.ndarray:
    assert 'nearest_node' in cvrp_instance.columns, \
        "The variable `cvrp_instance` must have 'nearest_node' column for distance calculations. Run `augment_instance_with_nearest_nodes()` first."
    n = len(cvrp_instance) # problem size + 1
    fuel_matrix = np.zeros((n, n)) # row=source, col=destination
    # Compute pairwise travel costs via Dijkstra's algorithm
    for i, source_node in cvrp_instance.nearest_node.items():
        # Calculate cheapest path lengths from source to all other graph nodes
        costs_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight='fuel')
        for j, target_node in cvrp_instance.nearest_node.items():
            # retrieve the fuel cost from source_node to target_node (infinity if no path exists)
            fuel_matrix[i, j] = costs_from_source.get(target_node, np.inf)
    return fuel_matrix



def generate_interactive_map(cvrp_instance: pd.DataFrame, 
                             cvrp_solution: List[List[int]], 
                             G: nx.Graph,
                             nodes_gdf: gpd.GeoDataFrame,
                             edges_gdf: gpd.GeoDataFrame,
                             fname: str = "interactive_map.html") -> folium.Map:
    '''Generates an interactive map visualization of the CVRP instance and its solution.
        - Draws the base road network, original VRP nodes, connector lines to the nearest
            graph nodes, and the actual vehicle routes based on the CVRP solution.
        - Computes route cost metrics (fuel consumption) and displays them in a floating
            information panel on the map.
    Args:
        - cvrp_instance: DataFrame containing the CVRP locations with 'lon', 'lat', and 'demand' columns.
        - cvrp_solution: A list of lists, where each inner list represents a vehicle route.
        - G: The road network graph from OSMnx.
        - nodes_gdf: GeoDataFrame of graph nodes for visualization.
        - edges_gdf: GeoDataFrame of graph edges for visualization.
        - fname: Filename to save the generated interactive map HTML.
    Returns:
        - None (the function saves the interactive map as an HTML file and prints a message with instructions to view it).
    '''
    assert fname[-5:] == ".html", "Output filename must end with '.html'."
    assert 'demand' in cvrp_instance.columns, "cvrp_instance must have a 'demand' column."
    
    if 'assigned_vehicle' not in cvrp_instance.columns:
        augment_instance_with_solution(cvrp_instance, cvrp_solution, G)
    
    df_geom = compute_instance_geometries(cvrp_instance, G, nodes_gdf)

    route_colors = generate_diverse_colors(len(cvrp_solution))
    df_geom['MarkerColor'] = [
        "#000000" if idx == 0 else route_colors[assigned_vehicle]
        for idx, assigned_vehicle in cvrp_instance['assigned_vehicle'].items()
    ]


    # 1. Initialize the "canvas" for our map and draw the base driving network
    m = edges_gdf.explore(
        color="#555555",
        tiles="CartoDB positron", 
        style_kwds={"weight": 1.0, "opacity": 0.2},
        popup=True,
        name="Base Street Network"
    )


    # 2. Draw the original vrp nodes
    points_gdf = gpd.GeoDataFrame(cvrp_instance, geometry=df_geom.node_geom, crs="EPSG:4326")
    points_gdf.explore(
        m=m, color=df_geom.MarkerColor, marker_kwds={"radius": 8}, 
        tooltip=["node_id", "demand", "assigned_vehicle"], name="Original VRP Nodes"
    )

    # 3. Draw connector lines from original locations to their nearest graph nodes
    connectors_gdf = gpd.GeoDataFrame(geometry=df_geom.connector_line, crs="EPSG:4326")
    m = connectors_gdf.explore(    
        m=m,
        color="#777777",
        style_kwds={"weight": 2, "dashArray": "5, 5"},
        name="Links to Drop Off Locations"
    )


    # 4. Draw the actual vehicle routes based on the CVRP solution
    #    and compute route cost metrics along the way
    route_summaries_html = ""
    total_cvrp_cost = 0
    
    for vehicle_id, stop_sequence in enumerate(cvrp_solution):
        vehicle_color = df_geom.MarkerColor.loc[stop_sequence[1]] 
        full_route_nodes = []
        route_fuel = 0
        route_demand = sum(cvrp_instance.loc[stop_sequence, 'demand'])

        for idx in range(len(stop_sequence) - 1):
            orig_node, dest_node = cvrp_instance.loc[stop_sequence[idx:idx+2]].nearest_node.values
            
            path_segment = ox.routing.shortest_path(G, orig_node, dest_node, weight="fuel")

            # Accumulate fuel consumption for our metric summary
            # Edges between path segment nodes hold the fuel attribute
            for u, v in zip(path_segment[:-1], path_segment[1:]):
                # Graphs can be MultiDiGraphs, grab the quickest edge attribute safely
                route_fuel += min(d.get('fuel', 0) for d in G[u][v].values())

            if idx == 0:
                full_route_nodes.extend(path_segment)
            else:
                full_route_nodes.extend(path_segment[1:])
                
        route_gdf = ox.routing.route_to_gdf(G, full_route_nodes)
        m = route_gdf.explore(
            m=m, 
            color=vehicle_color, 
            tooltip=False,
            style_kwds={"weight": 4.5, "opacity": 0.85}, 
            name=f"Vehicle Route {vehicle_id}"
        )

        total_cvrp_cost += route_fuel
        
        # Format text metadata block for this specific route string loop
        route_summaries_html += f"""
        <div style="margin-bottom: 8px; border-left: 4px solid {vehicle_color}; padding-left: 8px;">
            <b style="color: {vehicle_color};">Vehicle {vehicle_id}</b><br/>
            <b>Path:</b> {' → '.join(map(str, stop_sequence))}<br/>
            <b>Total Loaded Demand:</b> {route_demand} units<br/>
            <b>Route Cost:</b> {round(route_fuel, 3)} L
        </div>
        """

    # 5. Inject floaing info element panel
    floating_panel_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; 
        left: 30px; 
        width: 320px;
        max-height: 400px;
        overflow-y: auto;
        background-color: rgba(255, 255, 255, 0.95);
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 13px;
        color: #333333;
        z-index: 9999;
        line-height: 1.4;
    ">
        <h4 style="margin: 0 0 10px 0; font-size: 16px; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
            CVRP Optimization Solution
        </h4>
        <div style="margin-bottom: 15px; font-weight: bold; background: #f0f0f0; padding: 6px; border-radius: 4px;">
            Overall Solution Cost: <span style="color: #2b2b2b;">{round(total_cvrp_cost, 3)} (L) </span>
        </div>
        {route_summaries_html}
    </div>
    """

    # Append the floating CSS/HTML card to the interactive Leaflet root frame
    m.get_root().html.add_child(folium.Element(floating_panel_html))

    # Add layer control to toggle visibility of different map layers
    folium.LayerControl(collapsed=True).add_to(m)

    # Save map locally
    m.save(fname)
    print(f"Visualization saved as '{fname}'.\nOpen this file in a web browser to explore the interactive map and solution details.")


