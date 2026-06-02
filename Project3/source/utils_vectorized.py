import colorsys
from typing import List
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as sg
import osmnx as ox
import networkx as nx
import folium
from pyproj import Transformer


def convert_utm_to_latlon(coords, G):
    """
    Converts a NumPy array of UTM coordinates [[x1, y1], [x2, y2], ...]
    back into a NumPy array of [[lat1, lon1], [lat2, lon2], ...]
    
    Parameters:
        coords (np.ndarray): Shape (N, 2), containing UTM X and Y coordinates.
        G (nx.MultiDiGraph): The projected Zurich graph used to generate the points.
    """
    # 1. Get the current Projected CRS and the Target Geographic CRS from the graph
    crs_projected = G.graph['crs']
    crs_wgs84 = "epsg:4326"
    
    # 2. Initialize a fast, vectorized coordinate Transformer
    # always_xy=True ensures the input/output order matches standard (X, Y) -> (Lon, Lat)
    transformer = Transformer.from_crs(crs_projected, crs_wgs84, always_xy=True)
    
    # 3. Separate the X and Y columns
    utm_x = coords[:, 0]
    utm_y = coords[:, 1]
    
    # 4. Perform the vectorized mathematical transformation
    longitudes, latitudes = transformer.transform(utm_x, utm_y)
    
    # 5. Stack them together into a clean [Latitude, Longitude] matrix for plotting tools
    latlon_coords = np.column_stack((latitudes, longitudes))
    
    return latlon_coords

def generate_diverse_colors(n: int) -> List[str]:
    """Generates a list of n visually distinct hex colors."""
    if n <= 0: return []
    colors = []
    for i in range(n):
        hue, saturation, value = i / n, 0.85, 0.90
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
    return colors


def compute_instance_geometries(cvrp_instance: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    '''
    Computes geometry points for the sub-edge CVRP locations.
    Since coords are pre-calculated in UTM, we store them as Points.
    '''
    df_geom = pd.DataFrame(index=cvrp_instance.index)
    # Convert metric UTM coordinates to Point geometries
    df_geom['node_geom'] = cvrp_instance.apply(lambda row: sg.Point(row.utm_x, row.utm_y), axis=1)
    return df_geom


def augment_instance_with_solution(cvrp_instance: pd.DataFrame, cvrp_solution: List[List[int]]) -> None:
    '''Maps each stop to its assigned vehicle in the solution matrix.'''
    stop_to_vehicle = {stop: vid for vid, stops in enumerate(cvrp_solution) for stop in stops if stop != 0}
    stop_to_vehicle[0] = "depot"
    cvrp_instance['assigned_vehicle'] = cvrp_instance.index.map(stop_to_vehicle)


def generate_interactive_map(generator,
                             edge_indices: np.ndarray,
                             t: np.ndarray,
                             demands: np.ndarray,
                             cvrp_solution: List[List[int]], 
                             G: nx.Graph,
                             edges_gdf: gpd.GeoDataFrame,
                             fname: str = "interactive_map.html") -> folium.Map:
    '''
    Generates an interactive map visualization tailored for sub-edge CVRP instances.
    Traces paths accurately along structural edge transitions.
    '''
    assert fname[-5:] == ".html", "Output filename must end with '.html'."
    
    # 1. Build the instance DataFrame containing tracking attributes
    all_edges = list(G.edges(keys=True))
    
    # Calculate exact coordinates in UTM
    px = generator.u_x[edge_indices] + t * (generator.v_x[edge_indices] - generator.u_x[edge_indices])
    py = generator.u_y[edge_indices] + t * (generator.v_y[edge_indices] - generator.u_y[edge_indices])
    
    cvrp_instance = pd.DataFrame({
        'node_id': np.arange(len(edge_indices)),
        'utm_x': px,
        'utm_y': py,
        'demand': demands
    })
    
    # Augment details
    augment_instance_with_solution(cvrp_instance, cvrp_solution)
    df_geom = compute_instance_geometries(cvrp_instance, G)

    route_colors = generate_diverse_colors(len(cvrp_solution))
    df_geom['MarkerColor'] = [
        "#000000" if idx == 0 else route_colors[int(assigned_vehicle)]
        for idx, assigned_vehicle in cvrp_instance['assigned_vehicle'].items()
    ]

    # Convert Projected UTM points back to WGS84 for Folium rendering
    crs_projected = G.graph['crs']
    transformer = Transformer.from_crs(crs_projected, "epsg:4326", always_xy=True)
    
    # Transform base nodes for display
    lons, lats = transformer.transform(cvrp_instance.utm_x.values, cvrp_instance.utm_y.values)
    wgs_points = [sg.Point(lon, lat) for lon, lat in zip(lons, lats)]

    # 2. Initialize the canvas and project background network
    # We project edges back to lat/lon for folium compatibility
    edges_wgs = edges_gdf.to_crs("EPSG:4326")
    m = edges_wgs.explore(
        color="#555555",
        tiles="CartoDB positron", 
        style_kwds={"weight": 1.0, "opacity": 0.2},
        tooltip=False,
        popup=False,
        name="Base Street Network"
    )

    # 3. Draw the interactive GeoDataFrame drop-off markers
    points_gdf = gpd.GeoDataFrame(cvrp_instance, geometry=wgs_points, crs="EPSG:4326")
    points_gdf.explore(
        m=m, color=df_geom.MarkerColor, marker_kwds={"radius": 8}, 
        tooltip=["node_id", "demand", "assigned_vehicle"], popup=True, name="VRP Stops"
    )

    # 4. Route Tracing along Sub-Edges and Intersections
    route_summaries_html = ""
    total_cvrp_cost = 0
    
    for vehicle_id, stop_sequence in enumerate(cvrp_solution):
        vehicle_color = route_colors[vehicle_id]
        route_cost = 0
        route_demand = sum(cvrp_instance.loc[stop_sequence, 'demand'])
        geom_points = []

        for idx in range(len(stop_sequence) - 1):
            loc_from = stop_sequence[idx]
            loc_to = stop_sequence[idx + 1]
            
            edge_from = all_edges[edge_indices[loc_from]]
            edge_to = all_edges[edge_indices[loc_to]]
            
            u_from, v_from, _ = edge_from
            u_to, v_to, _ = edge_to
            
            # Sub-Edge Edge Case Shortcut (Same Street)
            if edge_indices[loc_from] == edge_indices[loc_to] and t[loc_from] <= t[loc_to]:
                t_steps = np.linspace(t[loc_from], t[loc_to], 10)
                sub_x = generator.u_x[edge_indices[loc_from]] + t_steps * (generator.v_x[edge_indices[loc_from]] - generator.u_x[edge_indices[loc_from]])
                sub_y = generator.u_y[edge_indices[loc_from]] + t_steps * (generator.v_y[edge_indices[loc_from]] - generator.u_y[edge_indices[loc_from]])
                lo, la = transformer.transform(sub_x, sub_y)
                geom_points.extend(zip(la, lo))
                route_cost += (t[loc_to] - t[loc_from]) * generator.edge_costs[edge_indices[loc_from]]
                
            # Regular Macro Route via Intersections
            else:
                # Part 1: Segment to Exit Edge
                t_steps = np.linspace(t[loc_from], 1.0, 10)
                sub_x = generator.u_x[edge_indices[loc_from]] + t_steps * (generator.v_x[edge_indices[loc_from]] - generator.u_x[edge_indices[loc_from]])
                sub_y = generator.u_y[edge_indices[loc_from]] + t_steps * (generator.v_y[edge_indices[loc_from]] - generator.u_y[edge_indices[loc_from]])
                lo, la = transformer.transform(sub_x, sub_y)
                geom_points.extend(zip(la, lo))
                route_cost += (1 - t[loc_from]) * generator.edge_costs[edge_indices[loc_from]]
                
                # Part 2: Macro Network Path Routing
                try:
                    osm_path = nx.shortest_path(G, source=v_from, target=u_to, weight="custom_cost")
                    route_cost += generator.apsp_matrix[generator.u_idx[edge_indices[loc_from]], generator.v_idx[edge_indices[loc_to]]]
                    
                    path_geom = ox.routing.route_to_gdf(G, osm_path)
                    for _, row in path_geom.iterrows():
                        if 'geometry' in row and row['geometry'] is not None:
                            geom_wgs84 = ox.projection.project_geometry(row['geometry'], crs=G.graph['crs'], to_latlong=True)[0]
                            geom_points.extend([(c[1], c[0]) for c in list(geom_wgs84.coords)])
                except nx.NetworkXNoPath:
                    pass
                
                # Part 3: Arrival Segment up to target position
                t_steps = np.linspace(0.0, t[loc_to], 10)
                sub_x = generator.u_x[edge_indices[loc_to]] + t_steps * (generator.v_x[edge_indices[loc_to]] - generator.u_x[edge_indices[loc_to]])
                sub_y = generator.u_y[edge_indices[loc_to]] + t_steps * (generator.v_y[edge_indices[loc_to]] - generator.u_y[edge_indices[loc_to]])
                lo, la = transformer.transform(sub_x, sub_y)
                geom_points.extend(zip(la, lo))
                route_cost += t[loc_to] * generator.edge_costs[edge_indices[loc_to]]

        if geom_points:
            folium.PolyLine(geom_points, color=vehicle_color, weight=4.5, opacity=0.85, name=f"Vehicle Route {vehicle_id}").add_to(m)

        total_cvrp_cost += route_cost
        route_summaries_html += f"""
        <div style="margin-bottom: 8px; border-left: 4px solid {vehicle_color}; padding-left: 8px;">
            <b style="color: {vehicle_color};">Vehicle {vehicle_id}</b><br/>
            <b>Path:</b> {' → '.join(map(str, stop_sequence))}<br/>
            <b>Total Loaded Demand:</b> {route_demand} units<br/>
            <b>Route Cost:</b> {round(route_cost, 2)}
        </div>
        """

    # 5. Floating Information Panel Output
    floating_panel_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 320px; max-height: 400px; overflow-y: auto;
        background-color: rgba(255, 255, 255, 0.95); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 8px;
        padding: 15px; font-family: sans-serif; font-size: 13px; color: #333333; z-index: 9999; line-height: 1.4;">
        <h4 style="margin: 0 0 10px 0; font-size: 16px; border-bottom: 2px solid #ddd; padding-bottom: 5px;">CVRP Optimization Solution</h4>
        <div style="margin-bottom: 15px; font-weight: bold; background: #f0f0f0; padding: 6px; border-radius: 4px;">
            Overall Solution Cost: <span style="color: #2b2b2b;">{round(total_cvrp_cost, 2)}</span>
        </div>
        {route_summaries_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(floating_panel_html))
    folium.LayerControl(collapsed=True).add_to(m)
    m.save(fname)
    return m