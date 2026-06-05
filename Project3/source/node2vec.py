import random
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm
from .parameters import (
    N2V_DIM,
    NEGATIVE,
    P,
    Q,
    NUM_WALKS,
    WALK_LENGTH,
)
import numpy as np
from sklearn.decomposition import PCA
import osmnx as ox
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

def build_node2vec(osmnx_graph, dimensions=N2V_DIM, 
                   p=P, q=Q, num_walks=NUM_WALKS, 
                   walk_length=WALK_LENGTH, negative=NEGATIVE, workers=4):
    """
    Build Node2Vec embeddings from an OSMnx graph.
    Parameters:
        osmnx_graph (nx.MultiDiGraph): The input graph from OSMnx
        dimensions (int): The dimensionality of the embeddings
        p (float): Return parameter (controls likelihood of immediately revisiting a node)
        q (float): In-out parameter (controls likelihood of exploring outward vs inward)
        num_walks (int): Number of random walks to simulate from each node
        walk_length (int): Length of each random walk
        negative (int): Number of negative samples for Word2Vec training
        workers (int): Number of worker threads to use for training the Word2Vec model
    Returns:
        embeddings (KeyedVectors): The learned node embeddings
    """
    print("Starting Node2Vec embedding generation with parameters:")
    print(f"  - Dimensions (dimensions): {dimensions}")
    print(f"  - Return parameter (p): {p}")
    print(f"  - In-out parameter (q): {q}")
    print(f"  - Number of walks (num_walks): {num_walks}")
    print(f"  - Walk length (walk_length): {walk_length}")
    print(f"  - Negative samples (negative): {negative}")
    print(f"  - Workers (workers): {workers}")
    

    print("--- Step 1: Parsing OSMnx Graph ---")
    # 1. Convert MultiDiGraph to DiGraph
    G = nx.DiGraph()
    for u, v, key, data in osmnx_graph.edges(keys=True, data=True):
        cost = data.get('fuel', 1.0)
        safe_cost = max(float(cost), 1e-6) # Prevent division by zero
        weight = 1.0 / safe_cost # Invert fuel cost (lower fuel = higher probability of traversal)
        
        if G.has_edge(u, v):
            # Keep the fastest/most efficient lane
            G[u][v]['weight'] = max(G[u][v]['weight'], weight)
        else:
            G.add_edge(u, v, weight=weight)

    print(f"Simplified to DiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    print("--- Step 2: Precomputing Transition Probabilities ---")
    alias_nodes = {}  # 1st-order transitions (for the very first step of a walk)
    alias_edges = {}  # 2nd-order transitions (for all subsequent steps)

    # Calculate 1st-order probabilities
    for node in G.nodes():
        neighbors = list(G.successors(node))
        if not neighbors:
            alias_nodes[node] = ([], [])
            continue
        unnorm_probs = [G[node][nbr]['weight'] for nbr in neighbors]
        norm_const = sum(unnorm_probs)
        alias_nodes[node] = (neighbors, [float(prob) / norm_const for prob in unnorm_probs])

    # Calculate 2nd-order probabilities (The core of Node2Vec)
    for t, v in G.edges():
        neighbors = list(G.successors(v))
        unnorm_probs = []
        for x in neighbors:
            weight_vx = G[v][x]['weight']
            if x == t:
                # Return probability (controlled by p)
                unnorm_probs.append(weight_vx / p)
            elif G.has_edge(t, x):
                # Local neighborhood / BFS probability
                unnorm_probs.append(weight_vx)
            else:
                # Outward exploration / DFS probability (controlled by q)
                unnorm_probs.append(weight_vx / q)
        
        norm_const = sum(unnorm_probs)
        if norm_const == 0:
            alias_edges[(t, v)] = ([], [])
        else:
            alias_edges[(t, v)] = (neighbors, [float(prob) / norm_const for prob in unnorm_probs])

    print("--- Step 3: Simulating Random Walks ---")
    walks = []
    nodes = list(G.nodes())
    
    for walk_iter in tqdm(range(num_walks), desc="Generating walks"):
            
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                
                # Use 1st order for the start, 2nd order for the rest
                if len(walk) == 1:
                    neighbors, probs = alias_nodes[curr]
                else:
                    prev = walk[-2]
                    neighbors, probs = alias_edges[(prev, curr)]
                
                # Stop if it's a dead end (should not happen in an SCC)
                if not neighbors:
                    break
                    
                # Pure Python random choice based on weights
                next_node = random.choices(neighbors, weights=probs, k=1)[0]
                walk.append(next_node)
                
            # Gensim expects strings
            walks.append([str(n) for n in walk])

    print("--- Step 4: Training Gensim Word2Vec Model ---")
    # Using modern Gensim Word2Vec implementation
    model = Word2Vec(
        sentences=walks, 
        vector_size=dimensions, 
        window=10, 
        min_count=1, 
        sg=1,          # Skip-gram architecture
        negative=negative, # number of negative samples
        workers=workers
    )
    
    print("Embeddings generated successfully!")
    return model.wv


# PCA Post-processing for zero-centering and rotation
def postprocess_embeddings(W_matrix, n_components=N2V_DIM):
    """
    Applies zero-centering and PCA rotation to eliminate anisotropy 
    while preserving 100% of the graph's structural information.
    """
    print(f"Original matrix shape: {W_matrix.shape}")
    print(f"Original average dot product: {np.mean(W_matrix @ W_matrix.T):.4f}")
    
    # Initialize PCA keeping ALL dimensions
    pca = PCA(n_components=n_components)
    
    # fit_transform automatically subtracts the mean and rotates the space
    print("Applying PCA for zero-centering and rotation...")
    W_optimized = pca.fit_transform(W_matrix)
    
    # Verify the math
    new_dot_products = W_optimized @ W_optimized.T
    print("--- Post-PCA Verification ---")
    print(f"Optimized average dot product: {np.mean(new_dot_products):.4f} (Should be near 0)")
    print(f"Total variance preserved: {sum(pca.explained_variance_ratio_)*100:.2f}% (Should be 100%)")
    
    return W_optimized

# Visualization via hierarchical clustering
def map_zurich_clusters(G_utm: nx.MultiDiGraph, 
                        W: np.ndarray, 
                        node_list: list, 
                        num_clusters: int = 50, 
                        metric: str = 'cosine', 
                        method: str = 'average',
                        fname="zurich_clusters.html") -> folium.Map:
    """
    Generates an interactive map where each structural cluster is an independent, 
    toggleable layer via Folium's LayerControl.
    Parameters:
        - G_utm: The original graph in UTM coordinates.
        - W: The node embedding matrix (shape: [num_nodes, embedding_dim]).
        - node_list: List of node IDs corresponding to the rows in W.
        - num_clusters: The number of clusters to form from the embeddings.
        - metric: The distance metric to use for clustering.
        - method: The linkage method to use for clustering.
        - fname: The filename to save the generated map.
    Returns:
        - A Folium Map object with clustered nodes.
    """
    # 1. Perform Hierarchical Clustering
    print(f"Performing hierarchical clustering into {num_clusters} clusters...")
    print(f"Distance metric: {metric}, Linkage method: {method}")
    distance_matrix = pdist(W, metric=metric)
    Z = linkage(distance_matrix, method=method)
    cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    
    node_to_cluster = {str(node): int(label) for node, label in zip(node_list, cluster_labels)}

    # 2. Generate dynamic hex color palette
    cmap = plt.get_cmap('tab20', num_clusters)
    cluster_colors = {
        cluster_id: mcolors.to_hex(cmap(i)) 
        for i, cluster_id in enumerate(range(1, num_clusters + 1))
    }

    # 3. Project to Lat/Lon for mapping
    print("Projecting graph to WGS84...")
    G_gps = ox.project_graph(G_utm, to_crs="epsg:4326")
    
    lats = [data['y'] for _, data in G_gps.nodes(data=True)]
    lons = [data['x'] for _, data in G_gps.nodes(data=True)]
    center_lat, center_lon = sum(lats)/len(lats), sum(lons)/len(lons)

    # 4. Initialize Map
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12, 
        tiles="Cartodb Positron", 
        prefer_canvas=True 
    )

    feature_groups = {}
    for c_id in range(1, num_clusters + 1):
        # Create the layer and name it
        fg = folium.FeatureGroup(name=f"Cluster {c_id}", show=True)
        feature_groups[c_id] = fg
        # Register the layer with the main map
        m.add_child(fg)

    # 5. Paint the nodes into their specific layers
    print("Populating map layers...")
    for node, data in G_gps.nodes(data=True):
        if str(node) in node_to_cluster:
            c_id = node_to_cluster[str(node)]
            hex_color = cluster_colors[c_id]
            
            # Attach the marker to the specific FeatureGroup, NOT the main map 'm'
            folium.CircleMarker(
                location=[data['y'], data['x']],
                radius=4,
                color=hex_color,
                weight=1,
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.8,
                popup=f"Node: {node}<br>Zone: {c_id}"
            ).add_to(feature_groups[c_id])

    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    # 6. Save output
    m.save('visualizations/' + fname)
    print(f"Success! Map saved to visualizations/{fname}.\nOpen this file in a web browser to interact with the map.")
    return m
