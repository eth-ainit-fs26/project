import random
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm
from .parameters import (
    N2V_DIM,
    P,
    Q,
    NUM_WALKS,
    WALK_LENGTH,
)

def build_node2vec(osmnx_graph, dimensions=N2V_DIM, 
                   p=P, q=Q, num_walks=NUM_WALKS, 
                   walk_length=WALK_LENGTH, workers=4):
    """
    Build Node2Vec embeddings from an OSMnx graph.
    Parameters:
        osmnx_graph (nx.MultiDiGraph): The input graph from OSMnx
        dimensions (int): The dimensionality of the embeddings
        p (float): Return parameter (controls likelihood of immediately revisiting a node)
        q (float): In-out parameter (controls likelihood of exploring outward vs inward)
        num_walks (int): Number of random walks to simulate from each node
        walk_length (int): Length of each random walk
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
        workers=workers
    )
    
    print("Embeddings generated successfully!")
    return model.wv