"""
    TSP Heuristics that return the value of the length of the calculated tour
"""
import itertools
import networkx as nx
from networkx.algorithms.approximation import christofides, greedy_tsp
import numpy as np

# disabeling capitalized variable names because of nx standard G for graph
# pylint: disable=C0103

# disabeling generator suggestions as we want to use also for each loops
# pylint: disable=R1728

def christo(instance, tree=None):
    """
        Christofides TSP Heuristic Length
        Inputs:
            instance: a TSP Instance codified in concorde style
            tree: a Minimum Spanning Tree Instance of the Graph as
                nx.Graph
    """
    G = get_nx_graph(instance['node_coordinates'])
    christo_tour = christofides(G, weight='weight', tree=tree)
    tmp = [(christo_tour[i],christo_tour[(i+1)]) for i in range(len(christo_tour)-1)]
    christo_len = sum([G[u][v]['weight'] for u,v in tmp])
    return christo_len


def mst(instance):
    """
        Minimum Spanning Tree Calculation
        Inputs:
            A TSP Instance codified in concorde style
    """
    G = get_nx_graph(instance['node_coordinates'])
    mstree = nx.minimum_spanning_tree(G)
    mslen = sum([G[u][v]['weight'] for u,v in mstree.edges])
    return mslen, mstree

def onetree(instance, tree=None):
    """
        1-Tree TSP lower bound length
        Inputs:
            instance: a TSP Instance codified in concorde style
            tree: a Minimum Spanning Tree Instance of the Graph as
                nx.Graph
    """
    G = get_nx_graph(instance['node_coordinates'])
    M = nx.Graph()

    if tree is None:
        _, tmp = mst(instance)
        M = tmp
    else:
        M = tree

    candidates = []
    for node in M.nodes:
        neighbors = list(M.neighbors(node))
        if len(neighbors) == 1:
            candidates += [(node, c) for c in M.neighbors(neighbors[0]) if c != node]

    best = (-1,-1,100000)
    for u,v in candidates:
        edge = G[u][v]['weight']
        if edge < best[2]:
            best = (u,v,edge)

    M.add_edge(best[0],best[1])
    M[best[0]][best[1]]['weight'] = best[2]
    one_tree_len = sum([M[u][v]['weight'] for u,v in M.edges])
    return one_tree_len

def fi(instance):
    """
        Farthest Insertion TSP Heuristic O(n²)
        returns length of farthest insertion tour
        Inputs:
            instance: a TSP Instance codified in concorde style

    """
    G = get_nx_graph(instance['node_coordinates'])
    unvisited = list(G.nodes)

    start = max(G.edges(data='weight'), key = lambda tuple: tuple[2])

    unvisited.remove(start[0])
    unvisited.remove(start[1])

    tour = [int(start[0]), int(start[1]), int(start[0])]

    while len(unvisited) > 0:
        # find closest node to tour
        possibilities = list(itertools.product(tour,unvisited))
        new_edge = max(possibilities, key=lambda tuple: G[tuple[0]][tuple[1]]['weight'])
        new_node = new_edge[1]

        unvisited.remove(new_node)

        # see possible tours
        c_len = tour_len(tour, G)
        tours = [
            (c_len - G[tour[i]][tour[i+1]]['weight'])
            + G[tour[i]][new_node]['weight']
            + G[tour[i+1]][new_node]['weight']
            for i in range(len(tour)-1)
        ]
        tour.insert(np.argmin(tours)+1, new_node)
    len_tour = tour_len(tour, G)
    return len_tour

def greedy(instance):
    """
        Greedy TSP Heuristic length
        Inputs:
            instance: a TSP Instance codified in concorde style

    """
    G = get_nx_graph(instance['node_coordinates'])
    greedy_tour = greedy_tsp(G, weight='weight')
    tmp = [(greedy_tour[i],greedy_tour[(i+1)]) for i in range(len(greedy_tour)-1)]
    greedy_len = sum([G[u][v]['weight'] for u,v in tmp])
    return greedy_len

def mstheu(instance, tree=None)-> nx.Graph():
    """
        Minimum Spanning Tree TSP Heuristic length
        Inputs:
            instance: a TSP Instance codified in concorde style
    """
    if tree is None:
        _, tmp = mst(instance)
        tree = tmp

    G = get_nx_graph(instance['node_coordinates'])

    MST = nx.MultiGraph()
    MST.add_weighted_edges_from(tree.edges(data='weight'))
    MST.add_weighted_edges_from(tree.edges(data='weight'))

    euler_circuit = hierholzer(MST)
    hamil_circuit = hamilton(euler_circuit)

    tour_edges = [(hamil_circuit[i], hamil_circuit[i+1]) for i in range(len(hamil_circuit)-1)]
    mstheu_len = sum([G[u][v]['weight'] for u,v in tour_edges])

    return mstheu_len

def hierholzer(G: nx.MultiGraph)-> list:
    """
        Hierholzer Algorithm
        Inputs:
            G: nx.Graph()
    """
    B = nx.eulerian_circuit(G)
    walk = [tup[0] for tup in list(B)]
    walk.append(walk[0])
    return walk

def hamilton(walk):
    """
        Hamilotonian Circle on a graph
        Inputs:
            walk on a graph
    """
    hamil = []
    for item in walk:
        if not item in hamil:
            hamil.append(item)
        else:
            continue
    hamil.append(hamil[0])
    return hamil

def ni(instance):
    """
        Nearest Insertion TSP Heuristic O(n²)
        returns length of nearest insertion tour
        Input:
            instance: a TSP Instance codified in concorde style
    """
    G = get_nx_graph(instance['node_coordinates'])
    unvisited = list(G.nodes)

    edges = list(G.edges(data='weight'))
    edges.sort(key=lambda tuple: tuple[2])
    edges = np.array(edges)
    start = edges[0]

    unvisited.remove(start[0])
    unvisited.remove(start[1])

    tour = [int(start[0]), int(start[1]), int(start[0])]

    while len(unvisited) > 0:
        # find closest node to tour
        possibilities = list(itertools.product(tour,unvisited))
        new_edge = min(possibilities, key=lambda tuple: G[tuple[0]][tuple[1]]['weight'])
        new_node = new_edge[1]

        unvisited.remove(new_node)

        # see possible tours
        c_len = tour_len(tour, G)
        tours = [
            (c_len - G[tour[i]][tour[i+1]]['weight'])
            + G[tour[i]][new_node]['weight']
            + G[tour[i+1]][new_node]['weight']
            for i in range(len(tour)-1)
        ]
        tour.insert(np.argmin(tours)+1, new_node)
    len_tour = tour_len(tour, G)
    return len_tour

def nn(instance):
    """
        Nearest Neighbor TSP Heuristic length
        Inputs:
            instance: a TSP Instance codified in concorde style
    """
    G = get_nx_graph(instance['node_coordinates'])
    current = list(G.nodes)[0]
    unvisited = list(G.nodes)
    unvisited.remove(current)
    tour = [current]
    while len(unvisited) > 0:
        neighbors = set(G.neighbors(current))
        possibilities = set(unvisited).intersection(neighbors)
        edges = [(current, possibility) for possibility in possibilities]
        best = (edges[0][0], edges[0][1], G[edges[0][0]][edges[0][1]]['weight'])
        for edge in edges:
            if G[edge[0]][edge[1]]['weight'] < best [2]:
                best = (edge[0], edge[1], G[edge[0]][edge[1]]['weight'])
        current = best[1]
        tour.append(current)
        unvisited.remove(current)

    tour.append(list(G.nodes)[0])
    tmp = [(tour[i],tour[(i+1)]) for i in range(len(tour)-1)]
    nn_len = sum([G[u][v]['weight'] for u,v in tmp])
    return nn_len

def opt(instance):
    """
        Optimal TSP Tour length
        Inputs:
            instance: a TSP Instance codified in concorde style
    """
    return instance['tourlength']

def ri(instance):
    """
        Random Insertion TSP Heuristic O(n²)
        returns length of random insertion tour
        Inputs:
            instance: a TSP Instance codified in concorde style
    """
    G = get_nx_graph(instance['node_coordinates'])
    unvisited = list(G.nodes)

    start = max(G.edges(data='weight'), key = lambda tuple: tuple[2])

    unvisited.remove(start[0])
    unvisited.remove(start[1])

    tour = [int(start[0]), int(start[1]), int(start[0])]

    while len(unvisited) > 0:
        # find closest node to tour
        new_node = np.random.choice(unvisited)

        unvisited.remove(new_node)

        # see possible tours
        c_len = tour_len(tour, G)
        tours = [
            (c_len - G[tour[i]][tour[i+1]]['weight'])
            + G[tour[i]][new_node]['weight']
            + G[tour[i+1]][new_node]['weight']
            for i in range(len(tour)-1)
        ]
        tour.insert(np.argmin(tours)+1, new_node)
    len_tour = tour_len(tour, G)
    return len_tour

def tour_len(tour, G):
    """
        Tour length of a given tour
        Inputs:
            tour: a tour of indeces over a graph
            G: The nx.Graph with weighted edges
    """
    edge_list = [(tour[j], tour[j+1]) for j in range(1,len(tour)-1)]
    len_tour = sum([G[u][v]['weight'] for u,v in edge_list])
    return len_tour

def tour_insert(tour, i, node):
    """
        Inserting a node at a given index in a tour
        Inputs:
            tour: a list that represents a tour over a graph
            node: the node that has to be inserted
            i: the index where node should be inserted
    """
    tmp = tour.copy()
    tmp.insert(i,node)
    return tmp

def get_nx_graph(coordinates) -> nx.Graph():
    """
        Function enters coordinates into an nx.Graph() Datastructure and Computes
        the EUC_2D Distance between nodes and enteres them as edge weight to the graph
        Inputs:
            coordinates: Tuplelist containing 2D nodes
    """
    G = nx.complete_graph(n=len(coordinates))
    for u,v in G.edges:
        dist = np.linalg.norm(np.array(coordinates[u]) - np.array(coordinates[v]))
        G.edges[u,v]['weight']=dist
    return G
