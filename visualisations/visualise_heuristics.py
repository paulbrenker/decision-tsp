import matplotlib.pyplot as plt
from visualisations.visualise_heuristics_distribution import heuristic_names

def visualise_heuristics(heuristics, n_range):
    """
        plot the behavior of the heuristics in comparison to the optimal tour
        x axis is |V| = n
    """
    plt.figure(figsize=(15, 7), dpi=100)
    
    
    for (heuristic, data) in heuristics.items():
        if heuristic == 'opt':
            plt.scatter(n_range, data, marker='.', alpha=0.6, label=heuristic_names[heuristic])
        else:
            plt.scatter(n_range, data, marker='x', alpha=0.4, label=heuristic_names[heuristic])
    
    plt.title('Heuristics')
    plt.xlabel('|V|')
    plt.ylabel('TSP tour')
    plt.legend()
    plt.grid(True)
    plt.show()