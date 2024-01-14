import matplotlib.pyplot as plt

# Abbreviation dict for heuristics
heuristic_names = {
    'ni': 'Nearest Insertion',
    'opt': 'Optimal Tourlength',
    'nn': 'Nearest Neighbor',
    'ri': 'Random Insertion',
    'mstheu': 'Minimum Spanning Tree Heuristic',
    'mst': 'Minimum Spanning Tree',
    'lplb': 'LP Relaxation',
    'greedy': 'Greedy Heuristic',
    'fi': 'Farthest Insertion',
    'ci': 'Cheapest Insertion',
    'christo': 'Christofides Heuristik',
    'assrel': 'Assignment Relaxation',
    '1tree': '1 Tree',
    'onetree': '1 Tree'
}
custom_cmap = {
    0: 'blue',
    1: 'orange',
    2: 'green',
    3: 'red',
    4: 'purple',
    5: 'brown',
    6: 'pink',
    7: 'gray',
    8: 'olive',
    9: 'cyan',
    10: 'black',
    11: 'lawngreen'
}

def visualise_heuristics_distribution(plot_data: dict):
    """Print A matrix of histograms that decribe given heuristics"""

    # Get the number of rows and columns for the matrix
    num_rows = 3
    num_cols = 4

    # Create a figure and subplots
    axs = plt.subplots(num_rows, num_cols, figsize=(24, 12))[1]

    # Flatten the axs array to iterate through it easily
    axs = axs.flatten()

    # Iterate over the dictionary and plot histograms
    for i, (array_name, array_data) in enumerate(plot_data.items()):
        axs[i].hist(array_data, bins=20, color = custom_cmap[i], alpha=0.4)
        axs[i].set_title(heuristic_names[array_name])
        axs[i].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()


    # Show the plot
    plt.show()