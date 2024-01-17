import matplotlib.pyplot as plt
import numpy as np

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



def visualise_regression(predictions, actual, results, errors, train_set_size, title):
    """
        visualise set of learning curve, distribution and error curve
    """
    figure, axis = plt.subplots(1, 3, figsize=(30,7))
    figure.suptitle(title, fontsize=16)
    _plot_distribution(axis[0], predictions, actual)
    _plot_learning_curve(axis[1], results, train_set_size)
    _plot_error_curve(axis[2], errors, train_set_size)
    plt.show()

def _plot_distribution(ax, predictions: list, actual: list):
    """
        Scatterplot with length of optimal tsp tour on x axis and
        estimated result on the y axis using a given regression model
    """
    ax.scatter(predictions, actual, marker='.', c='black', alpha=0.6)
    ax.set_title('Distribution of optimal Tour length vs. predicted tour length')
    ax.set_xlabel('predicted length of the tsp tour with regression model')
    ax.set_ylabel('optimal length of tsp tour')
    ax.grid(True)

def _plot_learning_curve(ax, results: dict, train_set_size):
    """
        Spagetti plot showing accuracy of regression with different
        acceptance limits. Training set size on the x axis and percentage
        of successful estimations on the y axis.
    """
    for key, value in results.items():
        ax.plot(train_set_size*2000, value, label=str(key*100)+'%')

    ax.set_title('Learning curve over training set size')
    ax.set_xlabel('Size of training set')
    ax.set_ylabel('Percentage of results within limit')
    ax.grid(True)
    ax.legend(loc='lower right')

def _plot_error_curve(ax, errors: dict, train_set_size):
    """
        PLot of development of mean squared error and mean absolute error
        over size of training set.
    """    
    for key, value in errors.items():
        ax.plot(train_set_size*2000, value, label=key.replace('_',' '))

    ax.set_title('Error curve over training set size')
    ax.set_xlabel('Size of training set')
    ax.set_ylabel('Errors')
    ax.grid(True)
    ax.legend(loc='upper right')



def visualise_tour(coordinates: np.array, tour=None, connect_tour=True, title='TSP Tour'):
    """
        Method to visualize a given tsp tour including a heuristic
            coordinates: 2D np.array of graph coordinates
            tour: 1D np.array of tour
            connect_tour: boolean connect first and last node of the tour
            title: title of the visualization
    """
    plt.figure(figsize =(15, 7), dpi=100)
    ax = plt.axes()
    plt.scatter(*coordinates.T, alpha=0.8, color='orange')
    if tour is not None:
        seq = []
        if connect_tour:
            seq = np.array([ [coordinates[step], coordinates[tour[(i+1)%len(tour)]]] for i, step in enumerate(tour)])
        else:
            seq = np.array([ [coordinates[step], coordinates[tour[(i+1)%len(tour)]]] for i, step in enumerate(tour[:-1])]) 
        plt.plot(*seq.T, color='black')


    plt.title(title) 
    plt.plot()
    plt.show()