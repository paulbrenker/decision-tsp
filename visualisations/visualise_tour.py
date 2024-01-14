import numpy as np
import matplotlib.pyplot as plt

def visualise_tour(coordinates: np.array, tour=None, connect_tour=True, title='TSP Tour'):
    """
        Method to visualize a given tsp tour including a heuristic
            coordinates: 2D np.array of graph coordinates
            tour: 1D np.array of tour
            connect_tour: boolean connect first and last node of the tour
            title: title of the visualization
    """
    plt.figure(figsize =(15, 7), dpi=100)
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