import matplotlib.pyplot as plt

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