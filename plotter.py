import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_points(points: NDArray[np.float64], label: str = "Points") -> None:

    # Plot the normalized points
    plt.figure(figsize=(10, 6))
    plt.plot(points, 'o', markersize=1, label=label)
    plt.legend()
    plt.title('Normalized Points')
    plt.xlabel('Index')
    plt.ylabel('Normalized Values')
    plt.ylim(0, 1)
    plt.show()
        
def plot_two_points(points_1: NDArray[np.float64], points_2: NDArray[np.float64]) -> None:
    # Plot the normalized points
    plt.figure(figsize=(10, 6))
    plt.plot(points_1, 'o', markersize=1, label="A")
    plt.plot(points_2, 'o', markersize=1, color='red', label='B')
    plt.legend()
    plt.title('Normalized Points')
    plt.xlabel('Index')
    plt.ylabel('Normalized Values')
    plt.ylim(0, 1)
    plt.show()
    
def plot_three_points(points_1: NDArray[np.float64], points_2: NDArray[np.float64], points_3: NDArray[np.float64]) -> None:
    # Plot the normalized points
    plt.figure(figsize=(10, 6))
    plt.plot(points_1, 'o', markersize=1, label="A")
    plt.plot(points_2, 'o', markersize=1, color='red', label='B')
    plt.plot(points_3, 'o', markersize=1, color='green', label='C')
    plt.legend()
    plt.title('Normalized Points')
    plt.xlabel('Index')
    plt.ylabel('Normalized Values')
    plt.ylim(0, 1)
    plt.show()
        
def analyze_distribution(points: NDArray[np.float64], label: str = "Points") -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax[0].hist(points, bins=30, alpha=0.75, color='blue', edgecolor='black')
    ax[0].set_title('Histogram of Points')
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Frequency')
    
    # Box plot
    ax[1].boxplot(points, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax[1].set_title('Box Plot of Points')
    ax[1].set_xlabel('Value')
    
    plt.suptitle('Distribution Analysis of Points')
    plt.show()
