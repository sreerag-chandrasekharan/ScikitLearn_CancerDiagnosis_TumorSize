import numpy as np
import copy
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./style.mplstyle')

def load_data():
    data = np.loadtxt("Small_data.txt", delimiter=',', skiprows=1)
    X = data[:,0]
    y = data[:,1]
    return X, y

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best', expand_factor=0.1):
    """
    Plots feature values (X) against corresponding class labels (y) with expanded x-axis.
    Works with reshaped X (n_samples, 1) and reshaped y (n_samples, 1).
    """
    
    # Ensure X and y are both 2D arrays
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Flatten y for easier indexing
    y = y.flatten()
    
    # Find indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    
    # Plot the data
    ax.scatter(X[pos, 0], y[pos], marker='x', s=s, c='red', label=pos_label)  # For y=1 (positive)
    ax.scatter(X[neg, 0], y[neg], marker='o', s=s, facecolors='none', c='blue', lw=3, label=neg_label)  # For y=0 (negative)
    
    # Set axis labels
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Class Label")
    
    # Expand x-axis limits for better visibility
    x_min, x_max = X.min(), X.max()
    x_range = x_max - x_min
    ax.set_xlim([x_min - expand_factor * x_range, x_max + expand_factor * x_range])
    
    # Set y-axis limits for clarity
    ax.set_ylim([-0.5, 1.5])  # Keep y-axis between -0.5 and 1.5 since class labels are 0 and 1
    
    # Add legend
    ax.legend(loc=loc)
    
    # Hide toolbar, header, and footer for a cleaner look
    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def plot_with_predictions(X, y, y_pred, ax, pos_label="True y=1", neg_label="True y=0", 
                          pred_pos_label="Pred y=1", pred_neg_label="Pred y=0", s=80, loc='best', expand_factor=0.1):
    """
    Plots feature values (X) against true labels (y) and predicted labels (y_pred) with expanded x-axis.
    Works with reshaped X (n_samples, 1), y (n_samples, 1), and y_pred (n_samples, 1).
    """
    
    # Ensure X, y, and y_pred are all 2D arrays
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Flatten y and y_pred for easier indexing
    y = y.flatten()
    y_pred = y_pred.flatten()
    
    # Find indices of Positive and Negative Examples for both true and predicted labels
    pos = y == 1
    neg = y == 0
    pred_pos = y_pred == 1
    pred_neg = y_pred == 0
    
    # Plot the true data
    ax.scatter(X[pos, 0], y[pos], marker='x', s=s, c='red', label=pos_label)  # For true y=1 (positive)
    ax.scatter(X[neg, 0], y[neg], marker='o', s=s, facecolors='none', c='blue', lw=3, label=neg_label)  # For true y=0 (negative)
    
    # Overlay the predicted data
    ax.scatter(X[pred_pos, 0], y_pred[pred_pos], marker='x', s=s, c='green', label=pred_pos_label)  # For predicted y=1
    ax.scatter(X[pred_neg, 0], y_pred[pred_neg], marker='o', s=s, facecolors='none', c='orange', lw=3, label=pred_neg_label)  # For predicted y=0
    
    # Set axis labels
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Class Label (True & Predicted)")
    
    # Expand x-axis limits for better visibility
    x_min, x_max = X.min(), X.max()
    x_range = x_max - x_min
    ax.set_xlim([x_min - expand_factor * x_range, x_max + expand_factor * x_range])
    
    # Set y-axis limits for clarity
    ax.set_ylim([-0.5, 1.5])  # Keep y-axis between -0.5 and 1.5 since class labels are 0 and 1
    
    # Add legend
    ax.legend(loc=loc)
    
    # Hide toolbar, header, and footer for a cleaner look
    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False
