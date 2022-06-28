import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


# Visualization
COLORS = ['cyan', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'navy']

class Node():
    def __init__(self, n_iters, node_prev=None, dir_prev=None):
        # Initialize the first node
        if not isinstance(node_prev, ndarray):
            node_prev = np.array([0.0, 0.0])
        if not isinstance(dir_prev, ndarray):
            dir_prev = np.array([0.2, 0.8])

        # Scaler of the length
        self.exp = 0.9

        # Depth of the fractal
        self.n_iters = n_iters

        # Recursively generate children of the node, until max depth is reached
        self.node_prev = node_prev
        self.dir_prev = dir_prev
        self.children = None
        if self.n_iters > 0:
            self.children = self.generate_children(2) # 2 - number of children
            self.plot_line()


    def generate_children(self, n_childrens):
        """Find the location of the next n_children points in the fractal and make them into new nodes."""
        y_dir = self.dir_prev[0]
        x_dir = self.dir_prev[1]
        children = list()
        for i in range(n_childrens):
            # Find coordinates of the next point
            angle_i = float(i + 1) * np.pi / float(n_childrens + 1) # Symmetrical angle 
            y_dir_i = y_dir * self.exp * np.sin(angle_i)
            x_dir_i = x_dir * self.exp * np.cos(angle_i)
            # Generate next node
            dir_i = np.array([y_dir_i, x_dir_i])
            node_i = self.node_prev + dir_i
            children.append(Node(self.n_iters - 1, node_i, dir_i))
        return children

    def plot_line(self):
        """Visualization of the fractal."""
        y_start = self.node_prev[0]
        x_start = self.node_prev[1]

        lw = self.n_iters / 25
        alp = self.n_iters / 12
        for i in range(len(self.children)):            
            y_end = self.children[i].node_prev[0]
            x_end = self.children[i].node_prev[1]
            plt.plot([x_start, x_end], [y_start, y_end], c=COLORS[i], linewidth=lw, alpha=alp)


if __name__ == "__main__":
    depth = 12
    a = Node(depth)
    plt.box(False)
    plt.yticks([])
    plt.xticks([])
    plt.show()
