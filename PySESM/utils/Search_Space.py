import numpy as np

class Search_Space:
    def __init__(self, a, b, T, n):
        self.a = a
        self.b = b
        self.T = T
        self.n = n
        self.corners = None

    def generate_partitions(self):
        partitions = [np.linspace(self.a, self.b, self.T + 1) for _ in range(self.n)]
        self.corners = np.array(np.meshgrid(*partitions)).T.reshape(-1, self.n)
        return self.corners

    def assign_to_subblock(self, x):
        if self.corners is None:
            raise ValueError("Generate partitions first.")

        n_dimensions = x.shape[0]
        sub_intervals = []
        sorted_corners = np.sort(self.corners, axis=0)

        for i in range(n_dimensions):
            index = np.digitize(x[i], sorted_corners[:, i]) - 1
            sub_interval = (sorted_corners[index, i], sorted_corners[index + 1, i])
            sub_intervals.append(sub_interval)

        return sub_intervals

    def get_corners(self):
        if self.corners is None:
            raise ValueError("Generate partitions first.")
        return self.corners

    def update_parameters(self, a=None, b=None, T=None, n=None):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if T is not None:
            self.T = T
        if n is not None:
            self.n = n
        self.corners = None 
