
class PartitionBlock:
    """
    Represents a sub-block in a 2D grid.

    Attributes:
    - vertices (np.ndarray): The vertices of the sub-block.
    - amplitude (int): The amplitude of the sub-block.
    - samples_inside (list): List of samples inside the sub-block.
    - output_values (list): List of output values.
    - index (list): List of index of the original point X

    Methods:
    - add_point(point): Add a point to the sub-block.
    """

    def __init__(self, amplitude=1, ista_layer=None):
        self.amplitude = amplitude
        self.ista_layer = ista_layer
        self._X = []
        self._index = []
        self.predicted_output = []
        self.target = []

    def new_point(self, point_x, point_y):
        self._X.append(point_x)
        self.target.append(point_y)

    def get_X(self):
        return self._X

    def set_X(self, new_X):
        self._X = new_X
