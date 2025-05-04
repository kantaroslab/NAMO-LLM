import numpy as np
from shapely.geometry import Polygon

class Obs:
    def __init__(self, x, y, l, d, theta):
        self.x = x
        self.y = y
        self.l = l
        self.d = d
        self.theta = theta
        self.polygon = self.set_polygon()

    def set_polygon(self):
        h_l = self.l / 2
        h_d = self.d / 2
        vertices = [[h_l, h_d],[h_l, -h_d],[-h_l, -h_d],[-h_l, h_d]]
        rotation_matrix = [
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]
        ]
        rotated_vertices = np.dot(vertices, rotation_matrix)
        final_vertices = rotated_vertices + np.array([self.x, self.y])
        return Polygon(final_vertices)
    
    def real_c(self, r=0.2):
        """
        Return a new Obs shrunk by r in both length and depth. r=0.2 by default.
        """
        new_l = self.l - 2*r
        new_d = self.d - 2*r
        if new_l <= 0 or new_d <= 0:
            raise ValueError(f"Shrink amount r={r} too large; resulting dimensions must be > 0.")
        return Obs(self.x, self.y, new_l, new_d, self.theta)