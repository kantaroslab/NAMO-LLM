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
    
 