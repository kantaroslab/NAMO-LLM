from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from obs import Obs
from matplotlib.patches import Polygon as MatPolygon
import imageio
import os


def is_collision_free(env, obs):
    if not env.contains(obs.polygon):
        return False
    return True

def sample_random_config(env, start_obj):
    """
    Sample a random configuration (x, y, theta) within the free space (env).
    """
    while True:
            minx, miny, maxx, maxy = env.bounds
            dx = round(random.uniform(minx, maxx),2)
            dy = round(random.uniform(miny, maxy),2)
            random_theta = random.uniform(0, math.pi)
            sample_obs = Obs(dx, dy, start_obj.l, start_obj.d, theta=random_theta)
            if env.contains(sample_obs.polygon):               
                return sample_obs

def config_distance(config1, config2):
    """
    Compute the distance between two configurations, considering position.
    """
    dx = config1[0] - config2[0]
    dy = config1[1] - config2[1]
    return math.hypot(dx, dy)

def nearest_vertex(tree, config):
    """
    Find the nearest vertex in the tree to a given configuration.
    """
    nearest = None
    min_dist = float("inf")
    nearest_idx = None
    for idx, (node_config, _, _) in enumerate(tree):
        dist = config_distance(node_config, config)
        if dist < min_dist:
            nearest = (node_config, idx)
            min_dist = dist
    return nearest[1], nearest[0]

def steer(from_obj, to_config, step_size):
    """
    Move the object towards the target configuration by a step size.
    """
    to_x, to_y, to_theta = to_config
    dx = to_x - from_obj.x
    dy = to_y - from_obj.y
    dtheta = to_theta - from_obj.theta
    # Normalize dtheta to be between -pi and pi
    dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
    distance = math.hypot(dx, dy)

    if distance <= 0.1:
        return from_obj  # Already at the point

    scale = min(step_size / distance, 1.0)
    new_x = from_obj.x + dx * scale
    new_y = from_obj.y + dy * scale
    new_theta = from_obj.theta + dtheta * scale
    new_theta = new_theta % (2 * math.pi)
    # print(f"from_theta={from_obj.theta}, to_theta={to_theta}, new_theta={new_theta}")
    new_obj = Obs(new_x, new_y, from_obj.l, from_obj.d, new_theta)
    return new_obj

def is_goal_reached(current_obj, goal_obj, tolerance=3):
    """
    Check if the current object is close enough to the goal object.
    """
    current_config = (current_obj.x, current_obj.y, current_obj.theta)
    goal_config = (goal_obj.x, goal_obj.y, goal_obj.theta)
    dist = config_distance(current_config, goal_config)
    return dist <= tolerance


def RRT(N, obstacles, start_obj, goal_obj, max_iter=5000, step_size=2.0):
    """
    RRT algorithm to find a path for moving an object from start to goal within a free space.
    Now also visualizes the best path even if the goal is not reached.
    """
    env = unary_union([N, start_obj.polygon])
    # Tree stores tuples: ( (x, y, theta), parent_index, Obj )
    tree = [((start_obj.x, start_obj.y, start_obj.theta), None, start_obj)]

    for i in range(max_iter):
        sample_obs = sample_random_config(env, start_obj)
        rand_config = (sample_obs.x, sample_obs.y, sample_obs.theta)
        nearest_idx, _ = nearest_vertex(tree, rand_config)
        nearest_obj = tree[nearest_idx][2]
        new_obj = steer(nearest_obj, rand_config, step_size)

        if is_collision_free(env, new_obj):
            tree.append(((new_obj.x, new_obj.y, new_obj.theta), nearest_idx, new_obj))
            if is_goal_reached(new_obj, goal_obj):
                # Successful: build and visualize the path
                path = []
                current_index = len(tree) - 1
                while current_index is not None:
                    _, parent_index, node_obj = tree[current_index]
                    path.append(node_obj)
                    current_index = parent_index
                path.reverse()
                # visualize(env, obstacles, path)
                return True

    # If goal was not reached, visualize the best-so-far path
    # Find the node closest to the goal
    dists = [
        config_distance((node_obj.x, node_obj.y, node_obj.theta),
                        (goal_obj.x, goal_obj.y, goal_obj.theta))
        for (_, _, node_obj) in tree
    ]
    best_idx = int(min(range(len(dists)), key=lambda idx: dists[idx]))

    # Build and visualize the best-so-far path
    path = []
    current_index = best_idx
    while current_index is not None:
        _, parent_index, node_obj = tree[current_index]
        path.append(node_obj)
        current_index = parent_index
    path.reverse()
    # visualize(env, obstacles, path)
    return False

def plot_polygon_rrt(polygon, color='blue', alpha=0.5):
    # Plot the exterior boundary using the provided color and alpha
    x, y = polygon.exterior.xy
    plt.fill(x, y, color=color, alpha=alpha)
    # Plot the interior rings (holes) with white color to represent holes
    for interior in polygon.interiors:
        x, y = interior.xy
        plt.fill(x, y, color='white', alpha=1.0)

def plot_rrt(area, start_obj, sample_obs, new_obj, i):
    plot_polygon_rrt(area, color='grey', alpha=0.3)
    plot_polygon_rrt(start_obj.polygon, color='red', alpha=0.2)
    plot_polygon_rrt(sample_obs.polygon, color='red', alpha=1)
    plot_polygon_rrt(new_obj.polygon, color='green', alpha=0.5)
    plt.xlim(-1, 51)
    plt.ylim(-1, 51)
    plt.gca().set_aspect('equal', adjustable='box')
    # image = f"rrt/plot_{i}.png"
    # plt.savefig(image)
    plt.show()
    # plt.close()

def visualize(area, obstacles, path, filename='rrt_animation.gif'):
    plot_polygon_rrt(area, color='grey', alpha=0.3)
    for obs in obstacles:
        plot_polygon_rrt(obs.polygon, color='red', alpha=0.2)
    for i in range(len(path)-1):
        plot_polygon_rrt(path[i].polygon, color='green', alpha=0.5)
        plt.xlim(-1, 51)
        plt.ylim(-1, 51)
        plt.gca().set_aspect('equal', adjustable='box')
        image = f"rrt/plot_{i}.png"
        plt.savefig(image)