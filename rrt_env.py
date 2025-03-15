from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely.affinity import translate
import matplotlib.pyplot as plt
import random
from openai import OpenAI
import base64
import requests
import api
import re
import time
import numpy as np
import math
import pandas as pd
from openpyxl import Workbook
import gemini
import rrt
from obs import Obs
import ast


class Rob:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def to_polygon(self):
        return Point(self.x, self.y).buffer(self.r)

def plot_rrt_obj(area, from_obj, to_obj):
    plot_polygon(area, color='grey', alpha=0.3)
    plot_polygon(from_obj.polygon, color='red', alpha=0.6)
    plot_polygon(to_obj.polygon, color='green', alpha=0.5)
    plt.xlim(-1, 41)
    plt.ylim(-1, 41)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_polygon(polygon, color='blue', alpha=0.5):
    # Plot the exterior boundary using the provided color and alpha
    x, y = polygon.exterior.xy
    plt.fill(x, y, color=color, alpha=alpha)
    # Plot the interior rings (holes) with white color to represent holes
    for interior in polygon.interiors:
        x, y = interior.xy
        plt.fill(x, y, color='white', alpha=1.0)

def plot(area, All_Obs, x_goal, i, robot=None):
    plot_polygon(area, color='grey', alpha=0.3)
    if robot is not None:
        plot_polygon(robot.to_polygon(), color='black', alpha=0.5)
    for id, obs in enumerate(All_Obs, start=1):
        plot_polygon(obs.polygon, color='red', alpha=0.6)
        plt.text(obs.x, obs.y, str(id), fontsize=6, ha='center', va='center')
    # plot_polygon(N, color='green', alpha=0.2)
    plot_polygon(x_goal, color='green', alpha=0.5)
    plt.xlim(-1, 21)
    plt.ylim(-1, 21)
    plt.gca().set_aspect('equal', adjustable='box')
    image = f"img/plot_{i}.png"
    plt.savefig(image)
    # plt.show()
    plt.close()

def plot_N(N, i):
    plot_polygon(N, color='green', alpha=0.2)
    plt.xlim(-1, 21)
    plt.ylim(-1, 21)
    plt.gca().set_aspect('equal', adjustable='box')
    image = f"img/plot_N_{i}.png"
    plt.savefig(image)
    # plt.show()
    plt.close()


def free_space(area, obstacles, robot=None, obs_c_i=None, i=None):
    f_space = area.difference(unary_union([obs.polygon for obs in obstacles]))
    if isinstance(f_space, Polygon):
        return f_space
    elif isinstance(f_space, MultiPolygon):
        if robot is not None:
            robot_position = Point(robot.x, robot.y)
            for polygon in f_space.geoms:
                if polygon.contains(robot_position):
                    return polygon
        elif obs_c_i is not None:
            N_list = []
            c_i_boundary = obs_c_i.polygon.boundary
            for polygon in f_space.geoms:
                # print("polygon: ", polygon)
                if polygon.touches(obs_c_i.polygon):
                    inter = polygon.boundary.intersection(c_i_boundary)
                    if not inter.is_empty and inter.geom_type not in ['Point', 'MultiPoint']:
                        N_list.append(polygon)
            if N_list:
                return random.choice(N_list)
            
def find_adj(N, obstacles):
    adj_obs = []
    if not N.is_valid:
        N = N.buffer(0)  # Clean the geometry
    id_list = []
    for id, obs in enumerate(obstacles, start=1):
        if not obs.polygon.is_valid:
            obs.polygon = obs.polygon.buffer(0)
        if N.touches(obs.polygon):
            # Compute the intersection of boundaries
            inter = N.boundary.intersection(obs.polygon.boundary)
            if inter.is_empty:
                continue
            # Check if the intersection is just a point or multiple points
            if inter.geom_type in ['Point', 'MultiPoint']:
                continue
            else:
                id_list.append(id)
                adj_obs.append(obs.set_polygon())

    # print(f"adj_obs: {id_list}")
    adj_obs = tuple(adj_obs)    
    id_list = tuple(id_list)
    return adj_obs, id_list

def poly_info(all_obs):
    obs_info = {}
    for id, obs in enumerate(all_obs, start=1):
        key = f"obs{id}"
        value = obs.set_polygon()
        obs_info[key] = value
    return obs_info
    

class RT:
    def __init__(self, x_init, x_goal):
        self.x_init = x_init
        self.x_goal = x_goal
        self.obs_init = x_init[0]
        self.N_init = x_init[1]
        self.value_init = x_init[2]
        self.adj_obs_init = x_init[3]
        self.id_list_init = x_init[4]
    
    def buildTree(self, time_counter):
        T = {tuple(self.x_init): None}
        path_found = False
        final_state = None
        i = 1 #iteration
        while not path_found:
            print(f"Iteration: {i}")
            p1 = random.random()
            values = [node[2] for node in T.keys()]
            max_value = max(values)
            highest_value_nodes = [node for node in T.keys() if node[2] == max_value]
            # print(f"prob1: {prob1}")
            if p1 < prob1: # Pick highest value node
                x = random.choice(highest_value_nodes)
            else: # Pick a random node
                x = random.choice(list(T.keys()))
            obs = x[0]
            N = x[1]
            value = x[2]
            adj_obs = x[3]
            id_list = x[4]
            j = x[5] # picture number
            x_new, time_counter = self.expand(N, obs, self.x_goal, i, value, time_counter, adj_obs, id_list, j)
            if x_new is None:
                continue
            T[x_new] = x
            # print(f"The length of T: {len(T)}")            
            i += 1
            if x_new[1].contains(self.x_goal):  # Check if the goal is reached
                path_found = True
                final_state = x_new
                # print("value: ", final_state[2])
                print("Path found! Terminate.")
                break
        path = []
        current_state = final_state
        while current_state is not None:
            path.append(list(current_state)[-1])  # Convert the tuple back to a list
            current_state = T[current_state]
        path.reverse()
        path_length = len(path)
        print(f"The path is {path}")
        return path_length, i, time_counter
    
    
    def expand(self, N, all_obs, x_goal, i, value, time_counter, adj_obs, id_list, j):
        p2 = random.random()
        M_id = 0
        obs_info = poly_info(all_obs)
        # print(f"prob2: {prob2}")
        if p2 < prob2: # run LLM
            while True:
                # select obs
                api_obs = api.api_Mi(area, N, obs_info, adj_obs, id_list, x_goal, j)
                # api_obs = gemini.api_Mi(area, N, obs_info, adj_obs, id_list, x_goal, j)
                print(f"api_obs: {api_obs}")
                if re.match(r"^\[.*\]$", api_obs.strip()):
                    try:
                        api_obs = ast.literal_eval(api_obs)
                    except:
                        M_id = random.choice(list(id_list))
                        break
                    api_obs = random.choice(api_obs)
                    print(f"after random choice: {api_obs}")
                    # if api_obs is None:
                    #     time.sleep(30)
                    #     time_counter += 30
                    match = re.match(r'obs(\d+)', api_obs)
                    if match:
                        try:
                            M_id = int(match.group(1))
                            if M_id in id_list:
                                # print("matched.")
                                # print(f"the obs we picked is {M_id}")
                                value = value + 1
                                break
                            else:
                                M_id = random.choice(list(id_list))
                                break
                        except:
                            M_id = random.choice(list(id_list))
                            break
                    else:
                        M_id = random.choice(list(id_list))
                        break
                else:
                    M_id = random.choice(list(id_list))
                    break

        else: # run random sampling
            M_id = random.choice(list(id_list))
        # print(f"In the picture {j}, The obs we picked is {M_id}")
        M_i = all_obs[M_id-1]
        c_mi = N.union(M_i.polygon)
        obs_c_i = self.pick_c_i_prime(c_mi, M_i)
        if rrt.RRT(N, all_obs, M_i, obs_c_i, max_iter=1000, step_size=0.5):
            all_obs_list = list(all_obs)
            all_obs_list[M_id-1] = obs_c_i
            N_new = free_space(area, all_obs_list, None, obs_c_i, i)
            updated_all_obs = tuple(all_obs_list)
            adj_obs_new, id_list_new = find_adj(N_new, updated_all_obs)
            if not adj_obs_new:
                plot_N(N_new, i)
                plot(area, updated_all_obs, x_goal, i)
                print("ATTENTION: No adjacent obstacles found.")
            x_new = [updated_all_obs, N_new, value, adj_obs_new, id_list_new, i]
            x_new = tuple(x_new)
            plot(area, updated_all_obs, x_goal, i)
            return x_new, time_counter
        else:
            # print(f"from {M_i.polygon} to {obs_c_i.polygon} failed.")
            # plot_rrt_obj(area, M_i, obs_c_i)
            # print("ATTENTION: RRT failed.")
            return None, time_counter
    

    def pick_c_i_prime(self, free_space, M_i):
        while True:
            minx, miny, maxx, maxy = free_space.bounds
            obs_minx, obs_miny, obs_maxx, obs_maxy = M_i.polygon.bounds
            dx = round(random.uniform(minx - obs_minx, maxx - obs_maxx),2)
            dy = round(random.uniform(miny - obs_miny, maxy - obs_maxy),2)
            random_theta = random.uniform(0, math.pi)
            c_i_prime = translate(M_i.polygon, xoff=dx, yoff=dy)
            obs_c_i = Obs(x= round(0.5 * (c_i_prime.bounds[0] + c_i_prime.bounds[2]),2),
                                y=round(0.5 * (c_i_prime.bounds[1] + c_i_prime.bounds[3]),2),
                                l=M_i.l, d=M_i.d, theta=random_theta)
            if free_space.contains(obs_c_i.polygon):               
                return obs_c_i



if __name__ == '__main__':
    probset1 = [0.8]
    probset2 = [0.8]

    for prob1 in probset1:
        for prob2 in probset2:
            j = 0
            time_list = []
            iteration_list = []
            path_list = []
            df = pd.DataFrame(columns=['Iteration', 'Time', 'Path'])
            for j in range(100):
                print(f"######{j}######")
                area = Polygon([(0, 0), (10, 0), (10, 2.5), (11, 2.5), (11, 1), (20, 1), (20, 20), (0, 20)])
                x_goal = Polygon([(11, 1), (20, 1), (20, 2), (11, 2)])
                rob = Rob(1, 1, 0.2)

                obs1  = Obs(15.5, 3, 9, 2, 0)
                obs2  = Obs(18, 5, 2, 2, 0)
                obs3  = Obs(16, 6, 2, 2, 0)
                obs4  = Obs(1, 4, 2, 2, 0)
                obs5  = Obs(7, 3, 1, 2, 0)
                obs6  = Obs(12, 9, 3, 2, 0)
                obs7 = Obs(10, 5, 2, 2, 0)
                obs8 = Obs(12, 5, 2, 2, 0)
                obs9 = Obs(3, 7, 1, 1, 0)
                obs10 = Obs(1, 19, 2, 2, 0)
                

                All_Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10]

                plot(area, All_Obs, x_goal, 0, rob)
                N = free_space(area, All_Obs, rob)
                obs_info = poly_info(All_Obs)
            
                All_Obs = tuple(All_Obs)
                value_init = 0
                adj_obs, id_list = find_adj(N, All_Obs)
                x_init = [All_Obs, N, value_init, adj_obs, id_list, 0]

                time_counter = 0
                start_time = time.time()
                T = RT(x_init, x_goal)
                path_length, iteration_number, time_counter = T.buildTree(time_counter)
                # print(f"Time counter: {time_counter}")
                end_time = time.time()
                elapsed_time = end_time - start_time - time_counter  
                # print(f"Elapsed time: {elapsed_time} seconds") 
                time_list.append(elapsed_time) 
                iteration_list.append(iteration_number)
                path_list.append(path_length)
                new_row = pd.DataFrame({
                'Iteration': [iteration_list[-1]],
                'Time': [time_list[-1]],
                'Path': [path_list[-1]]
                })
                # df = df.append(new_row, ignore_index=True)
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_excel(f'output{prob1}and{prob2}.xlsx', index=False)
                # time.sleep(60)