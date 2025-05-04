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

    # Case 1&3&5 with 50 obstacles
    plt.xlim(-1, 51)
    plt.ylim(-1, 51)

    # # Case 2&4 with 100 obstacles
    # plt.xlim(-1, 66)
    # plt.ylim(-1, 66)

    # # Case 6 with 10 obstacles
    # plt.xlim(-1, 21)
    # plt.ylim(-1, 21)

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


def free_space(area, obstacles, robot=None, c_i_prime=None, i=None):
    f_space = area.difference(unary_union([obs.polygon for obs in obstacles]))
    if isinstance(f_space, Polygon):
        return f_space
    elif isinstance(f_space, MultiPolygon):
        if robot is not None:
            robot_position = Point(robot.x, robot.y)
            for polygon in f_space.geoms:
                if polygon.contains(robot_position):
                    return polygon
        elif c_i_prime is not None:
            N_list = []
            c_i_boundary = c_i_prime.polygon.boundary
            for polygon in f_space.geoms:
                # print("polygon: ", polygon)
                if polygon.touches(c_i_prime.polygon):
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

def get_real_obs(all_obs):
    real_obs = []
    for obs in all_obs:
        real_obs.append(obs.real_c())
    return real_obs
    

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
        if p2 < prob2: # run LLM
            while True:
                # select obs

                api_obs = api.api_Mi(area, N, obs_info, adj_obs, id_list, x_goal, j) # Call the ChatGPT API
                # api_obs = gemini.api_Mi(area, N, obs_info, adj_obs, id_list, x_goal, j) # Call the Gemini API
                # print(f"api_obs: {api_obs}")

                # If the API response is in the format of a list (K>1 in our algorithm), randomly select one element from the list
                if re.match(r"^\[.*\]$", api_obs.strip()):
                    try:
                        api_obs = ast.literal_eval(api_obs)
                    except:
                        M_id = random.choice(list(id_list))
                        break
                    api_obs = random.choice(api_obs)

                # The time counter is used to adapt to the API's rate limit. ChatGPT API has different rate limits for different usage tiers, see https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-one.
                # If you are a new ChatGPT API user, you may have some issues when running the code due to the rate limit. In this case, we need time_counter to calculate the waiting time.
                # When you are on tier4 or tier5, you can remove the time_counter and the time.sleep() function.
                # If the API fails due to the rate limit, wait for 30 seconds and try again
                if api_obs is None:    
                    time.sleep(30)
                    time_counter += 30
                else:
                    match = re.match(r'obs(\d+)', api_obs)
                    if match:
                        try:
                            M_id = int(match.group(1))
                            if M_id in id_list:
                                # print("matched.")
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
        else: # run Random Sampling
            M_id = random.choice(list(id_list))
        # print(f"The obs we picked is {M_id}")
        M_i = all_obs[M_id-1]
        c_mi = N.union(M_i.polygon)
        c_i_prime = self.pick_c_i_prime(c_mi, M_i) # Pick a random configuration of M_i
        real_obs = get_real_obs(all_obs)
        real_M_i = M_i.real_c()
        real_c_i_prime = c_i_prime.real_c()
        if rrt.RRT(N, real_obs, real_M_i, real_c_i_prime, max_iter=5000, step_size=2):
            all_obs_list = list(all_obs)
            all_obs_list[M_id-1] = c_i_prime
            N_new = free_space(area, all_obs_list, None, c_i_prime, i) # Compute the new free space
            updated_all_obs = tuple(all_obs_list)
            adj_obs_new, id_list_new = find_adj(N_new, updated_all_obs) # Find the manipulable obstacles
            if not adj_obs_new:
                plot_N(N_new, i)
                plot(area, updated_all_obs, x_goal, i)
                print("ATTENTION: No adjacent obstacles found.")
            x_new = [updated_all_obs, N_new, value, adj_obs_new, id_list_new, i]
            x_new = tuple(x_new)
            plot(area, updated_all_obs, x_goal, i)
            return x_new, time_counter
        else:
            # print(f"from {M_i.polygon} to {c_i_prime.polygon} failed.")
            # plot_rrt_obj(area, M_i, c_i_prime)
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
            c_i_prime = Obs(x= round(0.5 * (c_i_prime.bounds[0] + c_i_prime.bounds[2]),2),
                                y=round(0.5 * (c_i_prime.bounds[1] + c_i_prime.bounds[3]),2),
                                l=M_i.l, d=M_i.d, theta=random_theta)
            if free_space.contains(c_i_prime.polygon):               
                return c_i_prime



if __name__ == '__main__':
    probset1 = [0.2, 0.8]
    probset2 = [0.2, 0.8]

    for prob1 in probset1:
        for prob2 in probset2:
            j = 0
            time_list = []
            iteration_list = []
            path_list = []
            df = pd.DataFrame(columns=['Iteration', 'Time', 'Path'])
            for j in range(100):
                print(f"######{j}######")

                # Case 1 (with 50 obs) & Case 2 (with 100 obs)
                area = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])  # Case 1
                # area = Polygon([(0, 0), (65, 0), (65, 65), (0, 65)])  # Case 2
                x_goal = Polygon([(16, 12.5), (17, 12.5), (17, 13), (16, 13)])
                rob = Rob(40, 45, 0.2)

                obs1   = Obs( 9.0,  9.0, 6, 2, 0)
                obs2   = Obs(14.5, 15.5, 1, 5, 0)
                obs3   = Obs(13.0,  6.5, 2, 3, 0)
                obs4   = Obs(13.0,  9.0, 2, 2, 0)
                obs5   = Obs(13.0, 11.0, 2, 2, 0)
                obs6   = Obs(13.5, 12.5, 3, 1, 0)
                obs7   = Obs(17.5, 17.0, 3, 2, 0)
                obs8   = Obs(19.5, 14.0, 1, 4, 0)
                obs9   = Obs(16.0, 11.0, 2, 2, 0)
                obs10  = Obs(18.5, 11.0, 3, 2, 0)
                obs11  = Obs(15.5,  9.0, 3, 2, 0)
                obs12  = Obs(15.5, 14.5, 1, 3, 0)
                obs13  = Obs(15.5,  4.5, 3, 1, 0)
                obs14  = Obs(18.5,  4.5, 3, 1, 0)
                obs15  = Obs(29.0, 34.0, 2, 2, 0)

                obs16  = Obs(32.0, 31.5, 4, 3, 0)
                obs17  = Obs(32.0, 34.0, 4, 2, 0)
                obs18  = Obs(29.5, 36.0, 1, 2, 0)
                obs19  = Obs(32.5, 38.0, 5, 2, 0)
                obs20  = Obs(37.0, 43.0, 2, 4, 0)
                obs21  = Obs(39.0, 39.0, 2, 4, 0)
                obs22  = Obs(36.0, 36.0, 2, 2, 0)
                obs23  = Obs(38.5, 36.0, 3, 2, 0)
                obs24  = Obs(35.5, 34.0, 3, 2, 0)
                obs25  = Obs(35.5, 40.0, 1, 2, 0)
                obs26  = Obs(35.5, 29.5, 3, 1, 0)
                obs27  = Obs( 8.0, 34.0, 4, 2, 0)
                obs28  = Obs(12.0, 31.5, 4, 3, 0)
                obs29  = Obs(12.0, 34.0, 4, 2, 0)
                obs30  = Obs( 9.5, 36.0, 1, 2, 0)

                obs31  = Obs(12.5, 38.0, 5, 2, 0)
                obs32  = Obs(19.0, 39.0, 2, 4, 0)
                obs33  = Obs(16.0, 36.0, 2, 2, 0)
                obs34  = Obs(18.5, 35.0, 3, 2, 0)
                obs35  = Obs(15.5, 34.0, 3, 2, 0)
                obs36  = Obs(15.5, 40.0, 1, 2, 0)
                obs37  = Obs(15.5, 29.5, 3, 1, 0)
                obs38  = Obs(32.0,  9.0, 4, 2, 0)
                obs39  = Obs(29.5, 11.0, 1, 2, 0)

                obs40  = Obs(39.0, 14.0, 2, 4, 0)
                obs41  = Obs(36.0, 11.0, 2, 2, 0)
                obs42  = Obs(38.5, 11.0, 3, 2, 0)
                obs43  = Obs(35.5,  9.0, 3, 2, 0)
                obs44  = Obs(35.5, 15.0, 1, 2, 0)
                obs45  = Obs(38.5,  4.5, 3, 1, 0)
                obs46  = Obs(18.5, 14.0, 1, 4, 0)
                obs47  = Obs(15.5, 17.0, 1, 1, 0)
                obs48  = Obs(15.5, 19.0, 1, 2, 0)
                obs49  = Obs(32.0,  4.0, 4, 1, 0)
                obs50  = Obs(38.0,  2.0, 4, 1, 0)
                obs51  = Obs(49.5, 11.0, 1, 2, 0)
                obs52  = Obs(52.0,  9.0, 4, 2, 0)
                obs53  = Obs(55.5,  9.0, 3, 2, 0)
                obs54  = Obs(52.0,  4.0, 4, 1, 0)
                obs55  = Obs(58.5,  4.5, 3, 1, 0)
                obs56  = Obs(58.0,  2.0, 4, 1, 0)
                obs57  = Obs(56.0, 11.0, 2, 2, 0)
                obs58  = Obs(58.5, 11.0, 3, 2, 0)
                obs59  = Obs(59.0, 14.0, 2, 4, 0)
                obs60  = Obs(55.5, 15.0, 1, 2, 0)

                obs61  = Obs(49.0, 34.0, 2, 2, 0)
                obs62  = Obs(52.0, 31.5, 4, 3, 0)
                obs63  = Obs(52.0, 34.0, 4, 2, 0)
                obs64  = Obs(49.5, 36.0, 1, 2, 0)
                obs65  = Obs(52.5, 38.0, 5, 2, 0)
                obs66  = Obs(57.0, 42.0, 2, 2, 0)
                obs67  = Obs(59.0, 39.0, 2, 4, 0)
                obs68  = Obs(56.0, 36.0, 2, 2, 0)
                obs69  = Obs(58.5, 36.0, 3, 2, 0)
                obs70  = Obs(55.5, 34.0, 3, 2, 0)
                obs71  = Obs(55.5, 40.0, 1, 2, 0)
                obs72  = Obs(55.5, 29.5, 3, 1, 0)

                obs73  = Obs(50.0, 54.0, 2, 1, 0)
                obs74  = Obs(49.5, 51.0, 1, 2, 0)
                obs75  = Obs(52.0, 49.0, 4, 2, 0)
                obs76  = Obs(55.5, 49.0, 3, 2, 0)
                obs77 = Obs(35.5, 54.0, 3, 2, 0)
                obs78 = Obs(35.5, 59.0, 1, 2, 0)
                obs79 = Obs(35.5, 49.5, 3, 1, 0)
                obs80  = Obs(56.0, 51.0, 2, 2, 0)
                obs81  = Obs(58.5, 51.0, 3, 2, 0)
                obs82  = Obs(59.0, 54.0, 2, 4, 0)
                obs83  = Obs(55.5, 55.0, 1, 2, 0)

                obs84  = Obs( 8.0, 54.0, 4, 2, 0)
                obs85  = Obs(12.0, 51.5, 4, 3, 0)
                obs86  = Obs(12.0, 54.0, 4, 2, 0)
                obs87  = Obs( 9.5, 56.0, 1, 2, 0)
                obs88  = Obs(19.0, 58.0, 2, 4, 0)
                obs89  = Obs(16.0, 56.0, 2, 2, 0)
                obs90  = Obs(18.5, 55.0, 3, 2, 0)
                obs91  = Obs(15.5, 54.0, 3, 2, 0)
                obs92  = Obs(15.5, 59.0, 1, 2, 0)
                obs93  = Obs(15.5, 49.5, 3, 1, 0)

                obs94  = Obs(28.0, 54.0, 4, 2, 0)
                obs95  = Obs(32.0, 51.5, 4, 3, 0)
                obs96  = Obs(32.0, 54.0, 4, 2, 0)
                obs97  = Obs(29.5, 56.0, 1, 2, 0)
                obs98  = Obs(39.0, 58.0, 2, 4, 0)
                obs99  = Obs(36.0, 56.0, 2, 2, 0)
                obs100 = Obs(38.5, 55.0, 3, 2, 0)

                # # Case 3 (with 50 obs) & Case 4 (with 100 obs)
                # area = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])  # Case 3
                # # area = Polygon([(0, 0), (65, 0), (65, 65), (0, 65)])  # Case 4
                # x_goal = Polygon([(16, 12.5), (17, 12.5), (17, 13), (16, 13)])
                # rob = Rob(40, 45, 0.2)

                # obs1  = Obs( 8.0,  9.0, 4, 2, 0)
                # obs2  = Obs(11.0, 17.0, 2, 6, 0)
                # obs3  = Obs(12.0,  6.5, 4, 3, 0)
                # obs4  = Obs(12.0,  9.0, 4, 2, 0)
                # obs5  = Obs( 9.5, 11.0, 1, 2, 0)
                # obs6  = Obs(12.5, 13.0, 5, 2, 0)
                # obs7  = Obs(17.0, 18.0, 2, 4, 0)
                # obs8  = Obs(19.0, 14.0, 2, 4, 0)
                # obs9  = Obs(16.0, 11.0, 2, 2, 0)
                # obs10 = Obs(18.5, 11.0, 3, 2, 0)
                # obs11 = Obs(15.5,  9.0, 3, 2, 0)
                # obs12 = Obs(15.5, 15.0, 1, 2, 0)
                # obs13 = Obs(15.5,  4.5, 3, 1, 0)
                # obs14 = Obs( 9.5, 16.0, 1, 8, 0)
                # obs15 = Obs(29.0, 34.0, 2, 2, 0)

                # obs16 = Obs(32.0, 31.5, 4, 3, 0)
                # obs17 = Obs(32.0, 34.0, 4, 2, 0)
                # obs18 = Obs(29.5, 36.0, 1, 2, 0)
                # obs19 = Obs(32.5, 38.0, 5, 2, 0)
                # obs20 = Obs(37.0, 43.0, 2, 4, 0)
                # obs21 = Obs(39.0, 39.0, 2, 4, 0)
                # obs22 = Obs(36.0, 36.0, 2, 2, 0)
                # obs23 = Obs(38.5, 36.0, 3, 2, 0)
                # obs24 = Obs(35.5, 34.0, 3, 2, 0)
                # obs25 = Obs(35.5, 40.0, 1, 2, 0)
                # obs26 = Obs(35.5, 29.5, 3, 1, 0)
                # obs27 = Obs( 8.0, 34.0, 4, 2, 0)
                # obs28 = Obs(12.0, 31.5, 4, 3, 0)
                # obs29 = Obs(12.0, 34.0, 4, 2, 0)
                # obs30 = Obs( 9.5, 36.0, 1, 2, 0)
 
                # obs31 = Obs(12.5, 38.0, 5, 2, 0)
                # obs32 = Obs(19.0, 18.0, 2, 4, 0)
                # obs33 = Obs(16.0, 36.0, 2, 2, 0)
                # obs34 = Obs(18.5, 35.0, 3, 2, 0)
                # obs35 = Obs(15.5, 34.0, 3, 2, 0)
                # obs36 = Obs(15.5, 40.0, 1, 2, 0)
                # obs37 = Obs(15.5, 29.5, 3, 1, 0)
                # obs38 = Obs(32.0,  9.0, 4, 2, 0)
                # obs39 = Obs(29.5, 11.0, 1, 2, 0)

                # obs40 = Obs(39.0, 14.0, 2, 4, 0)
                # obs41 = Obs(36.0, 11.0, 2, 2, 0)
                # obs42 = Obs(38.5, 11.0, 3, 2, 0)
                # obs43 = Obs(35.5,  9.0, 3, 2, 0)
                # obs44 = Obs(35.5, 15.0, 1, 2, 0)
                # obs45 = Obs(38.5,  4.5, 3, 1, 0)
                # obs46 = Obs(17.5, 14.0, 1, 4, 0)
                # obs47 = Obs(14.0, 17.0, 4, 1, 0)
                # obs48 = Obs(14.0, 19.0, 4, 2, 0)
                # obs49 = Obs(32.0,  4.0, 4, 1, 0)
                # obs50 = Obs(14.0, 21.0, 8, 2, 0)
                # obs51  = Obs(49.5, 11.0, 1, 2, 0)
                # obs52  = Obs(52.0,  9.0, 4, 2, 0)
                # obs53  = Obs(55.5,  9.0, 3, 2, 0)
                # obs54  = Obs(52.0,  4.0, 4, 1, 0)
                # obs55  = Obs(58.5,  4.5, 3, 1, 0)
                # obs56  = Obs(58.0,  2.0, 4, 1, 0)
                # obs57  = Obs(56.0, 11.0, 2, 2, 0)
                # obs58  = Obs(58.5, 11.0, 3, 2, 0)
                # obs59  = Obs(59.0, 14.0, 2, 4, 0)
                # obs60  = Obs(55.5, 15.0, 1, 2, 0)

                # obs61  = Obs(49.0, 34.0, 2, 2, 0)
                # obs62  = Obs(52.0, 31.5, 4, 3, 0)
                # obs63  = Obs(52.0, 34.0, 4, 2, 0)
                # obs64  = Obs(49.5, 36.0, 1, 2, 0)
                # obs65  = Obs(52.5, 38.0, 5, 2, 0)
                # obs66  = Obs(57.0, 42.0, 2, 2, 0)
                # obs67  = Obs(59.0, 39.0, 2, 4, 0)
                # obs68  = Obs(56.0, 36.0, 2, 2, 0)
                # obs69  = Obs(58.5, 36.0, 3, 2, 0)
                # obs70  = Obs(55.5, 34.0, 3, 2, 0)
                # obs71  = Obs(55.5, 40.0, 1, 2, 0)
                # obs72  = Obs(55.5, 29.5, 3, 1, 0)

                # obs73  = Obs(50.0, 54.0, 2, 1, 0)
                # obs74  = Obs(49.5, 51.0, 1, 2, 0)
                # obs75  = Obs(52.0, 49.0, 4, 2, 0)
                # obs76  = Obs(55.5, 49.0, 3, 2, 0)
                # obs77 = Obs(35.5, 54.0, 3, 2, 0)
                # obs78 = Obs(35.5, 59.0, 1, 2, 0)
                # obs79 = Obs(35.5, 49.5, 3, 1, 0)
                # obs80  = Obs(56.0, 51.0, 2, 2, 0)
                # obs81  = Obs(58.5, 51.0, 3, 2, 0)
                # obs82  = Obs(59.0, 54.0, 2, 4, 0)
                # obs83  = Obs(55.5, 55.0, 1, 2, 0)

                # obs84  = Obs( 8.0, 54.0, 4, 2, 0)
                # obs85  = Obs(12.0, 51.5, 4, 3, 0)
                # obs86  = Obs(12.0, 54.0, 4, 2, 0)
                # obs87  = Obs( 9.5, 56.0, 1, 2, 0)
                # obs88  = Obs(19.0, 58.0, 2, 4, 0)
                # obs89  = Obs(16.0, 56.0, 2, 2, 0)
                # obs90  = Obs(18.5, 55.0, 3, 2, 0)
                # obs91  = Obs(15.5, 54.0, 3, 2, 0)
                # obs92  = Obs(15.5, 59.0, 1, 2, 0)
                # obs93  = Obs(15.5, 49.5, 3, 1, 0)

                # obs94  = Obs(28.0, 54.0, 4, 2, 0)
                # obs95  = Obs(32.0, 51.5, 4, 3, 0)
                # obs96  = Obs(32.0, 54.0, 4, 2, 0)
                # obs97  = Obs(29.5, 56.0, 1, 2, 0)
                # obs98  = Obs(39.0, 58.0, 2, 4, 0)
                # obs99  = Obs(36.0, 56.0, 2, 2, 0)
                # obs100 = Obs(38.5, 55.0, 3, 2, 0)

                # # Case 5:
                # area = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
                # x_goal = Polygon([(38, 38), (38, 40), (40, 40), (40, 38)])
                # rob = Rob(10, 10, 0.5) # Need to change the default radius to 0.5 in real_c() parameter

                # obs1  = Obs(39, 37, 2, 2, 0)
                # obs2  = Obs(39, 35, 2, 2, 0)
                # obs3  = Obs(39, 33, 2, 2, 0)
                # obs4  = Obs(39, 31, 2, 2, 0)
                # obs5  = Obs(39, 29, 2, 2, 0)
                # obs6  = Obs(39, 27, 2, 2, 0)

                # obs7  = Obs(37, 39, 2, 2, 0)
                # obs8  = Obs(37, 37, 2, 2, 0)
                # obs9  = Obs(37, 35, 2, 2, 0)
                # obs10 = Obs(37, 33, 2, 2, 0)
                # obs11 = Obs(37, 31, 2, 2, 0)
                # obs12 = Obs(37, 29, 2, 2, 0)
                # obs13 = Obs(37, 27, 2, 2, 0)

                # obs14 = Obs(35, 39, 2, 2, 0)
                # obs15 = Obs(35, 33, 2, 2, 0)
                # obs16 = Obs(35, 31, 2, 2, 0)
                # obs17 = Obs(35, 29, 2, 2, 0)
                # obs18 = Obs(35, 27, 2, 2, 0)

                # obs19 = Obs(33, 39, 2, 2, 0)
                # obs20 = Obs(33, 37, 2, 2, 0)
                # obs21 = Obs(33, 35, 2, 2, 0)
                # obs22 = Obs(33, 33, 2, 2, 0)
                # obs23 = Obs(33, 31, 2, 2, 0)
                # obs24 = Obs(33, 29, 2, 2, 0)

                # obs25 = Obs(31, 39, 2, 2, 0)
                # obs26 = Obs(31, 37, 2, 2, 0)
                # obs27 = Obs(31, 35, 2, 2, 0)
                # obs28 = Obs(31, 33, 2, 2, 0)
                # obs29 = Obs(31, 31, 2, 2, 0)
                # obs30 = Obs(31, 29, 2, 2, 0)
                # obs31 = Obs(32, 27, 4, 2, 0)

                # obs32 = Obs(29, 39, 2, 2, 0)
                # obs33 = Obs(29, 37, 2, 2, 0)
                # obs34 = Obs(29, 35, 2, 2, 0)
                # obs35 = Obs(29, 33, 2, 2, 0)
                # obs36 = Obs(29, 31, 2, 2, 0)
                # obs37 = Obs(29, 29, 2, 2, 0)
                # obs38 = Obs(29, 27, 2, 2, 0)

                # obs39 = Obs(27, 39, 2, 2, 0)
                # obs40 = Obs(27, 37, 2, 2, 0)
                # obs41 = Obs(27, 34, 2, 4, 0)
                # obs42 = Obs(27, 27, 2, 2, 0)
                # obs43 = Obs(25, 39, 2, 2, 0)
                # obs44 = Obs(25, 37, 2, 2, 0)
                # obs45 = Obs(23, 39, 2, 2, 0)
                # obs46 = Obs(23, 37, 2, 2, 0)
                # obs47 = Obs(21, 39, 2, 2, 0)
                # obs48 = Obs(21, 37, 2, 2, 0)
                # obs49 = Obs(19, 39, 2, 2, 0)
                # obs50 = Obs(19, 37, 2, 2, 0)

                # # Case 6:
                # area = Polygon([(0, 0), (10, 0), (10, 2.5), (11, 2.5), (11, 1), (20, 1), (20, 20), (0, 20)])
                # x_goal = Polygon([(11, 1), (20, 1), (20, 2), (11, 2)])
                # rob = Rob(1, 1, 0.2)

                # obs1  = Obs(15.5, 3, 9, 2, 0)
                # obs2  = Obs(18, 5, 2, 2, 0)
                # obs3  = Obs(16, 6, 2, 2, 0)
                # obs4  = Obs(1, 4, 2, 2, 0)
                # obs5  = Obs(7, 3, 1, 2, 0)
                # obs6  = Obs(12, 9, 3, 2, 0)
                # obs7 = Obs(10, 5, 2, 2, 0)
                # obs8 = Obs(12, 5, 2, 2, 0)
                # obs9 = Obs(3, 7, 1, 1, 0)
                # obs10 = Obs(1, 19, 2, 2, 0)


                # Case 1&3&5
                All_Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14, obs15, obs16, obs17, obs18,obs19, obs20, 
                        obs21, obs22, obs23, obs24, obs25, obs26, obs27, obs28, obs29, obs30, obs31, obs32, obs33, obs34, obs35, obs36, obs37, obs38, obs39, obs40,
                        obs41, obs42, obs43, obs44, obs45, obs46, obs47, obs48, obs49, obs50]
                

                # # Case 2&4
                # All_Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14, obs15, obs16, obs17, obs18, obs19, obs20, 
                #         obs21, obs22, obs23, obs24, obs25, obs26, obs27, obs28, obs29, obs30, obs31, obs32, obs33, obs34, obs35, obs36, obs37, obs38, obs39, obs40,
                #         obs41, obs42, obs43, obs44, obs45, obs46, obs47, obs48, obs49, obs50, obs51, obs52, obs53, obs54, obs55, obs56, obs57, obs58, obs59, obs60,
                #         obs61, obs62, obs63, obs64, obs65, obs66, obs67, obs68, obs69, obs70, obs71, obs72, obs73, obs74, obs75, obs76, obs77, obs78, obs79, obs80,
                #         obs81, obs82, obs83, obs84, obs85, obs86, obs87, obs88, obs89, obs90, obs91, obs92, obs93, obs94, obs95, obs96, obs97, obs98, obs99, obs100]
                

                # # Case 6
                # All_Obs = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10]


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