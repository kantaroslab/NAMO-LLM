import google.generativeai as genai
import os
import PIL.Image
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

genai.configure(api_key='Your API Key')
model = genai.GenerativeModel("gemini-1.5-flash")
path1 = Path("img")
path2 = Path("pic")

def api_Mi(area, N, all_obs, adj_obs, id_list, x_goal, i):
    # p = PIL.Image.open(path1/f"plot_{i}.png")
    mi_prompt = [f"You are a robot operating in a 40x40 grid area defined as {area} with current positions of all obstacles={all_obs}. The goal area is {x_goal}.\
            Imagine you are a robot, your goal is to reach the goal area with the least number of steps, and your current free space N is {N}. \
            You can only move obstacles which are adjacent to N. The adjacent obstacles are all in adj_obs={adj_obs} and the adjacent-obstacle list is id_list={id_list}, you must pick name of one item from this id_list, as your output. ",
            "Please select one obstacle to remove according to the current free space, current position of obstacles and the current obtsacles' adjacency. \
            The output is constrained by adj_obs. You need to consider the positions of all obstacles, then choose the one you think is the best to move from the adjacent obstacles (adj_obs). \
            The goal is to reach the goal area with the least number of steps. \
            DO NOT choose obstacles which are not in the adj_obs.\
            You don't need to output any analysis. Your output MUST be and ONLY be the obstacle name, such as 'obs1', 'obs2', ..., 'obs100'...." #Edit this line to change the output requirement
        ]
    mi_resp = model.generate_content(mi_prompt,
            generation_config = genai.GenerationConfig(
            max_output_tokens=10    # Edit the max_output_tokens to meet the output requirement
            ))
    # print(mi_resp.text)
    return mi_resp.text