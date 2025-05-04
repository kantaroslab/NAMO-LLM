from openai import OpenAI
import base64
import requests


client = OpenAI(api_key="Your API Key")
MODEL="gpt-4o"
# MODEL="gpt-4-turbo"
# MODEL="o1-mini"


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def api_Mi(area, N, all_obs, adj_obs, id_list, x_goal, i):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {'Your API Key'}"
    }

    # image_ex = f"img/plot_{i}.png"
    # image_ex = encode_image(image_ex)

    build = {
    "model": MODEL,
    "messages": [
        {
        "role": "user",
        "content": [
            # {
            # "type": "image_url",
            # "image_url": {
            #     "url": f"data:image/jpeg;base64,{image_ex}"
            # }
            # },
            {
            "type": "text",
            "text": f"Task Description: You are a robot operating in a 40x40 grid area defined as {area}. The current positions of all obstacles are {all_obs}, and your goal is to reach the goal area located at {x_goal}.\
                  Your current free space is N = {N}. To achieve your goal, you must remove one obstacle adjacent to the free space. "
            },
            {
            "type": "text",
            "text": f"Rules:\
                    Adjacent Obstacles Only: You can only remove obstacles that are adjacent to the free space N. The list of adjacent obstacles is adj_obs={adj_obs}, and their IDs are id_list={id_list}. Obstacles not in adj_obs or id_list cannot be selected.\
                    Minimize Steps to the Goal: Select the obstacle that helps clear a path to the goal area {x_goal} with the fewest possible steps.\
                    Avoid Obstruction: Consider the layout of all obstacles. Avoid situations where removing one obstacle causes another to become blocked. For example: If obstacle A can only move upward but is blocked by obstacle B, prioritize removing obstacle B before A. Similarly, if multiple obstacles in adj_obs are interdependent, determine the correct order of movement to avoid obstruction."
            },
            # The output requirement for K=1
            {
            "type": "text",
            "text": "Output Requirement: Output the name of the selected obstacle as a single string, such as 'obs1', 'obs2', etc.\
                    DO NOT include any additional analysis or explanation in your output."
            }

            # # Edit the output requirement to change the value of K. The following is the output requirement for K=2
            # {
            # "type": "text",
            # "text": "Output Requirement: Your output MUST be a list containing the names of two selected obstacles, formatted as: ['obs1', 'obs2'].\
            #         DO NOT include any additional analysis or explanation."
            # }
        ]
        }
    ],
    "max_tokens": 10, # Edit the max_tokens to meet the output requirement
    }
    response_Mi = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=build)
    # print(response_Mi.json()['choices'][0]['message']['content'])
    if 'choices' in response_Mi.json():
        return response_Mi.json()['choices'][0]['message']['content']
    else:
        print(response_Mi.json())
        return None