# NAMO-LLM: Efficient Navigation Among Movable Obstacles with Large Language Model Guidance

## How to use this code

###env.py:
This script implements Cases 1 through 5 as described in our paper. In these cases, any obstacle that is blocked by others is assumed to be unreachable by the robot. All relevant case parameters are defined within the file.

###rrt_env.py
This script implements Case 6 from the paper. In this scenario, the movement of a specific adjacent obstacle is hindered by nearby obstacles. To address this, we have integrated the RRT (Rapidly-exploring Random Tree) algorithm for obstacle validation.

### api.py & gemini.py
These modules require API keys to access the ChatGPT and Gemini services. Please obtain your own API key from the respective websites and replace the placeholder "Your API Key" in the files with your actual key.
