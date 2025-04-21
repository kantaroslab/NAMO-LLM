# NAMO-LLM: Efficient Navigation Among Movable Obstacles with Large Language Model Guidance

## Introduction
NAMO-LLM is a novel sampling-based planner designed to address Navigation Among Movable Obstacles (NAMO) problems in highly cluttered environments. The planner aims to guide the robot in incrementally relocating obstacles until a collision-free path to the goal region is found.

## Requirements:
* [Python >=3.6](https://www.python.org/downloads/)
* [Shapely](https://github.com/Toblerity/Shapely)
* [matplotlib](https://matplotlib.org)
* [openai](https://platform.openai.com/docs/overview)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [openpyxl](https://openpyxl.readthedocs.io/)
* [requests](https://docs.python-requests.org/)
* Built-in: `re`, `time`, `math`, `random`, `ast`

## Custom Modules
- `api.py`: Local module for accessing the **ChatGPT API**. You must implement this and add your API key.
- `gemini.py`: Local module for accessing the **Gemini API**. You must implement this and add your API key.
- `rrt.py`: Module implementing RRT path validation (used in `rrt_env.py`).
- `obs.py`: Contains the `Obs` class, defining obstacle geometry (used in `rrt_env.py`).
  
## System Configuration
- OS: macOS 13
- Python Version: 3.11
- Tested on: MacBook Pro (M1 chip) with 16GB RAM
- GPU: Not required

## File Descriptions

| File          | Description |
|---------------|-------------|
| `env.py`      | Implements Cases 1–5 using a hybrid LLM + sampling planner for NAMO.|
| `rrt_env.py`  | Implements Case 6, which includes **RRT-based validation** for obstacle movement feasibility. |
| `api.py` / `gemini.py` | Interface modules to call ChatGPT or Gemini for selecting obstacles to move. |
| `rrt.py`      | Implements a classical RRT planner to validate the proposed obstacle movement. |
| `obs.py`      | Defines the `Obs` class for obstacle geometry. |

## Inputs and Parameters

All inputs are **defined directly in the code** (in `__main__` section of `env.py` and `rrt_env.py`):

### Common Inputs:
- **Environment Geometry** (`area`) – defined as a polygon.
- **Goal Region** (`x_goal`) – defined as a small goal polygon.
- **Robot** (`Rob(x, y, r)`) – defined by center location and radius.
- **Obstacles** (`Obs(x, y, l, d, θ)`) – defined by center, length, width, and orientation.

### Case Switching:
You can toggle between different cases by **commenting/uncommenting** the corresponding blocks in `env.py`.  
- **Cases 1–2**: Area with 100–200 obstacles, goal near the center.
- **Case 5**: Dense obstacles near the goal corner.
- **Case 6 (in `rrt_env.py`)**: Complex corridor + RRT validation.

### LLM Settings:
- `prob1` – the probability to expand the highest-valued node in the tree.
- `prob2` – the probability to invoke LLM for choosing which obstacle to move.

### API Keys:
Replace `"Your API Key"` in `api.py` and `gemini.py` with your actual keys. See:
- [ChatGPT API](https://platform.openai.com/)
- [Gemini API](https://ai.google.dev/)

## Example Usage
To run NAMO-LLM with default test cases:

```bash
python env.py       # For Cases 1–5
python rrt_env.py   # For Case 6 (with RRT validation)
```
## Output

The planner will generate the following outputs during execution:

### 📊 Data Logs
- **Excel Files**: Performance metrics are saved as Excel spreadsheets.
  - Filename: `output{prob1}and{prob2}.xlsx`
  - Contains:
    - `Iteration`: Number of tree expansions until goal is reached
    - `Time`: Elapsed time (excluding API wait time)
    - `Path`: Length of the solution path (number of states)

### 🖼️ Visualizations
- **Free Space and Obstacle Plots**:
  - `img/plot_{i}.png` – Current environment state at iteration `i`

> All images are saved in the `/img` directory (make sure it exists or create it manually).
