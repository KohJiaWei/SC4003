import numpy as np
from typing import Tuple

class State:
    def __init__(self, reward_value=-0.05, is_goal_state=False, is_obstacle=False):
        self.reward_value = float(reward_value)
        self.is_goal_state = is_goal_state
        self.is_obstacle = is_obstacle
    def __repr__(self):
        if self.is_goal_state and self.reward_value == 1:
            return "P"  # Positive Goal
        elif self.is_goal_state and self.reward_value == -1:
            return "N"  # Negative Goal
        elif self.is_obstacle:
            return "W"  # Wall
        return "." #non terminal state

POS = State(reward_value=1, is_goal_state=True)
NEG = State(reward_value=-1, is_goal_state=True)
NTS = State(reward_value=-0.05) # non terminal state
WALL = State(reward_value=0, is_obstacle=True)

def get_q1_maze():
    return [
        [POS, WALL, POS, NTS, NTS, POS],
        [NTS, NEG, NTS, POS, WALL, NEG],
        [NTS, NTS, NEG, NTS, POS, NTS],
        [NTS, NTS, NTS, NEG, NTS, POS],
        [NTS, WALL, WALL, WALL, NEG, NTS],
        [NTS, NTS, NTS, NTS, NTS, NTS]
    ]

def generate_random_maze(size:Tuple[int, int]):
    choices = [POS, NEG, NTS, WALL]
    probabilities = [0.1, 0.1, 0.7, 0.1]
    print( np.random.choice(choices, size=size, p=probabilities))
    return np.random.choice(choices, size=size, p=probabilities)

def print_maze(maze):
    for row in maze:
        print(" ".join(str(cell) for cell in row))

# Generate and print a 5x5 random maze
maze = generate_random_maze((5, 5))
print_maze(maze)

