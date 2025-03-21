import numpy as np
from typing import Tuple

from typing import Tuple

class Direction:
    """
    A class representing a direction with an associated vector and icon.
    Contains singleton-like instances for UP, DOWN, LEFT, and RIGHT.
    """
    UP: "Direction"
    DOWN: "Direction"
    LEFT: "Direction"
    RIGHT: "Direction"

    def __init__(self, vector: Tuple[int, int], icon: str):
        self.vector = vector
        self.icon = icon

    def __str__(self):
        return self.icon

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Direction):
            return NotImplemented
        return self.vector == other.vector and self.icon == other.icon

    def __hash__(self):
        """Make the Direction instances hashable by hashing their unique vector."""
        return hash(self.vector)

    def rotate_clockwise(self) -> "Direction":
        """Rotate this direction 90 degrees clockwise."""
        clockwise_map = {
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT
        }
        return clockwise_map[self]

    def rotate_anticlockwise(self) -> "Direction":
        """Rotate this direction 90 degrees counter-clockwise."""
        anticlockwise_map = {
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT
        }
        return anticlockwise_map[self]

# Define the singleton directions after the class has been declared
Direction.UP = Direction((-1, 0), '↑')
Direction.DOWN = Direction((1, 0), '↓')
Direction.LEFT = Direction((0, -1), '←')
Direction.RIGHT = Direction((0, 1), '→')



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
    choices = [
        lambda: State(reward_value=1, is_goal_state=True),   # POS
        lambda: State(reward_value=-1, is_goal_state=True),  # NEG
        lambda: State(reward_value=-0.05),  # NTS
        lambda: State(reward_value=0, is_obstacle=True)  # WALL
    ]
    probabilities = [0.1, 0.1, 0.7, 0.1]

    maze = [[np.random.choice(choices, p=probabilities)() for _ in range(size[1])]
            for _ in range(size[0])]

    print_maze(maze)  # Print generated maze
    return maze


def print_maze(maze):
    for row in maze:
        print(" ".join(str(cell) for cell in row))

# Generate and print a 5x5 random maze
# maze = generate_random_maze((5, 5))
# print_maze(maze)

