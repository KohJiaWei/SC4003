from typing import List
import time
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from MazeLayout import State, Direction

class UtilityPlot:
    """Handles tracking and plotting of utility estimates over iterations."""
    def __init__(self):
        self.data = {}

    def add_data(self, utilities: List[List[float]], layout: List[List[State]]):
        for i, row in enumerate(utilities):
            for j, utility in enumerate(row):
                if layout[i][j].is_obstacle:
                    continue
                key = f"State({i},{j})"
                self.data[key] = self.data.get(key, []) + [utility]

    def plot(self):
        fig, ax = plt.subplots(figsize=(18.5, 9.5))
        for label, data in self.data.items():
            ax.plot(data, label=label)
        ax.set_xlabel("Iterations", fontsize=20)
        ax.set_ylabel("Utility Estimates", fontsize=20)
        ax.legend(ncol=2, loc="lower right", fontsize=14)
        plt.tight_layout()
        plt.show()

class GridWorldPlotter(tk.Frame):
    """Handles visualization of the maze, utilities, and policy actions."""
    def __init__(self, master=None, cell_size=70, cols=1):
        super().__init__(master)
        self.master = master
        self.cell_size = cell_size
        self.cols = cols
        self.n_cols = self.n_rows = 0
        self.canvas_count = 0

    def _load_canvas(self, layout, title=""):
        """Creates a new canvas for plotting the grid."""
        self.n_rows = len(layout)
        self.n_cols = len(layout[0])
        row_pos = self.canvas_count // self.cols
        col_pos = self.canvas_count % self.cols

        container = tk.Frame(self)
        container.grid(row=row_pos, column=col_pos, padx=5, pady=5)
        if title:
            tk.Label(container, text=title).pack(side=tk.TOP, pady=(0, 5))

        canvas = tk.Canvas(container, width=self.n_cols * self.cell_size,
                           height=self.n_rows * self.cell_size)
        canvas.pack()
        self.canvas_count += 1
        return canvas

    def _draw_cell(self, canvas, x1, y1, x2, y2, text="", fill="#FFFFFF", text_color="#000000", font_size=10):
        """Draws a single cell with text inside."""
        canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=fill)
        center_x, center_y = x1 + self.cell_size / 2, y1 + self.cell_size / 2
        canvas.create_text(center_x, center_y, text=text, fill=text_color, font=("Purisa", font_size))

    def draw_maze(self, layout: List[List[State]], title="Maze", font_size=15):
        """Draws the maze with walls, goals, and non-terminal states."""
        canvas = self._load_canvas(layout, title)
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                state = layout[row][col]

                fill, text = self._get_state_color_text(state)
                self._draw_cell(canvas, x1, y1, x2, y2, text, fill, font_size=font_size)

    def _get_state_color_text(self, state):
        """Determines the color and text for each cell."""
        if state.is_obstacle:
            return "#808080", ""  # Dark Grey for Walls
        elif state.reward_value > 0:
            return "#A0D995", f"+{state.reward_value}"  # Desaturated Green for Positive Goals
        elif state.reward_value < -0.5:
            return "#D99352", f"{state.reward_value}"  # Muted Orange for Negative Goals
        else:
            return "#DADADA", ""  # Light Grey for Non-terminal States

    def draw_estimated_utilities(self, layout: List[List[State]], utilities: List[List[float]], title="Estimated Utilities", font_size=9):
        """Draws estimated utility values inside each state."""
        canvas = self._load_canvas(layout, title)
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                state = layout[row][col]

                if state.is_obstacle:
                    fill = "#808080"
                    text = ""
                else:
                    fill = "#DADADA"
                    text = f"{utilities[row][col]:.3f}"

                self._draw_cell(canvas, x1, y1, x2, y2, text, fill, font_size=font_size)

    def draw_action(self, layout: List[List[State]], policy: List[List[Direction]], title="Policy", font_size=20):
        """Draws the optimal action arrows inside each state."""
        canvas = self._load_canvas(layout, title)
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                state = layout[row][col]

                if state.is_obstacle:
                    fill = "#808080"
                    text = ""
                else:
                    fill = "#DADADA"
                    text = policy[row][col].icon

                self._draw_cell(canvas, x1, y1, x2, y2, text, fill, font_size=font_size)

class ComplexityPlotter:
    """Handles tracking and plotting of algorithm complexity metrics."""
    def __init__(self):
        self.times = {}
        self.sizes = {}
        self.num_iterations = {}

    def add_data(self, key, size, func, *args, **kwargs):
        """Measures execution time and number of iterations for convergence."""
        self.sizes[key] = self.sizes.get(key, []) + [size]
        start_time = time.perf_counter()
        iterations = func(*args, **kwargs)
        end_time = time.perf_counter()

        self.times[key] = self.times.get(key, []) + [end_time - start_time]
        self.num_iterations[key] = self.num_iterations.get(key, []) + [np.log(iterations)]

    def _plot(self, data, ylabel, title):
        fig, ax = plt.subplots(figsize=(18.5, 9.5))
        for label, values in data.items():
            ax.plot(self.sizes[label], values, label=label)
        ax.set_xlabel("Size of maze", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.legend(loc="upper left", fontsize=14)
        plt.title(title, fontsize=22)
        plt.tight_layout()
        plt.show()

    def plot_times(self):
        """Plots time complexity."""
        self._plot(self.times, "Time (s)", "Time Complexity")

    def plot_iterations(self):
        """Plots log number of iterations."""
        self._plot(self.num_iterations, "Log Iterations", "Convergence Complexity")
