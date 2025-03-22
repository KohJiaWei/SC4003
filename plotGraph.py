from typing import List
import time
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from MazeLayout import State, Direction


class UtilityPlot:
    
    
    def __init__(self):
        self.data = {}
    
    def add_data(self, utilities:List[List[float]], layout:List[List[State]]):
        for i, row in enumerate(utilities):
            for j, utility in enumerate(row):
                if layout[i][j].is_obstacle:
                    continue
                key = f"State({i},{j})"
                self.data[key] = self.data.get(key, []) + [utility]
    
    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 9.5)
        for label, data in self.data.items():
            ax.plot(data, label=label)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Utility Estimates', fontsize=20)
        ax.legend(ncol=2, loc='lower right', fontsize=20)
        plt.tight_layout()
        plt.show()

class GridWorldPlotter(tk.Frame):
    """
    A grid-plotter that creates a separate Canvas each time you call a draw method.
    It arranges these canvases in a grid layout based on 'cols'.

    - cell_size: size of each cell in pixels
    - cols: how many subplots (canvases) to place per row
    - max_size: maximum total pixel width/height to avoid an enormous window
    """
    def __init__(self, master=None, cell_size=70, cols=1, max_size=600):
        super().__init__(master)
        self.master = master
        self.cell_size = cell_size
        self.cols = cols
        self.max_size = max_size
        self.canvas_count = 0
        self.n_rows = 0
        self.n_cols = 0
        
        # Container for grid layout
        self.main_container = tk.Frame(self)
        self.main_container.pack(fill='both', expand=True)
        


    # ------------------------------------------------------------
    #   1) Helper: Auto-scale the cell size to keep large mazes from exploding
    # ------------------------------------------------------------
    def _resize_cell_size(self, layout: List[List['State']]):
        """Auto-scales cells so large grids stay within max_size."""
        self.n_rows = len(layout)
        self.n_cols = len(layout[0])
        max_dim = max(self.n_rows, self.n_cols)

        # If it's too large, reduce cell_size so it fits in max_size
        if max_dim * self.cell_size > self.max_size:
            self.cell_size = self.max_size // max_dim

    # ------------------------------------------------------------
    #   2) Helper: Create a new sub-canvas for each "draw_XXX" call
    # ------------------------------------------------------------
    def _load_canvas(self, layout, title=""):
        self._resize_cell_size(layout)
        
        row_pos = self.canvas_count // self.cols
        col_pos = self.canvas_count % self.cols
        self.canvas_count += 1

        container = tk.Frame(self.main_container)
        container.grid(row=row_pos, column=col_pos, padx=5, pady=5)

        if title:
            tk.Label(container, text=title).pack()

        canvas = tk.Canvas(
            container,
            width=self.n_cols * self.cell_size,
            height=self.n_rows * self.cell_size
        )
        canvas.pack()
        return canvas
    # ------------------------------------------------------------
    #   3) Helpers to draw individual cells (walls & states)
    # ------------------------------------------------------------
    def _draw_wall(self, canvas: tk.Canvas, x1, y1, x2, y2):
        """Draws a wall (obstacle) cell."""
        canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="#808080")

    def _draw_state(self,
                    canvas: tk.Canvas,
                    x1, y1, x2, y2,
                    text="",
                    fill="#FFFFFF",
                    text_color="#000000",
                    font_size=10):
        """Draws a single rectangular cell with optional text."""
        canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=fill)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        canvas.create_text(cx, cy, text=text, fill=text_color, font=("Purisa", font_size))

    def _get_state_color_text(self, state):
        """
        Returns (fill_color, text_string) depending on the state's reward value.
        - Obstacle => #808080
        - Reward > 0 => #A0D995
        - Reward < -0.5 => #D99352
        - Else => #DADADA
        """
        if state.is_obstacle:
            return "#808080", ""
        elif state.reward_value > 0:
            return "#A0D995", f"+{state.reward_value}"
        elif state.reward_value < -0.5:
            return "#D99352", f"{state.reward_value}"
        else:
            return "#DADADA", ""

    # ------------------------------------------------------------
    #   4) Main "draw" methods
    # ------------------------------------------------------------
    def draw_maze(self,
                  layout: List[List['State']],
                  title="Maze",
                  font_size=15):
        """
        Draws the basic maze with colored cells according to reward values
        (green for positive, orange for negative) and grey for walls.
        """
        canvas = self._load_canvas(layout, title=title)

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                state = layout[row][col]
                if state.is_obstacle:
                    self._draw_wall(canvas, x1, y1, x2, y2)
                else:
                    fill, text = self._get_state_color_text(state)
                    self._draw_state(
                        canvas, x1, y1, x2, y2,
                        text=text, fill=fill, font_size=font_size
                    )

    def draw_estimated_utilities(self,
                                 layout: List[List['State']],
                                 utilities: List[List[float]],
                                 title="Estimated Utilities",
                                 font_size=9):
        """
        Draws the estimated utilities in each cell. Obstacles are walls.
        """
        canvas = self._load_canvas(layout, title=title)

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                state = layout[row][col]
                if state.is_obstacle:
                    self._draw_wall(canvas, x1, y1, x2, y2)
                else:
                    util = utilities[row][col]
                    # Optionally color them all a neutral color
                    # so utilities stand out as text
                    fill = "#DADADA"
                    text = f"{util:.3f}"

                    self._draw_state(
                        canvas, x1, y1, x2, y2,
                        text=text,
                        fill=fill,
                        font_size=font_size
                    )

    def draw_action(self,
                    layout: List[List['State']],
                    policy: List[List['Direction']],
                    title="Action",
                    font_size=20):
        """
        Draws the chosen policy action in each cell. Typically
        policy[row][col].icon might be something like '^', 'v', '<', '>'.
        """
        canvas = self._load_canvas(layout, title=title)

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                state = layout[row][col]
                if state.is_obstacle:
                    self._draw_wall(canvas, x1, y1, x2, y2)
                else:
                    action = policy[row][col]
                    # Use a light fill color so the arrow is visible
                    fill = "#DADADA"
                    text = f"{action.icon}"

                    self._draw_state(
                        canvas, x1, y1, x2, y2,
                        text=text,
                        fill=fill,
                        font_size=font_size
                    )



    def add_data(self, utilities: List[List[float]], layout: List[List[State]]):
        for i, row in enumerate(utilities):
            for j, utility in enumerate(row):
                if layout[i][j].is_obstacle:
                    continue
                key = f"State({i},{j})"
                self.data[key] = self.data.get(key, []) + [utility]

    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)
        for label, data in self.data.items():
            ax.plot(data, label=label)
        ax.set_xlabel("Iteration", fontsize=20)
        ax.set_ylabel("Utility Estimate", fontsize=20)
        ax.legend(ncol=2, loc="lower right", fontsize=15)
        plt.tight_layout()
        plt.show()

class Part2Plotter:
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
        # ** Make the figure window fill the screen **
        mng = plt.get_current_fig_manager()
        try:
            # On Windows, this should "zoom" to fill the screen
            mng.window.state("zoomed")
        except:
            pass
        plt.show()

    def plot_times(self):
        """Plots time complexity."""
        self._plot(self.times, "Time (s)", "Time Complexity")

    def plot_iterations(self):
        """Plots log number of iterations."""
        self._plot(self.num_iterations, "Log Iterations", "Convergence Complexity")


    def draw_maze(self, layout: List[List[State]], title="Maze"):
        """Draws the maze and ensures proper sizing."""
        self._resize_cell_size(layout)  # Auto-resize cells
        width = self.n_cols * self.cell_size
        height = self.n_rows * self.cell_size
        self.canvas.config(scrollregion=(0, 0, width, height))

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                state = layout[row][col]

                fill, text = self._get_state_color_text(state)
                self._draw_cell(x1, y1, x2, y2, text, fill)

    def _get_state_color_text(self, state):
        """Determines the color and text for each cell."""
        if state.is_obstacle:
            return "#808080", ""  # Dark Grey for Walls
        elif state.reward_value > 0:
            return "#A0D995", f"+{state.reward_value}"  # Green for Rewards
        elif state.reward_value < -0.5:
            return "#D99352", f"{state.reward_value}"  # Orange for Negative Goals
        else:
            return "#DADADA", ""  # Light Grey for Normal States

    def _draw_cell(self, x1, y1, x2, y2, text="", fill="#FFFFFF"):
        """Draws a single cell with text inside."""
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=fill)
        center_x, center_y = x1 + self.cell_size / 2, y1 + self.cell_size / 2
        self.canvas.create_text(center_x, center_y, text=text, fill="black", font=("Purisa", 10))

