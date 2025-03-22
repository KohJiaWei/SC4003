import tkinter as tk
from plotGraph import GridWorldPlotter, Part2Plotter
from MazeLayout import generate_random_maze
from MDP import ValueIteration, PolicyIteration, ModifiedPolicyIteration

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")  # Maximize window (Windows)
    root.geometry("1920x1080") 

    plotter = Part2Plotter()
    maze_plotter = GridWorldPlotter(root, cell_size=40, cols=3, max_size=300) 

    plotter = Part2Plotter()
    for s in range(5, 15, 3):
    # In your part2 code, replace the loop with:

        maze = generate_random_maze(size=(s, s))
        
        # Create NEW plotter instance for each maze
        maze_plotter = GridWorldPlotter(root, cell_size=45, cols=3)
        maze_plotter.pack()
        
        maze_plotter.draw_maze(maze, title=f"Size = {s}")
        
        # Keep original cell_size and font_size
        solver = ValueIteration(maze, discount=0.99)
        plotter.add_data("Value Iteration", s, solver.solve, error=1e-4)

        solver = PolicyIteration(maze, discount=0.99)
        plotter.add_data("Policy Iteration", s, lambda: solver.solve(max_iterations=50))

        solver = ModifiedPolicyIteration(maze, discount=0.99)
        plotter.add_data("Modified Policy Iteration", s, lambda: solver.solve(k=50))


        # Scale cell size down to ensure overall size is the same
        # cell_size = cell_size / (s + 3) * s
        # font_size = int(font_size / (s + 3) * s)

    plotter.plot_times()
    plotter.plot_iterations()

    root.mainloop()
