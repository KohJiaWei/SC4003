import tkinter as tk
from MDP import ValueIteration, PolicyIteration, ModifiedPolicyIteration
from plotGraph import GridWorldPlotter
from MazeLayout import get_q1_maze

def solve_and_plot(maze_plotter, solver_class, maze, title, **solver_kwargs):
    """
    Solves an MDP problem using the specified solver and plots the results.
    
    Parameters:
    - maze_plotter: GridWorldPlotter instance
    - solver_class: The MDP solver class (ValueIteration, PolicyIteration, etc.)
    - maze: The maze environment
    - title: The title for the plots
    - solver_kwargs: Additional keyword arguments for the solver's solve() method
    """
    print(f"Starting {title}...")
    solver = solver_class(maze, discount=0.99)
    solver.solve(**solver_kwargs)
    print(f"Finished {title}, now plotting...")

    solver.plot_utilities()
    maze_plotter.draw_estimated_utilities(solver.layout, solver.utilities, title=title, font_size=9)
    maze_plotter.draw_action(solver.layout, solver.policy, title=title, font_size=20)
    print(f"Plotted {title}.")


def main():
    root = tk.Tk()
    maze_plotter = GridWorldPlotter(root, cell_size=70, cols=4)
    maze_plotter.pack()

    q1_maze = get_q1_maze()
    maze_plotter.draw_maze(q1_maze)

    # Define solvers and their parameters
    solvers = [
        (ValueIteration, "Value Iteration", {"error": 1e-5}),
        (PolicyIteration, "Policy Iteration", {}),
        (ModifiedPolicyIteration, "Modified Policy Iteration", {"k": 50})
    ]

    # Solve and plot for each solver
    for solver_class, title, params in solvers:
        solve_and_plot(maze_plotter, solver_class, q1_maze, title, **params)

    root.mainloop() #close the matplotlib graph to proceed on

if __name__ == "__main__":
    main()
