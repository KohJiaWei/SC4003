from typing import List
import numpy as np
from MazeLayout import State, Direction
from plotGraph import UtilityPlot


class MDP:
    def __init__(self, layout:List[List[State]], discount:float):
        self.layout = layout
        self.height:int = len(self.layout)
        self.width:int = len(self.layout[0])
        self.discount = discount

        self.policy = [
            [Direction.LEFT for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.utilities = [
            [0.0 for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.prev_utilities = [
            [0.0 for _ in range(self.width)]
            for _ in range(self.height)
        ]

        self.utility_plotter = UtilityPlot()
    
    def _agent_will_move(self, i:int, j:int, di:int, dj:int):
        """
        Returns true if the action will move agent to a square
        Returns false if the action will move agent out of bounds or into a wall

        Args:
            i (int): x-coordinate of state
            j (int): y-coordinate of state
            di (int): x-direction of action vector
            dj (int): y-direction of action vector
        """ 
        return 0 <= i + di < self.height and 0 <= j + dj < self.width and \
            not self.layout[i+di][j+dj].is_obstacle
    
    def _get_expected_utility(self, i:int, j:int, action:Direction):
        '''
        Calculates ∑P(s'|s,a)U(s') - the expected utility of taking action a in state s

        Args:
            i (int): x-coordinate of state
            j (int): y-coordinate of state
            action (Direction): The action taken by the agent at this state

        Returns:
            float: The expected utilty of taking action a in state s
        '''
        value = 0

        # Add the discounted rewards of current policies to value
        # Intended outcome happens (P = 0.8)
        di, dj = action.vector
        # Action results in movement
        if self._agent_will_move(i, j, di, dj):
            value += 0.8 * self.prev_utilities[i+di][j+dj]
        # Action results in staying
        else:
            value += 0.8 * self.prev_utilities[i][j]

        # 1st unintended outcome happens (P = 0.1)
        di, dj = action.rotate_anticlockwise().vector
        # Action results in movement
        if self._agent_will_move(i, j, di, dj):
            value += 0.1 * self.prev_utilities[i+di][j+dj]
        # Action results in staying
        else:
            value += 0.1 * self.prev_utilities[i][j]

        # 2nd unintended outcome happens
        di, dj = action.rotate_clockwise().vector
        # Action results in movement
        if self._agent_will_move(i, j, di, dj):
            value += 0.1 * self.prev_utilities[i+di][j+dj]
        # Action results in staying
        else:
            value += 0.1 * self.prev_utilities[i][j]
        
        return value

    def _update_prev_values(self):
        """
        Updates the previous Q-values with the newly calculated Q-values
        """
        for i in range(self.height):
            for j in range(self.width):
                self.prev_utilities[i][j] = self.utilities[i][j]

    def plot_utilities(self):
        self.utility_plotter.plot()

    # For dev purposes
    def print_utilities(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.layout[i][j].is_obstacle:
                    print(f"[ {'#':^5.5} ]", end=" ")
                else:
                    print(f"[ {str(self.utilities[i][j]):^5.5} ]", end=" ")
            print()
    
    def print_actions(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.layout[i][j].is_obstacle:
                    print("[ # ]", end=" ")
                else:
                    print(f"[ {self.policy[i][j].icon} ]", end=" ")
            print()


class ValueIteration(MDP):
    def solve(self, error:float):
        """
        Finds the optimum policy and estimated utilities of the MDP

        Args:
            error (float): The threshold to terminate the value iteration algorithm
        """
        theta = error * (1 - self.discount) / self.discount
        iteration = 0

        while True:
            delta = 0.0
            iteration += 1
            
            # Iterate through each state in the maze
            for i in range(self.height):
                for j in range(self.width):
                    state = self.layout[i][j]
                    # Ignore state if state is a wall
                    if state.is_obstacle:
                        continue

                    # Find best action by calculating Q(s,a) for each action
                    max_q = float('-inf')
                    best_action = action = Direction.LEFT

                    # Try all four possible actions
                    for _ in range(4):
                        value = state.reward_value + self.discount * self._get_expected_utility(i, j, action)
                        if value > max_q:
                            max_q = value
                            best_action = action
                        action = action.rotate_clockwise()
                    
                    # Update the utlities and best action
                    self.policy[i][j] = best_action
                    self.utilities[i][j] = max_q
        
                    # Update maximum delta
                    delta = max(delta, abs(self.utilities[i][j] - self.prev_utilities[i][j]))
            
            # Updates the value of each state synchronously
            self._update_prev_values()

            # Add data to plot
            self.utility_plotter.add_data(self.utilities, self.layout)
        
            # If delta < theta, the policy has converged and we terminate the evaluation
            if delta < theta:
                break
                
        print(f"Value Iteration took {iteration} iterations to converge")
        return iteration

class PolicyIteration(MDP):
    def _get_linear_equation(self, i: int, j: int, action: Direction):
        """
        Constructs a linear equation for a given state-action pair based on the Bellman equation.
        Format: -U(s) + γ∑P(s'|s,π(s))U(s') = -R(s)
        """
        eqn = [0] * (self.width * self.height)
        if self.layout[i][j].is_obstacle:
            return eqn

        eqn[i * self.width + j] = -1  # -U(s)

        for prob, direction in [(0.8, action),
                                (0.1, action.rotate_anticlockwise()),
                                (0.1, action.rotate_clockwise())]:
            di, dj = direction.vector
            ni, nj = i + di, j + dj
            idx = (ni * self.width + nj) if self._agent_will_move(i, j, di, dj) else (i * self.width + j)
            eqn[idx] += self.discount * prob

        return eqn

    def _policy_evaluation(self):
        """
        Solves the system of equations to evaluate the current policy.
        """
        A = []
        B = [-self.layout[i][j].reward_value for i in range(self.height) for j in range(self.width)]
        for i in range(self.height):
            for j in range(self.width):
                A.append(self._get_linear_equation(i, j, self.policy[i][j]))
        x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return x.reshape((self.height, self.width))

    def _improve_policy(self):
        """
        Updates the policy using a greedy approach based on current utilities.
        Returns True if policy is unchanged.
        """
        unchanged = True
        for i in range(self.height):
            for j in range(self.width):
                if self.layout[i][j].is_obstacle:
                    continue

                best_action = None
                max_utility = float("-inf")
                action = Direction.LEFT
                for _ in range(4):
                    utility = self._get_expected_utility(i, j, action)
                    if utility > max_utility:
                        max_utility = utility
                        best_action = action
                    action = action.rotate_clockwise()

                if best_action != self.policy[i][j]:
                    self.policy[i][j] = best_action
                    unchanged = False
        return unchanged

    def solve(self, max_iterations=500, verbose=False):
        """
        Executes the full policy iteration loop until convergence or max_iterations is reached.
        
        Parameters:
        - max_iterations: maximum number of iterations to avoid infinite loops
        - verbose: whether to print utility matrices and policy each iteration
        """
        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Policy evaluation: solve the linear system
            self.utilities = self.prev_utilities = self._policy_evaluation()

            # Policy improvement
            unchanged = self._improve_policy()

            # For plotting convergence
            self.utility_plotter.add_data(self.utilities, self.layout)

            if verbose:
                print(f"Iteration {iteration}")
                self.print_utilities()
                self.print_actions()
                print("-" * 30)

            if unchanged:
                break

        if iteration == max_iterations:
            print(f"⚠️ Policy Iteration reached the maximum of {max_iterations} iterations without full convergence.")
        else:
            print(f"✅ Policy Iteration converged in {iteration} iterations.")

        return iteration

class ModifiedPolicyIteration(MDP):
    def _policy_evaluation(self, k:int):
        """
        Evaluates the policy approximately to give a reasonably good approximation of the utilities

        Args:
            k (int): The number of iterations of Bellman update
        """
        for _ in range(k):
            delta = 0.0
            
            # Iterate through each state in the maze
            for i in range(self.height):
                for j in range(self.width):
                    state = self.layout[i][j]
                    # Ignore state if state is a wall
                    if state.is_obstacle:
                        continue

                    action = self.policy[i][j]
                    self.utilities[i][j] = state.reward_value + self.discount * self._get_expected_utility(i, j, action)
        
                    # Update maximum delta
                    delta = max(delta, abs(self.utilities[i][j] - self.prev_utilities[i][j]))
            
            # Updates the value of each state synchronously
            self._update_prev_values()
        return self.utilities  # <-- Ensure utilities are returned
    
    def solve(self, k:int):
        """
        Finds the optimum policy and estimated utilities of the MDP

        Args:
            error (float): The number of iterations of Bellman update for policy evaluation
        """
        iteration = 0
        while True:
            iteration += 1
            self._policy_evaluation(k)
            unchanged = True
            for i in range(self.height):
                for j in range(self.width):
                    state = self.layout[i][j]
                    # Ignore state if state is a wall
                    if state.is_obstacle:
                        continue

                    # Find best action by calculating Q(s,a) for each action
                    max_utility = float('-inf')
                    best_action = action = Direction.LEFT

                    # Try all four possible actions
                    for _ in range(4):
                        utility = self._get_expected_utility(i, j, action)
                        if utility > max_utility:
                            max_utility = utility
                            best_action = action
                        action = action.rotate_clockwise()
                    
                    # If best action is different from current action, set unchanged to False
                    if best_action != self.policy[i][j]:
                        self.policy[i][j] = best_action
                        unchanged = False
            
            # Add data to plot
            self.utility_plotter.add_data(self.utilities, self.layout)

            # If policy has converged, exit algorithm
            if unchanged:
                break
        
        print(f"Modified Policy Iteration took {iteration} iterations to converge")
        return iteration