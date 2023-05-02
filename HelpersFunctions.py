import numpy as np
from scipy.linalg import block_diag
import matplotlib
from matplotlib import pyplot as plt


class GridWorld:

    def __init__(self, goal=None):

        if not hasattr(self, 'layout'):
            raise ValueError("Need layout in subclass")

        layout_lines = self.layout.splitlines()
        self._occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line))
                                    for line in layout_lines])

        # From any state the agent can perform one of four actions, up, down, left or right
        self._n_actions = 4
        self._n_states = int(np.sum(self._occupancy == 0))

        self._directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]

        self._to_state = {}  # maps (x, y) to state #
        state_num = 0
        for i in range(len(layout_lines)):
            for j in range(len(layout_lines[0])):
                if self._occupancy[i, j] == 0:
                    self._to_state[(i, j)] = state_num
                    state_num += 1
        self._to_cell = {v: k for k, v in self._to_state.items()}  # maps state # to (x, y)

        self._goal = goal
        self._init_states = list(range(self._n_states))
        if goal is not None:
            self._reward_func = self._compute_reward_func()
        self._current_cell = None
        self._transition_func = self._compute_transition_func()

        self.step_count = 0

    def _compute_reward_func(self):
        """
            Compute the reward function r(s, a)
        """

        reward_func = np.zeros([self._n_states,
                                self._n_actions])

        goal_cell = self._to_cell[self._goal]
        # cells reachable from goal
        neighbor_cells = self._alt_cells(goal_cell, None)
        for cell in neighbor_cells:
            state = self._to_state[cell]
            for action in range(self._n_actions):
                next_cell = tuple(cell +
                                  self._directions[action])
                if not self._occupancy[next_cell]:
                    next_state = self._to_state[next_cell]
                    if next_state == self._goal:
                        reward_func[state, action] += 1

        # it is possible to keep hitting walls after reaching the goal
        for action in range(self._n_actions):
            next_cell = tuple(goal_cell +
                              self._directions[action])
            if self._occupancy[next_cell]:
                reward_func[self._goal, action] += 1

        return reward_func.reshape([self._n_states * self._n_actions, 
                                    1])

    def _compute_transition_func(self):
        # compute the transition probability matrix P

        p = np.zeros([self._n_states, self._n_actions, self._n_states])

        for s in range(self._n_states):

            cell = self._to_cell[s]

            for a in range(self._n_actions):

                next_cell = tuple(cell + self._directions[a])
                if not self._occupancy[next_cell]:
                    p[s, a, self._to_state[next_cell]] = 1
                    alt_cells = self._alt_cells(cell, next_cell)
                    for c in alt_cells:
                        p[s, a, self._to_state[c]] = 0
                else:  # bump into wall, so stay the same
                    p[s, a, s] = 1.

        return p.reshape([self._n_states * self._n_actions, 
                          self._n_states])

    def _alt_cells(self, cell, next_cell=None):

        alt_cells = []
        for action in range(self._n_actions):
            alt_cell = tuple(cell + self._directions[action])
            if not self._occupancy[alt_cell] and alt_cell != next_cell:
                alt_cells.append(alt_cell)
        return alt_cells

    @property
    def P(self):
        return self._transition_func

    @property
    def init_states(self):
        return self._init_states

    @property
    def goal(self):
        return self._goal

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_states(self):
        return self._n_states

    @property
    def r(self):
        return self._reward_func

    @property
    def to_cell(self):
        return self._to_cell

    @property
    def to_state(self):
        return self._to_state



class FourRoom(GridWorld):

    def __init__(self, goal=None):

        self.layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        super().__init__(goal=goal)


def iterativePolicyEvaluation(env, Pi, gamma, tol=1e-10):
    """
        Iterative policy evaluation for a given policy
    """
    v = np.random.rand(env.n_states, 1)
    v[env.goal, 0] = 0.
    diff = float('inf')
    while diff > tol:
        v_new = Pi @ (env.r + gamma * env.P @ v)
        diff = np.amax(np.abs(v_new - v))
        v = v_new
    return v


def diagonalization(A, n_states, n_actions):
    """
        Input A is a matrix of shape (n_states, n_actions), OR a column vector
        of shape (n_states * n_actions, 1).
        This function returns the diagonalization of A. The resultant matrix
        is a block-diagonal matrix of shape (n_states, n_states * n_actions) 
        such that each row has at most n_actions non-zero elements.
    """
    A = A.reshape([n_states, n_actions])
    return block_diag(*list(A))


def deDiagonalization(A):
    """
        Reverse of diagonalization. It converts a block-diagona lmatrix of 
        shape (n_states, n_states * n_actions) into a matrix of shape
        (n_states, n_actions).
    """
    n_states = A.shape[0]
    n_actions = A.shape[1] // n_states
    a = [A[i, i*n_actions:(i+1)*n_actions] for i in range(n_states)]
    return np.array(a)


# Ploting helpers below
def func_to_vectorize(x, y, dx, dy, color, scalarMap, scaling=0.3):

    if dx != 0 or dy != 0:

        arrow_color = scalarMap.to_rgba(color)
        plt.arrow(x, y,
                  dx * scaling, dy * scaling,
                  color=arrow_color,
                  head_width=0.15, head_length=0.15)


def convert_grid_to_numpy(grid):

    mappings = {'w': 1.,
                ' ': 0}

    grid = grid.split('\n')
    grid = list(filter(lambda x: len(x) > 0, grid))
    m, n = len(grid), len(grid[0])

    matrix = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            matrix[i, j] = mappings[grid[i][j]]

    return matrix


def plot_policy(env, xv, yv, xdir, ydir, policy, action):

    xdir = np.zeros_like(xdir)
    ydir = np.zeros_like(ydir)
    arrow_colors = np.zeros_like(xdir)

    cNorm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='gray_r')
    for s in range(env.n_states):

        x, y = env.to_cell[s]
        if action == 0:
            ydir[x, y] = -policy[s, action]
        elif action == 1:
            ydir[x, y] = policy[s, action]
        elif action == 2:
            xdir[x, y] = -policy[s, action]
        else:
            xdir[x, y] = policy[s, action]
        arrow_colors[x, y] = policy[s, action]

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)
    # arrow_colors = scalarMap.to_rgba(policy[:, action])
    vectorized_arrow_drawing(xv, yv, xdir, ydir, arrow_colors, scalarMap)
    return xdir, ydir


def plot_grid_world(env, Pi, v):

    n_states = env.n_states
    n_actions = env.n_actions

    matrix = convert_grid_to_numpy(env.layout)
    v = 0.5 * (v - np.amin(v)) / (np.amax(v) - np.amin(v))
    for s in range(n_states):
        x, y = env.to_cell[s]
        matrix[x, y] = v[s]

    policy = deDiagonalization(Pi)

    m, n = matrix.shape
    # xv, yv = np.meshgrid(np.arange(m), np.arange(n))
    xv, yv = np.meshgrid(np.arange(0.5, 0.5 + n), np.arange(0.5, 0.5 + m))
    xdir = np.zeros_like(matrix)
    ydir = np.zeros_like(matrix)

    plt.figure(figsize=(9, 9))

    plt.pcolormesh(matrix, edgecolors='k', linewidth=0.5, cmap='hot_r')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.invert_yaxis()

    plot_policy(env, xv, yv, xdir, ydir, policy, 0)
    plot_policy(env, xv, yv, xdir, ydir, policy, 1)
    plot_policy(env, xv, yv, xdir, ydir, policy, 2)
    plot_policy(env, xv, yv, xdir, ydir, policy, 3)

    plt.xticks([])
    plt.yticks([])
    plt.show()
