import operator
from time import time
from numpy import *
import Queue

"""""[-1, 0] up; [0, 1] right; [1, 0] down; [0, -1] left
     turn left, inc = -1
     headings: the direction of ahead """
orientations = [(-1, 0), (1, 0), (0, -1), (0, 1)]
turns = LEFT, RIGHT = (-1, +1)


def turn_heading(heading, inc, headings=[(-1, 0), (0, 1), (1, 0), (0, -1)]):
    return headings[(headings.index(heading) + inc) % len(headings)]

def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))
def act_multiply(actlist, step):
    res = [];
    for a in actlist:
        res.append((a[0] * step, a[1] * step))
    return res
def vector_multiply(a, b):
    return (a[0] * b, a[1] * b);

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(self, orientations, terminals, transitions=None, reward=None, states=None, gamma=0.9, step = 1):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states
        # self.states = states or self.get_states_from_transitions(transitions)
        self.step = step

        if isinstance(orientations, list):
            # if actlist is a list, all states have the same actions
            self.orientations = orientations;
            self.actlist = act_multiply(orientations, self.step)

        elif isinstance(orientations, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = act_multiply(orientations, self.step)
            self.orientations = orientations;

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""
        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""
        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def get_orientations(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.orientations

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals,  gamma=.9, step = 1, p = 0.8):
        #grid.reverse()  # because we want row 0 on bottom, not on top

        self.p_ = (1 - p) / 2
        self.p = p
        reward = {}

        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(0, self.rows):
            for y in range(0, self.cols):
                if grid[x][y] is not None:
                    states.add((x, y))
                    reward[(x, y)] = grid[x][y]
        self.states = states
        actlist = orientations

        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][vector_multiply(a, step)] = self.calculate_T(s, a, step)

        # transitions: is an matrix,
        #   the row is the state, the col is the direction,
        #   and the value is the all possibilites of action:  <probability, destination>
        #   example: transitions[(0, 0)][(-1, 0)]:  [(0.8, (0, 0)), (0.1, (0, 2)), (0.1, (0, 0))]
        MDP.__init__(self, orientations=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma, step = step)


    """calculate the action: return all the possible results of (probability, destination)
    parameter:
        state: original position
        action: the direction that the robot face to
    example:
        state: (0, 1)
        action: (1, 0)
        self.go((0,1), (1, 0)) = (0, 1)
        L X    X X
        S None X X
        R X    X X
        turn_right: (0, 2)
        turn_left: (0, 0)"""


    def calculate_T(self, state, action, step):

        if action:
            if self.p == 1:
                return [(self.p, self.go(state, action, step))]
            else:
                return [(self.p, self.go(state, action, step)),
                    (self.p_, self.go(state, turn_right(action), step)),
                    (self.p_, self.go(state, turn_left(action), step))]
        else:
            return [(0.0, state)]

    def T(self, state, actions):
        return self.transitions[state][actions] if actions else [(0.0, state)]

    def go(self, state, direction, step):
        """Return the state that results from going in this direction.
        Vector_add: state + direction
        exm: (1, 0) + (0, 1) = (1, 1)"""
        state_updated = state
        for i in range(1, step + 1):
            state_updated = vector_add(state_updated, direction)
            if state_updated not in self.states:
                return state
        return state_updated

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
       # list = zeros(self.rows * self.cols, str).reshape(self.rows, self.cols)
        list = [[0 for i in range(self.cols)] for j in range(self.rows)]

        for k in mapping.keys():
            i = k[0]
            j = k[1]
            list[i][j] = mapping[k]
        return list

    def to_words(self, policy):
        chars = {(0, 1): 'Walk Right', (-1, 0): 'Walk Up', (0, -1): 'Walk Left', (1, 0): 'Walk Down',(0, 2): 'Run Right', (-2, 0): 'Run Up', (0, -2): 'Run Left', (2, 0): 'Run Down', None: 'Exit'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

#--------------------------revised -----------------------------------------------------------------
def value_iteration(values, mdp1, mdp2, epsilon=0.001):
    """Solving an MDP by value iteration. [Figure 17.4]"""
    U1 = values  # initial utility
    R1, T1, gamma1 = mdp1.R, mdp1.T, mdp1.gamma
    R2, T2, gamma2 = mdp2.R, mdp2.T, mdp2.gamma
    # R: reward
    # T: action
    # gama: discount
    now_time = time()
    count = 0;
    while True:
        print "Iterative: ", count
        print(str(time() - start) + "s")
        count += 1
    # for i in range(10000):
        U = U1.copy()
        delta = 0
        for s in mdp1.states:
            tmp1 = R1(s) + gamma1 * max(sum(p * U[s1] for (p, s1) in T1(s, a)) for a in mdp1.actions(s))
            tmp2 = R2(s) + gamma2 * max(sum(p * U[s1] for (p, s1) in T2(s, a)) for a in mdp2.actions(s))
            U1[s] = max(tmp1, tmp2);
            delta = max(delta, abs(U1[s] - U[s]))


        print U1
        if delta < epsilon * (1 - gamma1) / gamma1:
            return U
    return U1

def best_policy(mdp1, mdp2, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp1.states:
        pi[s] = max(mdp1.actions(s) + mdp2.actions(s), key=lambda a: expected_utility(a, s, U, mdp1, mdp2))

    return pi

def expected_utility(a, s, U, mdp1, mdp2):
    """The expected utility of doing a in state s, according to the MDP and U."""
    if a in mdp1.actions(s):
        res = mdp1.R(s) + mdp1.gamma * sum(p * U[s1] for (p, s1) in mdp1.T(s, a));
    elif a in mdp2.actions(s):
        res = mdp2.R(s) + mdp2.gamma * sum(p * U[s1] for (p, s1) in mdp2.T(s, a));
    return res;


#-------------------------------------------------------------------------------------------


def test(answer, my_answer):
    row = len(answer)
    col = len(answer[0])
    if row != len(my_answer) or col != len(my_answer[0]):
        print "Size not equal"
        return

    # for i in range(0, row):
    #     print answer[i]
    # print "begin to print my_answer: "
    # for i in range(0, row):
    #     print my_answer[i]

    fail = False
    for i in range(0, row):
        for j in range(0, col):
            flag = answer[i][j] == my_answer[i][j],
            # print flag,
            if not flag:
                fail = True
    if fail:
        print "Fail"
    else:
        print "All pass"

def write_file(table):
    rows = len(table)
    cols = len(table[0])
    fo = open("output.txt", 'w')
    for i in range(0, rows):
        for j in range(0, cols - 1):
            fo.write(table[i][j])
            fo.write(',')
        fo.write(table[i][cols - 1])
        fo.write("\n")
    fo.close()


def res_reverse(tmp):
    rows = len(tmp)
    cols = len(tmp[0])
    my_answer = [['None' for i in range(cols)] for j in range(rows)]

    # my_answer = zeros(rows * cols, str).reshape(rows, cols)
    for i in range(0, rows):
        for j in range(0, cols):
            if tmp[rows - 1 - i][j] == 0:
                my_answer[i][j] = 'None'
            elif tmp[rows - 1 - i][j] == 'Run Up':
                my_answer[i][j] = 'Run Down'
            elif tmp[rows - 1 - i][j] == 'Run Down':
                my_answer[i][j] = 'Run Up'
            elif tmp[rows - 1 - i][j] == 'Walk Down':
                my_answer[i][j] = 'Walk Up'
            elif tmp[rows - 1 - i][j] == 'Walk Up':
                my_answer[i][j] = 'Walk Down'
            else:
                my_answer[i][j] = tmp[rows - 1 - i][j]
    return my_answer

def read_output(filename):
    fo = open(filename)
    output = [line.strip("\n") for line in fo.readlines()]
    fo.close()
    output1 = []
    for line in output:
        output1.append(line.split(','))
    return output1

def create_grid(rows, cols, r, walls, terminals):
    grid = [[0 for i in range(cols)] for j in range(rows)]
    #grid = zeros(cols * rows, str).reshape(rows, cols);
    for i in range(0, rows):
        for j in range(0, cols):
            if (i, j) in walls:
                grid[i][j] = None
            elif (i, j) in terminals:
                grid[i][j] = terminals[(i, j)]
            else:
                grid[i][j] = r
    return grid

def read_and_process_input(filename):
    # import the file
    fo = open(filename)
    input = [line.strip("\n") for line in fo.readlines()]
    fo.close()

    # line 1: the rows number and cols number of grid
    size = input[0].split(',')
    rows, cols = int(size[0]), int(size[1])

    # line 2: wall number
    wall_number = int(input[1])

    # walls
    walls = set()
    for i in range(0, wall_number):
        w = input[2 + i].split(',')
        walls.add((int(w[0]) - 1, int(w[1]) - 1))
    # terminals number
    terminals_number = int(input[2 + wall_number])

    # terminals
    terminals_line = 2 + wall_number + 1
    terminals = {};
    for i in range(0, terminals_number):
        t = input[terminals_line + i].split(',')
        terminals[(int(t[0]) - 1, int(t[1]) - 1)] = float(t[2])

    # probability: p_run and p_walk
    p_line = terminals_line + terminals_number
    p = input[p_line].split(',')
    p_walk, p_run = float(p[0]), float(p[1])

    # rewards
    r = input[p_line + 1].split(',')

    gamma = float(input[p_line + 2])

    walk_grid = create_grid(rows, cols, float(r[0]), walls, terminals);
    run_grid = create_grid(rows, cols, float(r[1]), walls, terminals);
    return walk_grid, run_grid, gamma, p_walk, p_run, terminals
def test_2(my_output, output):
    my_output = read_output(my_output);
    output = read_output(output)
    test(output, my_output)

def equation(U, U1, mdp1, mdp2, s, gamma):
    R1, T1 = mdp1.R, mdp1.T
    R2, T2 = mdp2.R, mdp2.T

    tmp1 = R1(s) + gamma * max(sum(p * U1[s1] for (p, s1) in T1(s, a)) for a in mdp1.actions(s))
    tmp2 = R2(s) + gamma * max(sum(p * U1[s1] for (p, s1) in T2(s, a)) for a in mdp2.actions(s))
    U1[s] = max(tmp1, tmp2);

    return abs(U1[s] - U[s])

def valid(s, rows, cols, mdp):
    return s[0] < rows and s[0] >= 0 and s[1] < cols and s[1] >= 0 and mdp.grid[s[0]][s[1]] is not None;

def preprocessing_BFS(U, queue, mdp1, mdp2, visited):

    U1 = U.copy()
    rows = len(mdp1.grid)
    cols = len(mdp1.grid[0])
    delta = 0;
    count = 0

    while not queue.empty():
        node = queue.get()
        coun = count + 1

        # print node
        new_delta = equation(U, U1, mdp1, mdp2, node, mdp1.gamma)
        # print node, U1[node]
        delta = max(delta, new_delta)

        for dir in orientations:
            new_node = (node[0] + dir[0], node[1] + dir[1])
            if valid(new_node, rows, cols, mdp1) and new_node not in visited:
                queue.put(new_node)
                visited.add(new_node)

    return U1, delta

def BFS_iteration(input_walk, input_run, terminals, gamma):
    values = {s: 0 for s in input_walk.states}  # initial utility
    # print("Start to iteration" + str(time() - start) + "s")

    # values = zeros(input_walk.rows * input_walk.cols).reshape(input_walk.rows, input_walk.cols)

    epsilon = .000001
    count = 0

    for i in range(0):
        print "IterativeBFS: ", count
        print("time: " + str(time() - start) + "s")
        visited = set()
        queue = Queue.Queue()

        for seed in terminals:
            queue.put(seed)
            visited.add(seed)
            values[seed] = terminals[seed]
        values, delta = preprocessing_BFS(values, queue, input_run, input_walk, visited)
        if delta < epsilon * (1 - gamma) / gamma:
            break;
    return values
def main(input_file):


    #read and processing input file
    walk_grid, run_grid, gamma, p_walk, p_run, terminals = read_and_process_input(input_file)
    print("Start to create first mdp" + str(time() - start) + "s")

    #create two objects for run and walk
    input_run = GridMDP(run_grid, terminals,  gamma, 2, p_run)
    print("Start to create second mdp" + str(time() - start) + "s")

    input_walk = GridMDP(walk_grid, terminals,  gamma, 1, p_walk)

    print("Start to iteration_BFS" + str(time() - start) + "s")

    values = BFS_iteration(input_walk, input_run, terminals, gamma)

    #get the optimal utility
    print("start to iteration_value: " + str(time() - start) + "s")

    values = value_iteration(values, input_walk, input_run, .000001)

    print("End iteration, start best policy" + str(time() - start) + "s")

    #get the best policy
    pi = best_policy(input_walk, input_run, values)

    print("End best policy" + str(time() - start) + "s")
    #change the vector action to words action
    tmp = input_walk.to_words(pi)
    #reverse the result
    my_answer = res_reverse(tmp)

    #write my_answer to the file
    write_file(my_answer)

    test_2("output.txt", "output1.txt")

start = time()
main("input1.txt")
# [-0.2 -0.2 -0.2  0.   2.8]
# [-0.2 -0.2 -0.2 -0.2  2.7]
# [-0.2 -0.2  6.5  2.7  5. ]
# [-0.2  0.   5.7 -0.2  2.7]
# [ 5.8  5.7 10.   5.7  6.5]
# [-0.2 -0.2  5.7 -0.2 -0.2]