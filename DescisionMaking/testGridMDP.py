import operator
orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]

#orientations = EAST, NORTH, WEST, SOUTH, EAST2, NORTH2, WEST2, SOUTH2 = [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 0), (0, 2), (-2, 0), (0, -2)]
turns = LEFT, RIGHT = (+1, -1)

def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]
def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)
def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))
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

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9, ):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

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

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None

    def check_consistency(self):

        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001
class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9, step = 2):
        grid.reverse()  # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations

        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a, step)

        # transitions: is an matrix,
        #   the row is the state, the col is the direction,
        #   and the value is the all possibilites of action:  <probability, destination>
        #   example: transitions[(0, 0)][(-1, 0)]:  [(0.8, (0, 0)), (0.1, (0, 2)), (0.1, (0, 0))]
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma)

    #TODO: need to change the probability and step
    #TODO: if actions is 2 step, then return ... else return ...

    #calculate the action: return all the possible results of (probability, destination)
    # parameter:

    # state: original position
    # action: the direction that the robot face to

    def calculate_T(self, state, action, step):
        if action:
            #example:
            #state: (0, 1)
            #action: (1, 0)
            #self.go((0,1), (1, 0)) = (0, 1)

            #L X    X X
            #S None X X
            #R X    X X

            #turn_right: (0, 2)
            #turn_left: (0, 0)
            return [(0.8, self.go(state, action, step)),
                    (0.1, self.go(state, turn_right(action), step)),
                    (0.1, self.go(state, turn_left(action), step))]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

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

    # def go(self, state, direction):
    #     """Return the state that results from going in this direction.
    #     Vector_add: state + direction
    #     exm: (1, 0) + (0, 1) = (1, 1)"""
    #
    #     state1 = vector_add(state, direction)
    #     return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        #chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', (2, 0): '>>', (0, 2): '^^', (-2, 0): '<<', (0, -2): 'vv', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration. [Figure 17.4]"""
    U1 = {s: 0 for s in mdp.states} # initial utility
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    #R: reward
    #T: action
    #gama: discount
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max(sum(p * U[s1] for (p, s1) in T(s, a)) for a in mdp.actions(s))

            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U



def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
        #argmax: max for the value. pi is the best policy, which determine where to go:
        # example: (0, 1): (0, 1), from (0, 1) go to (0, 1) direction
    return pi

#--------------------------revised -----------------------------------------------------------------
def value_iteration(mdp1, mdp2, epsilon=0.001):
    """Solving an MDP by value iteration. [Figure 17.4]"""
    U1 = {s: 0 for s in mdp1.states}  # initial utility
    R1, T1, gamma1 = mdp1.R, mdp1.T, mdp1.gamma
    R2, T2, gamma2 = mdp2.R, mdp2.T, mdp2.gamma
    # R: reward
    # T: action
    # gama: discount
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp1.states:
            U1[s] = R1(s) + gamma1 * max(sum(p * U[s1] for (p, s1) in T1(s, a)) for a in mdp1.actions(s))
            U1[s] = R2(s) + gamma2 * max(sum(p * U[s1] for (p, s1) in T2(s, a)) for a in mdp2.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma1) / gamma1:
            return U

def best_policy(mdp1, mdp2, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp1.states:
        tmp1 = max(mdp1.actions(s), key=lambda a: expected_utility(a, s, U, mdp1))
        tmp2 = max(mdp2.actions(s), key=lambda a: expected_utility(a, s, U, mdp2))
        pi[s] = max(tmp1, tmp2);
        #argmax: max for the value. pi is the best policy, which determine where to go:
        # example: (0, 1): (0, 1), from (0, 1) go to (0, 1) direction
    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""
    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))


#-------------------------------------------------------------------------------------------



input1 = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])
input2 = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, '__int__')
def print_table(table, header=None, sep='   ', numfmt='{}'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
        map(lambda seq: max(map(len, seq)),
            list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))

# pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))
# print pi
# print sequential_decision_environment.to_arrows(pi)
# print_table(sequential_decision_environment.to_arrows(pi))


pi = best_policy(input1, input2, value_iteration(input1, input2, .01))
print pi
print input1.to_arrows(pi)
print_table(input1.to_arrows(pi))

