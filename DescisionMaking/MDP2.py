#!/usr/bin/env python
"""Some code are referred from textbook material at aima.cs.berkeley.edu"""
import time

class MDP:
    def __init__(self, init, actlist, terminals, transitions=None, states=None, gamma=0.9):
        self.states = states
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        self.transitions = transitions or {}
        self.gamma = gamma

    def R(self, action):
        pass

    def T(self, state, action):
        pass

    def actions(self):
        return self.actlist


class GridMDP(MDP):
    def __init__(self, grid, terminals, pwalk, prun, rwalk, rrun, init, gamma):

        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        self.rwalk = rwalk
        self.rrun = rrun
        for x in range(self.rows):
            for y in range(self.cols):
                if grid[x][y]:
                    states.add((x, y))

        self.states = states

        walkopts = WALKUP, WALKRIGHT, WALKDOWN, WALKLEFT = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        runopts = RUNUP, RUNRIGHT, RUNDOWN, RUNLEFT = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        actlist = WALKUP, WALKDOWN, WALKLEFT, WALKRIGHT, RUNUP, RUNDOWN, RUNLEFT, RUNRIGHT

        patialpwalk = 0.5 * (1 - pwalk)
        patialprun = 0.5 * (1 - prun)

        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in walkopts:
                transitions[s][a] = self.calculate_Tforwalk(s, a, pwalk, patialpwalk, walkopts)
            for a in runopts:
                transitions[s][a] = self.calculate_Tforrun(s, a, prun, patialprun, runopts, walkopts)

        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     states=states, gamma=gamma)

    def calculate_Tforwalk(self, state, action, pwalk, patialpwalk, walkopts):

            state0 = state[0]
            state1 = state[1]

            #Straight
            rowstraight = state0 + action[0]
            colstraight = state1 + action[1]
            if 0 <= rowstraight < self.rows and 0 <= colstraight < self.cols and self.grid[rowstraight][colstraight]:
                straight = True
            else:
                straight = False

            #Left
            leftaction = walkopts[(walkopts.index(action) - 1) % 4]
            rowleft = state0 + leftaction[0]
            colleft = state1 + leftaction[1]

            if 0 <= rowleft < self.rows and 0 <= colleft < self.cols and self.grid[rowleft][colleft]:
                left = True
            else:
                left = False

            #Right
            rightaction = walkopts[(walkopts.index(action) + 1) % 4]
            rowright = state0 + rightaction[0]
            colright = state1 + rightaction[1]

            if 0 <= rowright < self.rows and 0 <= colright < self.cols and self.grid[rowright][colright]:
                right = True
            else:
                right = False

            if straight and left and right:
                return [(pwalk, (rowstraight, colstraight)), (patialpwalk, (rowleft, colleft)), (patialpwalk, (rowright, colright))]

            elif not straight and left and right:
                return [(pwalk, (state0, state1)), (patialpwalk, (rowleft, colleft)), (patialpwalk, (rowright, colright))]

            elif straight and not left and right:
                return [(pwalk, (rowstraight, colstraight)), (patialpwalk, (state0, state1)), (patialpwalk, (rowright, colright))]

            elif straight and left and not right:
                return [(pwalk, (rowstraight, colstraight)), (patialpwalk, (rowleft, colleft)), (patialpwalk, (state0, state1))]

            elif straight and not left and not right:
                return [(pwalk, (rowstraight, colstraight)), ((1-pwalk), (state0, state1))]

            elif not straight and left and not right:
                return [(patialpwalk, (rowleft, colleft)), (1 - patialpwalk, (state0, state1))]

            elif not straight and not left and right:
                return [(patialpwalk, (rowright, colright)), (1 - patialpwalk, (state0, state1))]

            else:
                return [(1, state0, state1)]

    def calculate_Tforrun(self, state, action, prun, patialprun, runopts, walkopts):

        state0 = state[0]
        state1 = state[1]

        # Straight
        rowstraight = state0 + action[0]
        colstraight = state1 + action[1]

        walkaction = walkopts[runopts.index(action)]
        row0 = state0 + walkaction[0]
        col0 = state1 + walkaction[1]


        if 0 <= rowstraight < self.rows and 0 <= colstraight < self.cols and self.grid[rowstraight][colstraight] and self.grid[row0][col0]:
            straight = True
        else:
            straight = False

        # Left
        leftaction = runopts[(runopts.index(action) - 1) % 4]
        rowleft = state0 + leftaction[0]
        colleft = state1 + leftaction[1]

        walkaction = walkopts[runopts.index(leftaction)]
        row0 = state0 + walkaction[0]
        col0 = state1 + walkaction[1]

        if 0 <= rowleft < self.rows and 0 <= colleft < self.cols and self.grid[rowleft][colleft] and self.grid[row0][col0]:
            left = True
        else:
            left = False

        # Right
        rightaction = runopts[(runopts.index(action) + 1) % 4]
        rowright = state0 + rightaction[0]
        colright = state1 + rightaction[1]

        walkaction = walkopts[runopts.index(rightaction)]
        row0 = state0 + walkaction[0]
        col0 = state1 + walkaction[1]

        if 0 <= rowright < self.rows and 0 <= colright < self.cols and self.grid[rowright][colright] and self.grid[row0][col0]:
            right = True
        else:
            right = False

        if straight and left and right:
            return [(prun, (rowstraight, colstraight)), (patialprun, (rowleft, colleft)),
                    (patialprun, (rowright, colright))]

        elif not straight and left and right:
            return [(prun, (state0, state1)), (patialprun, (rowleft, colleft)),
                    (patialprun, (rowright, colright))]

        elif straight and not left and right:
            return [(prun, (rowstraight, colstraight)), (patialprun, (state0, state1)),
                    (patialprun, (rowright, colright))]

        elif straight and left and not right:
            return [(prun, (rowstraight, colstraight)), (patialprun, (rowleft, colleft)),
                    (patialprun, (state0, state1))]

        elif straight and not left and not right:
            return [(prun, (rowstraight, colstraight)), ((1 - prun), (state0, state1))]

        elif not straight and left and not right:
            return [(patialprun, (rowleft, colleft)), (1 - patialprun, (state0, state1))]

        elif not straight and not left and right:
            return [(patialprun, (rowright, colright)), (1 - patialprun, (state0, state1))]

        else:
            return [(1, (state0, state1))]

    def T(self, state, action):
        return self.transitions[state][action]

    def R(self, action):
        if self.actlist.index(action) < 4:
            return self.rwalk
        else:
            return self.rrun


class SolverAgent:

    def __init__(self):
        self.row = 0
        self.column = 0
        self.wallCellNum = 0
        self.wallCellList = []
        self.terminalNum = 0
        self.terminalCellList = []
        self.terminalUtilityList = []
        self.pwalk = 0
        self.rrun = 0
        self.rwalk = 0
        self.rrun = 0
        self.gamma = 1.0

    def readFile(self):
        lineNum = 1
        for line in open('input2.txt'):
            line = line.strip()

            if lineNum == 1:
                self.row = int(line.split(',')[0])
                self.column = int(line.split(',')[1])
            elif lineNum == 2:
                self.wallCellNum = int(line)
            elif 2 < lineNum < 3 + self.wallCellNum:
                tmp = line.split(',')
                self.wallCellList.append((self.row - int(tmp[0]), int(tmp[1]) - 1))

            elif lineNum == 3 + self.wallCellNum:
                self.terminalNum = int(line)
            elif 3 + self.wallCellNum < lineNum < 4 + self.wallCellNum + self.terminalNum:
                tmp = line.split(',')
                self.terminalCellList.append((self.row - int(tmp[0]), int(tmp[1]) - 1))
                self.terminalUtilityList.append(float(tmp[2]))

            elif lineNum == 4 + self.wallCellNum + self.terminalNum:
                self.pwalk = float(line.split(',')[0])
                self.prun = float(line.split(',')[1])
            elif lineNum == 4 + self.wallCellNum + self.terminalNum + 1:
                self.rwalk = float(line.split(',')[0])
                self.rrun = float(line.split(',')[1])
            elif lineNum == 4 + self.wallCellNum + self.terminalNum + 2:
                self.gamma = float(line)

            lineNum += 1

    def initBoard(self):
        board = [['X' for _ in range(self.column)] for _ in range(self.row)]
        for cell in self.wallCellList:
            board[cell[0]][cell[1]] = None

        for i in range(len(self.terminalCellList)):
            cell = self.terminalCellList[i]
            board[cell[0]][cell[1]] = self.terminalUtilityList[i]

        return board

    def to_grid(self, mapping):
            return list(([[mapping.get((x, y), None)
                                   for y in range(self.column)]
                                  for x in range(self.row)]))

    def to_arrows(self, policy):
        chars = {(-1, 0): 'Walk Up', (0, 1): 'Walk Right', (1, 0): 'Walk Down', (0, -1): 'Walk Left',
                 (-2, 0): 'Run Up', (0, 2): 'Run Right', (2, 0): 'Run Down', (0, -2): 'Run Left', 'Exit': 'Exit'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

    def print_table(self, table):
        outputfile = open('output.txt', 'w')
        for i in range(len(table)):
            outputfile.write(','.join(map(str, table[i])))
            outputfile.write('\n')
        outputfile.close()

#----------------------------------------------------------------
def positionaviliable(x, y, mdp, visitedmap):
    if 0 <= x < mdp.rows and 0 <= y < mdp.cols and not visitedmap[x][y] and (x, y) in mdp.states:
        return True
    else:
        return False


def value_iteration(mdp, terminalUList, epsilon=0.0000001):
        U1 = {s: 0 for s in mdp.states}
        R, T, gamma = mdp.R, mdp.T, mdp.gamma
        while True:
            U = U1.copy()
            delta = 0
            for s in mdp.states:
                if s in mdp.terminals:
                    U1[s] = terminalUList[mdp.terminals.index(s)]
                else:
                    U1[s] = max(R(a) + gamma * (sum(p * U[s1] for (p, s1) in T(s, a))) for a in mdp.actions())

                delta = max(delta, abs(U1[s] - U[s]))
            if delta < epsilon * (1 - gamma) / gamma:
                return U


def value_iteration_ver1(mdp, terminalUList, approximateU, epsilon=0.0000001):
    U1 = approximateU
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    maxdistance = epsilon * (1 - gamma) / gamma
    terminalset = set(mdp.terminals)
    actlist = mdp.actions()

    for s in terminalset:
        U1[s] = terminalUList[mdp.terminals.index(s)]

    while True:
        U = U1.copy()
        delta = 0

        for s in mdp.states - terminalset:
            U1[s] = max(R(a) + gamma * (sum(p * U[s1] for (p, s1) in T(s, a))) for a in actlist)
            delta = max(delta, abs(U1[s] - U[s]))

        if delta < maxdistance or time.time() - starttime > 165:
            return U

def value_iteration_ver2(mdp, terminalUList, epsilon=0.0000001):
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma

    actlist = mdp.actions()
    while True:

        U = U1.copy()
        delta = 0
        bfsqueue = []
        visitedmap = [[0 for _ in range(mdp.cols)] for _ in range(mdp.rows)]

        for s in mdp.terminals:
            bfsqueue.append(s)
            U1[s] = terminalUList[mdp.terminals.index(s)]
            visitedmap[s[0]][s[1]] = 1

        while len(bfsqueue):
            cell = bfsqueue.pop(0)
            cellx = cell[0]
            celly = cell[1]

            if positionaviliable(cellx - 1, celly, mdp, visitedmap):
                s = (cellx - 1, celly)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx - 1][celly] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx + 1, celly, mdp, visitedmap):
                s = (cellx + 1, celly)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx + 1][celly] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx, celly - 1, mdp, visitedmap):
                s = (cellx, celly - 1)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx][celly - 1] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx, celly + 1, mdp, visitedmap):
                s = (cellx, celly + 1)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx][celly + 1] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx - 2, celly, mdp, visitedmap):
                s = (cellx - 2, celly)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx - 2][celly] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx + 2, celly, mdp, visitedmap):
                s = (cellx + 2, celly)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx + 2][celly] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx, celly - 2, mdp, visitedmap):
                s = (cellx, celly - 2)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx][celly - 2] = 1
                delta = max(delta, abs(U1[s] - U[s]))

            if positionaviliable(cellx, celly + 2, mdp, visitedmap):
                s = (cellx, celly + 2)
                bfsqueue.append(s)
                U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                visitedmap[cellx][celly + 2] = 1
                delta = max(delta, abs(U1[s] - U[s]))

        if delta < epsilon * (1 - gamma) / gamma:
            return U



def generateU_usingBFS(mdp, terminalUList):
        U1 = {s: 0 for s in mdp.states}
        R, T, gamma = mdp.R, mdp.T, mdp.gamma
        actlist = mdp.actions()

        for i in range(1):
            bfsqueue = []
            visitedmap = [[0 for _ in range(mdp.cols)] for _ in range(mdp.rows)]


            for s in mdp.terminals:
                bfsqueue.append(s)
                U1[s] = terminalUList[mdp.terminals.index(s)]
                visitedmap[s[0]][s[1]] = 1

            while len(bfsqueue):
                cell = bfsqueue.pop(0)
                cellx = cell[0]
                celly = cell[1]

                if positionaviliable(cellx - 1, celly, mdp, visitedmap):
                    s = (cellx - 1, celly)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx - 1][celly] = 1

                if positionaviliable(cellx + 1, celly, mdp, visitedmap):
                    s = (cellx + 1, celly)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx + 1][celly] = 1

                if positionaviliable(cellx, celly - 1, mdp, visitedmap):
                    s = (cellx, celly - 1)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx][celly - 1] = 1

                if positionaviliable(cellx, celly + 1, mdp, visitedmap):
                    s = (cellx, celly + 1)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx][celly + 1] = 1

                if positionaviliable(cellx - 2, celly, mdp, visitedmap):
                    s = (cellx - 2, celly)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx - 2][celly] = 1

                if positionaviliable(cellx + 2, celly, mdp, visitedmap):
                    s = (cellx + 2, celly)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx + 2][celly] = 1

                if positionaviliable(cellx, celly - 2, mdp, visitedmap):
                    s = (cellx, celly - 2)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx][celly - 2] = 1

                if positionaviliable(cellx, celly + 2, mdp, visitedmap):
                    s = (cellx, celly + 2)
                    bfsqueue.append(s)
                    U1[s] = max(R(a) + gamma * (sum(p * U1[s1] for (p, s1) in T(s, a))) for a in actlist)
                    visitedmap[cellx][celly + 2] = 1

        return U1


def best_policy(mdp, U):
        pi = {}
        terminalset = set(mdp.terminals)

        for s in mdp.terminals:
            pi[s] = 'Exit'

        for s in mdp.states - terminalset:
            pi[s] = max(mdp.actions(), key=lambda a: expected_utility(a, s, U, mdp))
        return pi

def expected_utility(a, s, U, mdp):
        return mdp.R(a) + mdp.gamma * sum(p * U[s1] for (p, s1) in mdp.T(s, a))

#-----------------------------------------------------------

if __name__ == '__main__':
    starttime = time.time()
    agent = SolverAgent()
    agent.readFile()
    board = agent.initBoard()

    mdp = GridMDP(board, agent.terminalCellList, agent.pwalk, agent.prun, agent.rwalk, agent.rrun,
                  (agent.row - 1, 0), agent.gamma)

    """optimization"""
    approximateU = generateU_usingBFS(mdp, agent.terminalUtilityList)
    U = value_iteration_ver1(mdp, agent.terminalUtilityList, approximateU)
    policy = best_policy(mdp, U)

    res = agent.to_arrows(policy)
    agent.print_table(res)
