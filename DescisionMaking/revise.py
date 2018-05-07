from numpy import *

from time import time;


#This is the reversed priority order, cannot change
actlist_walk = WALK_RIGHT, WALK_LEFT, WALK_DOWN, WALK_UP = [(0, 1), (0, -1), (1, 0), (1, 0)]
actlist_run = RUN_RIGHT, RUN_LEFT, RUN_DOWN, RUN_UP = [(0, 2), (0, -2), (2, 0), (2, 0)]


def best_policy(a, b, gamma, r1, r2, U, rows, cols, p1, p1_, p2, p2_):
    pi = {}


    #2 0 1 3
    # iter_order = [[3, 2, 0], [2, 3, 1], [1, 2, 0], [0, 1, 3]]

    # transition order: [up right down left] -> priority order: [right left down up]
    iter_order = [[1, 0, 2], [3, 2, 0], [2, 1, 3], [0, 1, 3]] #The index is the transition
    for i in range(rows):
        for j in range(cols):
            if (i, j) in walls:
                pi[(i, j)] = None
            elif (i, j) in terminals:
                pi[(i, j)] = (0, 0)
            else:
                max_state = float(-inf)
                res = (0, 0)
                for k in range(4):
                    #transitions
                    trans = b[i][j]#[up right down left]

                    iter = iter_order[k]

                    x_ahead, y_ahead = trans[iter[0]]
                    x1, y1 = trans[iter[1]]

                    x2, y2 = trans[iter[2]]

                    sum_dir = r2[i][j] + gamma * ((U[x1][y1] + U[x2][y2]) * p2_ + U[x_ahead][y_ahead] * p2)
                    if max_state <= sum_dir:
                        max_state = sum_dir
                        res = actlist_run[k]# priority order
                for k in range(4):
                    trans = a[i][j]
                    iter = iter_order[k]
                    x_ahead, y_ahead = trans[iter[0]]
                    x1, y1 = trans[iter[1]]

                    x2, y2 = trans[iter[2]]

                    sum_dir = r1[i][j] + gamma * ((U[x1][y1] + U[x2][y2]) * p1_ + U[x_ahead][y_ahead] * p1)
                    if max_state <= sum_dir:
                        max_state = sum_dir
                        res = actlist_walk[k]
                pi[(i, j)] = res
    return pi


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

    return walls, terminals, rows, cols, gamma, p_walk, p_run, r


def test(answer, my_answer):
    row = len(answer)
    col = len(answer[0])
    if row != len(my_answer) or col != len(my_answer[0]):
        print "Size not equal"
        return

    fail = False
    for i in range(0, row):
        for j in range(0, col):
            flag = answer[i][j] == my_answer[i][j],
            print flag,
            if not flag[0]:
                fail = True
        print
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
def test_2(my_output, output):
    my_output = read_output(my_output);
    output = read_output(output)
    test(output, my_output)

def transitions(step, rows, cols, walls):
    a = [[[] for i in range(cols)] for j in range(rows)]

    for i in range(rows):
        for j in range(cols):
            # up   right    down   left
            a[i][j] = [[-step + i, 0 + j], [0 + i, step + j], [step + i, 0 + j], [0 + i, -step + j]]


            if i <= step - 1 or (i - step, j) in walls or (i - step + 1, j) in walls:
                a[i][j][0][0] = a[i][j][0][0] + step
            if j >= cols - step or (i, j + step) in walls or (i, j + step - 1) in walls:
                a[i][j][1][1] = a[i][j][1][1] - step


            if i >= rows - step or (i + step, j) in walls or (i + step - 1, j) in walls:
                a[i][j][2][0] = a[i][j][2][0] - step

            if j <= step - 1 or (i, j - step) in walls or (i, j - step + 1) in walls:
                a[i][j][3][1] = a[i][j][3][1] + step

    return a




def read_output(filename):
    fo = open(filename)
    output = [line.strip("\n") for line in fo.readlines()]
    fo.close()
    output1 = []
    for line in output:
        output1.append(line.split(','))
    return output1



def iteration(U, gamma, p, p_, p2, p2_):
    U1 = U.copy()
    delta = 0
    for i in range(rows):
        for j in range(cols):

            if (i, j) not in walls and (i, j) not in terminals:
                max_state = float("-inf")

                for k in range(4):
                    list = a[i][j]
                    x1, y1 = list[k]
                    x2, y2 = list[(k + 1) % 4]
                    x3, y3 = list[(k + 2) % 4]
                    # print x1, y1, x2, y2, x3, y3
                    sum_dir = r1[i][j] + gamma * ((U[x1][y1] + U[x3][y3]) * p_ + U[x2][y2] * p)
                    max_state = max(sum_dir, max_state)
                for k in range(4):
                    list = b[i][j]
                    x1, y1 = list[k]
                    x2, y2 = list[(k + 1) % 4]
                    x3, y3 = list[(k + 2) % 4]
                    # print x1, y1, x2, y2, x3, y3
                    sum_dir = r2[i][j] + gamma * ((U[x1][y1] + U[x3][y3]) * p2_ + U[x2][y2] * p2)
                    max_state = max(sum_dir, max_state)
                U1[i][j] = max_state

            elif (i, j) in terminals:
                # U1[2][4] = 5
                # U1[4][2] = 10
                for ele in terminals:
                    (x, y) = ele
                    U1[x][y] = terminals[ele]
            delta = max(delta, abs(U1[i][j] - U[i][j]))


    return U1, delta


def iteration_main(p1, p1_, p2, p2_, epsilon = 0.000001):
    U = zeros(rows * cols).reshape(rows, cols)
    print("Start to iteration" + str(time() - start) + "s")
    count = 0
    while True:
    # for i in range(22):
        print("Start to iteration:" + str(count))
        print("cost"+ str(time() - start) + "s")
        count = count + 1

        U, delta = iteration(U, gamma, p1, p1_, p2, p2_)
        # print U
        if delta < epsilon * (1 - gamma) / gamma:
            break;
    return U


def to_words(policy, rows, cols):
    chars = {(0, 1): 'Walk Right', (-1, 0): 'Walk Up', (0, -1): 'Walk Left', (1, 0): 'Walk Down', (0, 2): 'Run Right',
             (-2, 0): 'Run Up', (0, -2): 'Run Left', (2, 0): 'Run Down', None: 'None', (0, 0):'Exit'}
    # return to_grid({s: chars[a] for (s, a) in policy.items()})
    # return to_grid({s: chars[a] for (s, a) in policy.items()})
    list = [[0 for i in range(cols)] for j in range(rows)]
    for k in policy.keys():
        i = k[0]
        j = k[1]
        # print policy[k]
        list[i][j] = chars[policy[k]]
    return list

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


def equation(node):
    i, j = node[0], node[1]

    if (i, j) not in walls or (i, j) not in terminals:
        max_state = float("-inf")

        for k in range(4):
            list = a[i][j]
            x1, y1 = list[k]
            x2, y2 = list[(k + 1) % 4]
            x3, y3 = list[(k + 2) % 4]
            # print x1, y1, x2, y2, x3, y3
            sum_dir = r1[i][j] + gamma * ((U[x1][y1] + U[x3][y3]) * p_ + U[x2][y2] * p)
            max_state = max(sum_dir, max_state)
        for k in range(4):
            list = b[i][j]
            x1, y1 = list[k]
            x2, y2 = list[(k + 1) % 4]
            x3, y3 = list[(k + 2) % 4]
            # print x1, y1, x2, y2, x3, y3
            sum_dir = r2[i][j] + gamma * ((U[x1][y1] + U[x3][y3]) * p2_ + U[x2][y2] * p2)
            max_state = max(sum_dir, max_state)

        U1[i][j] = max_state
        if (i, j) in terminals:
            # U1[2][4] = 5
            # U1[4][2] = 10
            for ele in terminals:
                (x, y) = ele
                U1[x][y] = terminals[ele]
        delta = max(delta, abs(U1[i][j] - U[i][j]))

def preprocessing_BFS(U, queue, mdp1, mdp2, visited):

    U1 = U.copy()
    rows = len(mdp1.grid)
    cols = len(mdp1.grid[0])
    delta = 0;
    count = 0

    while not queue.empty():
        node = queue.get()
        count = count + 1

        # print node
        new_delta = equation(node, )
        # print node, U1[node]
        delta = max(delta, new_delta)

        for dir in orientations:
            new_node = (node[0] + dir[0], node[1] + dir[1])
            if valid(new_node, rows, cols, mdp1) and new_node not in visited:
                queue.put(new_node)
                visited.add(new_node)

    return U1, delta
import Queue

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
            #TODO:
        values, delta = preprocessing_BFS(values, queue, input_run, input_walk, visited)
        if delta < epsilon * (1 - gamma) / gamma:
            break;
    return values



start = time()

walls, terminals, rows, cols, gamma, p_walk, p_run, r = read_and_process_input("input2.txt")
p1 = p_walk
p1_ = (1 - p_walk) / 2
p2 = p_run
p2_ = (1 - p_run) / 2

a = transitions(1, rows, cols, walls)
b = transitions(2, rows, cols, walls)

reward_1 = float(r[0])
reward_2 = float(r[1])
r1 = zeros(rows * cols).reshape(rows, cols) + reward_1
r2 = zeros(rows * cols).reshape(rows, cols) + reward_2

for ele in terminals:
    (x, y) = ele
    r1[x][y] = r2[x][y] = terminals[ele]

values = iteration_main(p1, p1_, p2, p2_, 0)

print("start to best_policy: " + str(time() - start) + "s")

pi = best_policy(a, b, gamma, r1, r2, values, rows, cols, p1, p1_, p2, p2_)

my_answer = to_words(pi, rows, cols)
my_answer = res_reverse(my_answer)



# write my_answer to the file
write_file(my_answer)
print("end: " + str(time() - start) + "s")

test_2("output.txt", "output2.txt")