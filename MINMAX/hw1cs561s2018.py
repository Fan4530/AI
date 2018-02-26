
import sys



def valid(board, i, j):
    return i >= 0 and j >= 0 and i < len(board) and j < len(board[0])

# change the format
def type_and_number(board, i, j):
    if board[i][j] == '0':
        return '0', 0
    else:
        return board[i][j][0], int(board[i][j][1: len(board[i][j])])


# move from (oi, oj) -> (i, j), and remove opposite if applicable
def move(board, oi, oj, i, j):
    type, n = type_and_number(board, i, j);
    type_o, n_o = type_and_number(board, oi, oj)
    n_o = n_o - 1
    n = n + 1
    type = type_o
    if n_o == 0:
        type_o = '0'
    if type_o == '0':
        board[oi][oj] = '0'
    else:
        board[oi][oj] = type_o + str(n_o)

    board[i][j] = type + str(n)
    #remove the opposite one
    if abs(oi - i) == 2:
        board[(oi + i) / 2][(oj + j) / 2] = '0';

#maxvalue determine the sign
#star or circle will influence the value number
def get_val(board, positive, row_val):
    sum = 0;
    for i in range(0, len(board)):
        for j in range(0, len(board[i])):
            type, number = type_and_number(board, i, j);
            if type == 'S':
                sum = sum + number * int(row_val[7 - i])
            else:
                sum = sum - number * int(row_val[i])
    if positive:
        return sum
    else:
        return -sum


# change i, j to path
hash = "HGFEDCBA"
def create_path(i1, j1, i2, j2):
    global hash;
    return hash[i1] + str(j1 + 1) + "-" + hash[i2] + str(j2 + 1)




# one_path = 2, terminate
one_path = 0
# node number
count = 0
not_prune = False;
def dfs_alphabet(depth, board, player, maxvalue, row_val, alpha, beta):

    # for list in board:
    #     print list
    # print "\n"
    # print count
    # print alpha
    # print beta

    global one_path
    global count
    count = count + 1
    # base case
    if depth == 0 or one_path == 2:#terminated
        # calculate the val
        this_val = get_val(board, not (maxvalue ^ (player == 'Star')), row_val)
        return this_val, "pass"## pass??

    #alpha beta prutning



    best_value = -sys.maxsize - 1;
    best_value = sys.maxsize;
    path = ""
    cur_stats = 0;
    # cur_stats = 0 : terminate, doesn't exist an player
    # cur_stats = 1 : has at least one player, but none of them are valid, requtre pass
    # cur_stats = 2: at least one player is valid
    not_prune = True
    if maxvalue:
        best_value = alpha
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):

                if board[i][j][0] == player[0]:
                    if cur_stats != 2:
                        cur_stats = 1
                    # two players have different path
                    #star
                    if player == "Star":
                        # left direction: one unit
                        if valid(board, i - 1, j - 1) and not_prune and not_prune and (board[i - 1][j - 1] == '0' or (board[i - 1][j - 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j - 1);7
                            #for maxvalue:
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, best_value, beta)#the lower bound is at least max_val, init is MIN_VALUE
                            move(board, i - 1, j - 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j - 1)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
                        # left direction: two units
                        elif valid(board, i - 2, j - 2) and not_prune and board[i - 1][j - 1][0] == 'C' and ((board[i - 2][j - 2][0] == 'S' and i == 2) or board[i - 2][j - 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j - 1]
                            move(board, i, j, i - 2, j - 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, best_value, beta)
                            move(board, i - 2, j - 2, i , j)
                            board[i - 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j - 2)
                            if  best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
                        #right direction
                        if valid(board, i - 1, j + 1) and not_prune and (board[i - 1][j + 1] == '0' or (board[i - 1][j + 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j + 1);
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, best_value, beta)
                            move(board, i - 1, j + 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j + 1)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
                        # right direction: two units
                        elif valid(board, i - 2, j + 2) and not_prune and board[i - 1][j + 1][0] == 'C' and ((board[i - 2][j + 2][0] == 'S' and i == 2) or board[i - 2][j + 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j + 1]
                            move(board, i, j, i - 2, j + 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, best_value, beta)
                            move(board, i - 2, j + 2, i , j)
                            board[i - 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j + 2)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                                alpha = best_value
                            if best_value >= beta:
                                not_prune = False
                    #Circle
                    else:
                        passThis = 'true';
                        # left direction: one unit
                        if valid(board, i + 1, j - 1) and not_prune and (board[i + 1][j - 1] == '0' or (board[i + 1][j - 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j - 1);
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, best_value, beta)
                            move(board, i + 1, j - 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j - 1)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
                        # left direction: two units
                        elif valid(board, i + 2, j - 2) and not_prune and board[i + 1][j - 1][0] == 'S' and ((board[i + 2][j - 2][0] == 'C' and i == 5) or board[i + 2][j - 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j - 1]
                            move(board, i, j, i + 2, j - 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, best_value, beta)
                            move(board, i + 2, j - 2, i, j)
                            board[i + 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j - 2)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
                        # right direction
                        if valid(board, i + 1, j + 1) and not_prune and (board[i + 1][j + 1] == '0' or (board[i + 1][j + 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j + 1);
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, best_value, beta)
                            move(board, i + 1, j + 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j + 1)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
                        # right direction: two units
                        elif valid(board, i + 2, j + 2) and not_prune and board[i + 1][j + 1][0] == 'S' and ((board[i + 2][j + 2][0] == 'C' and i == 5) or board[i + 2][j + 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j + 1]
                            move(board, i, j, i + 2, j + 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, best_value, beta)
                            move(board, i + 2, j + 2, i, j)
                            board[i + 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j + 2)
                            if best_value < last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value >= beta:
                                not_prune = False
    else:
        best_value = beta
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):

                if board[i][j][0] == player[0]:
                    if cur_stats != 2:
                        cur_stats = 1
                    # two players have different path
                    #star
                    if player == "Star":
                        # left direction: one unit
                        if valid(board, i - 1, j - 1) and not_prune and (board[i - 1][j - 1] == '0' or (board[i - 1][j - 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j - 1);
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, alpha, best_value)
                            move(board, i - 1, j - 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j - 1)
                            if best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False
                        # left direction: two units
                        elif valid(board, i - 2, j - 2) and not_prune and board[i - 1][j - 1][0] == 'C' and ((board[i - 2][j - 2][0] == 'S' and i == 2) or board[i - 2][j - 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j - 1]
                            move(board, i, j, i - 2, j - 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, alpha, best_value)
                            move(board, i - 2, j - 2, i , j)
                            board[i - 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j - 2)
                            if best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False
                        #right direction
                        if valid(board, i - 1, j + 1) and not_prune and (board[i - 1][j + 1] == '0' or (board[i - 1][j + 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j + 1);
                            last_val, res =  dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, alpha, best_value)
                            move(board, i - 1, j + 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j + 1)
                            if best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False
                        # right direction: two units
                        elif valid(board, i - 2, j + 2) and not_prune and board[i - 1][j + 1][0] == 'C' and ((board[i - 2][j + 2][0] == 'S' and i == 2) or board[i - 2][j + 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j + 1]
                            move(board, i, j, i - 2, j + 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, alpha, best_value)
                            move(board, i - 2, j + 2, i , j)
                            board[i - 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j + 2)
                            if best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False
                    #Circle
                    else:
                        # left direction: one unit
                        if valid(board, i + 1, j - 1) and not_prune and (board[i + 1][j - 1] == '0' or (board[i + 1][j - 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j - 1);
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, alpha, best_value)
                            move(board, i + 1, j - 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j - 1)
                            if best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False

                        # left direction: two units
                        elif valid(board, i + 2, j - 2) and not_prune and board[i + 1][j - 1][0] == 'S' and ((board[i + 2][j - 2][0] == 'C' and i == 5) or board[i + 2][j - 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j - 1]
                            move(board, i, j, i + 2, j - 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, alpha, best_value)
                            move(board, i + 2, j - 2, i, j)
                            board[i + 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j - 2)
                            if best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False
                        # right direction
                        if valid(board, i + 1, j + 1) and not_prune and (board[i + 1][j + 1] == '0' or (board[i + 1][j + 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j + 1);
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, alpha, best_value)
                            move(board, i + 1, j + 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j + 1)
                            if  best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False
                        # right direction: two units
                        elif valid(board, i + 2, j + 2) and not_prune and board[i + 1][j + 1][0] == 'S' and ((board[i + 2][j + 2][0] == 'C' and i == 5) or board[i + 2][j + 2] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j + 1]
                            move(board, i, j, i + 2, j + 2)
                            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, alpha, best_value)
                            move(board, i + 2, j + 2, i, j)
                            board[i + 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j + 2)
                            if  best_value > last_val:
                                best_value = last_val
                                path = path_tmp
                            if best_value <= alpha:
                                not_prune = False

    if cur_stats == 0: #terminate
        return get_val(board, not (maxvalue ^ (player == 'Star')), row_val), "pass"
    elif cur_stats == 1: # two cases
        one_path = one_path + 1
        if player == "Circle": #case 2: the first pass, continue dfs
            last_val, res = dfs_alphabet(depth - 1, board, "Star", not maxvalue, row_val, alpha, beta)
        else:
            last_val, res = dfs_alphabet(depth - 1, board, "Circle", not maxvalue, row_val, alpha, beta)
        path = "pass"
        best_value = last_val

    #res_val: last value, final result
    #last_val: this value, used for choose path
    return best_value, path
def dfs(depth, board, player, maxvalue, row_val):

    # for list in board:
    #     print list
    # print "\n"
    global one_path
    global count

    count = count + 1
    # base case
    if depth == 0 or one_path == 2:#terminated
        # calculate the val
        this_val = get_val(board, not (maxvalue ^ (player == 'Star')), row_val)
        return this_val, "pass"## pass??



    max_val = -sys.maxsize - 1;
    min_val = sys.maxsize;
    path = ""
    cur_stats = 0;
    # cur_stats = 0 : terminate, doesn't exist an player
    # cur_stats = 1 : has at least one player, but none of them are valid, requtre pass
    # cur_stats = 2: at least one player is valid

    for i in range(0, len(board)):
        for j in range(0, len(board[i])):
            if board[i][j][0] == player[0]:
                if cur_stats != 2:
                    cur_stats = 1
                if maxvalue:
                    # two players have different path
                    #star
                    if player == "Star":
                        # left direction: one unit
                        if valid(board, i - 1, j - 1) and (board[i - 1][j - 1] == '0' or (board[i - 1][j - 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j - 1);
                            #for maxvalue:
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 1, j - 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j - 1)
                            if  max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                        # left direction: two units
                        elif valid(board, i - 2, j - 2) and board[i - 1][j - 1][0] == 'C' and ((board[i - 2][j - 2][0] == 'S' and i == 2) or board[i - 2][j - 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j - 1]
                            move(board, i, j, i - 2, j - 2)
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 2, j - 2, i , j)
                            board[i - 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j - 2)
                            if  max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                        #right direction
                        if valid(board, i - 1, j + 1) and (board[i - 1][j + 1] == '0' or (board[i - 1][j + 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j + 1);
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 1, j + 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j + 1)
                            if max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                        # right direction: two units
                        elif valid(board, i - 2, j + 2) and board[i - 1][j + 1][0] == 'C' and ((board[i - 2][j + 2][0] == 'S' and i == 2) or board[i - 2][j + 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j + 1]
                            move(board, i, j, i - 2, j + 2)
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 2, j + 2, i , j)
                            board[i - 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j + 2)
                            if max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                                alpha = max_val
                    #Circle
                    else:
                        passThis = 'true';
                        # left direction: one unit
                        if valid(board, i + 1, j - 1) and (board[i + 1][j - 1] == '0' or (board[i + 1][j - 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j - 1);
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 1, j - 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j - 1)
                            if max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                        # left direction: two units
                        elif valid(board, i + 2, j - 2) and board[i + 1][j - 1][0] == 'S' and ((board[i + 2][j - 2][0] == 'C' and i == 5) or board[i + 2][j - 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j - 1]
                            move(board, i, j, i + 2, j - 2)
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 2, j - 2, i, j)
                            board[i + 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j - 2)
                            if max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                        # right direction
                        if valid(board, i + 1, j + 1) and (board[i + 1][j + 1] == '0' or (board[i + 1][j + 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j + 1);
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 1, j + 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j + 1)
                            if max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                        # right direction: two units
                        elif valid(board, i + 2, j + 2) and board[i + 1][j + 1][0] == 'S' and ((board[i + 2][j + 2][0] == 'C' and i == 5) or board[i + 2][j + 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j + 1]
                            move(board, i, j, i + 2, j + 2)
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 2, j + 2, i, j)
                            board[i + 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j + 2)
                            if max_val < last_val:
                                max_val = last_val
                                path = path_tmp
                else:
                    # two players have different path
                    #star
                    if player == "Star":
                        # left direction: one unit
                        if valid(board, i - 1, j - 1) and (board[i - 1][j - 1] == '0' or (board[i - 1][j - 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j - 1);
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 1, j - 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j - 1)
                            if min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                        # left direction: two units
                        elif valid(board, i - 2, j - 2) and board[i - 1][j - 1][0] == 'C' and ((board[i - 2][j - 2][0] == 'S' and i == 2) or board[i - 2][j - 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j - 1]
                            move(board, i, j, i - 2, j - 2)
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 2, j - 2, i , j)
                            board[i - 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j - 2)
                            if min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                        #right direction
                        if valid(board, i - 1, j + 1) and (board[i - 1][j + 1] == '0' or (board[i - 1][j + 1][0] == player[0] and i == 1)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i - 1, j + 1);
                            last_val, res =  dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 1, j + 1, i , j)
                            path_tmp = create_path(i, j, i - 1, j + 1)
                            if min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                        # right direction: two units
                        elif valid(board, i - 2, j + 2) and board[i - 1][j + 1][0] == 'C' and ((board[i - 2][j + 2][0] == 'S' and i == 2) or board[i - 2][j + 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i - 1][j + 1]
                            move(board, i, j, i - 2, j + 2)
                            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
                            move(board, i - 2, j + 2, i , j)
                            board[i - 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i - 2, j + 2)
                            if min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                    #Circle
                    else:
                        passThis = 'true';
                        # left direction: one unit
                        if valid(board, i + 1, j - 1) and (board[i + 1][j - 1] == '0' or (board[i + 1][j - 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j - 1);
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 1, j - 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j - 1)
                            if min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                        # left direction: two units
                        elif valid(board, i + 2, j - 2) and board[i + 1][j - 1][0] == 'S' and ((board[i + 2][j - 2][0] == 'C' and i == 5) or board[i + 2][j - 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j - 1]
                            move(board, i, j, i + 2, j - 2)
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 2, j - 2, i, j)
                            board[i + 1][j - 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j - 2)
                            if min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                        # right direction
                        if valid(board, i + 1, j + 1) and (board[i + 1][j + 1] == '0' or (board[i + 1][j + 1][0] == player[0] and i == 6)):
                            cur_stats = 2
                            one_path = 0;
                            move(board, i, j, i + 1, j + 1);
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 1, j + 1, i, j)
                            path_tmp = create_path(i, j, i + 1, j + 1)
                            if  min_val > last_val:
                                min_val = last_val
                                path = path_tmp
                        # right direction: two units
                        elif valid(board, i + 2, j + 2) and board[i + 1][j + 1][0] == 'S' and ((board[i + 2][j + 2][0] == 'C' and i == 5) or board[i + 2][j + 2][0] == '0'):
                            cur_stats = 2
                            one_path = 0;
                            tmp = board[i + 1][j + 1]
                            move(board, i, j, i + 2, j + 2)
                            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
                            move(board, i + 2, j + 2, i, j)
                            board[i + 1][j + 1] = tmp;
                            path_tmp = create_path(i, j, i + 2, j + 2)
                            if  min_val > last_val:
                                min_val = last_val
                                path = path_tmp

    if cur_stats == 0: #terminate
        return get_val(board, not (maxvalue ^ (player == 'Star')), row_val), "pass"
    elif cur_stats == 1: # two cases
        one_path = one_path + 1
        if player == "Circle": #case 2: the first pass, continue dfs
            last_val, res = dfs(depth - 1, board, "Star", not maxvalue, row_val)
        else:
            last_val, res = dfs(depth - 1, board, "Circle", not maxvalue, row_val)
        path = "pass"
        max_val = last_val
        min_val = last_val

    #res_val: last value, final result
    #last_val: this value, used for choose path
    if maxvalue is True:
        return max_val, path
    else:
        return min_val, path


def run_algorithm(algorithm, depth, player, row_value, board):
    if algorithm == "MINIMAX":
        return dfs(depth, board, player, True, row_value)#""????
    else:
        return dfs_alphabet(depth, board, player, True, row_value, - sys.maxsize - 1, sys.maxsize)#""????


def get_instant_utility(board, next_move, row_value, player):
    if next_move == "pass":
        return get_val(board, player == "Star", row_value)
    oi = 7 - ord(next_move[0]) + ord('A')
    oj = ord(next_move[1]) - ord('1')
    i = 7 - ord(next_move[3]) + ord('A')
    j = ord(next_move[4]) - ord('1')
    move(board, oi, oj, i, j)
    return get_val(board, player == "Star", row_value)


def main(filename, output):
    #inport the file
    fo = open(filename)
    input = [line.strip("\n") for line in fo.readlines()]
    fo.close()
    # transfer format
    player = input[0]
    algorithm = input[1]
    depth = int(input[2])
    board = [line.split(",") for line in input[3: len(input) - 1]]


    row_value = input[len(input) - 1].split(",")
    #next move:
    FARSIGHTED_UTILITY, NEXT_MOVE = run_algorithm(algorithm, depth, player, row_value, board)
    global count
    global one_path
    NUMBER_OF_NODES = count;
    count = 0
    one_path = 0;

    fo = open(output, 'w')
    fo.write(NEXT_MOVE)
    fo.write("\n")

    fo.write(str(get_instant_utility(board, NEXT_MOVE, row_value, player)))
    fo.write("\n")
    fo.write(str(FARSIGHTED_UTILITY))
    fo.write("\n")
    fo.write(str(NUMBER_OF_NODES))
    fo.close()
    # print NEXT_MOVE
    # print FARSIGHTED_UTILITY
    # print get_instant_utility(board, NEXT_MOVE, row_value, player)
    # print NUMBER_OF_NODES

main("input.txt", "output.txt")

