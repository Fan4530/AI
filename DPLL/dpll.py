import random
import copy
from time import time

class X(object):
    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    def __invert__(self):   return X('~', self)

    def __and__(self, rhs): return X('&', self, rhs)

    def __or__(self, rhs):
        if isinstance(rhs, X):
            return X('|', self, rhs)

    def __eq__(self, other):
        return (isinstance(other, X)
                and self.op == other.op
                and self.args == other.args)

    def __hash__(self):
        return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if len(args) == 0:
            return op
        elif len(args) == 1:  # -x or -(x + 1)
            return op + args[0]
        else:  # (x - y)
            opp = op
            return '(' + opp.join(args) + ')'


class Agent:
    pots = []
    UEFA = 0
    pots_set = []
    normal_area = []
    group_number = 0
    clauses = set();
    first_pot = {}


    def get_area(self, input):
        normal = []
        UEFA = []
        for line in input:
            if len(line) > 1:
                if line[0] == 'UEFA':
                    UEFA = line[1:len(line)]
                elif line[1] is not 'None':
                    normal.append(line[1:len(line)])
        return normal, UEFA

    def get_file(self, filename):
        # inport the file
        #filename = "input1.txt"
        fo = open(filename)
        input = [line.strip("\r\n") for line in fo.readlines()]
        fo.close()
        # transfer format
        self.group_number = int(input[0])
        pot_number = int(input[1])

        self.pots = [line.split(",") for line in input[2:pot_number + 2]]
        for line in self.pots:
            line_set = set()
            for ele in line:
                line_set.add(ele)
            self.pots_set.append(line_set)

        confederations = [line.replace(':', ',').split(",") for line in input[pot_number + 2: len(input)]]
        self.normal, self.UEFA = self.get_area(confederations)
        for i in range(0, len(self.pots[0])):
             self.first_pot[self.pots[0][i]] = i

    def __init__(self):
        self.pots_set = []
        self.pots = []
        self.UEFA = 0
        self.normal_area = []
        self.group_number = 0
        self.clauses = set()

    def one_country_at_least_one_group(self, group_number, pots):
        for array in pots:
            for member in array:
                s = X(member + str(0))
                for i in range(1, group_number):
                    s = s | X(member + str(i))
                self.clauses.add(s)
        return self.clauses
    #optimize
    def optimize(self):
        pot = self.pots[0];
        for i in range(0, len(pot)):
            a = X(pot[i] + str(i));
            self.clauses.add(a);


    # give you group number and the pots array,  (~A0 | ~A1) AND (~A0 | ~A2)
    def one_country_at_most_one_group(self, group_number, pots):
        s = ""
        for array in pots:
            for member in array:
                for i in range(0, group_number):
                    for j in range(i + 1, group_number):
                        s = ~X(member + str(i)) | ~ X(member + str(j))
                        self.clauses.add(s)
        return self.clauses


    # ~A0 | ~B0 AND
    def same_pots_diff_group(self, group_number, pots):
        s = ""
        for array in pots:
            for i in range(0, len(array)):
                for j in range(i + 1, len(array)):
                    for k in range(0, group_number):
                        s = ~ X(array[i] + str(k)) | ~ X(array[j] + str(k))
                        self.clauses.add(s)
        return self.clauses

    def is_valid(self, c1, c2, c3, group):
        for set in self.pots_set:
            count = 0
            if c1 in set:
                count += 1
            if c2 in set:
                count += 1
            if c3 in set:
                count += 1
            if count >= 2:
                return True

        if self.first_pot.has_key(c1) and self.first_pot[c1] != group:
            return True
        if self.first_pot.has_key(c2) and self.first_pot[c2] != group:
            return True
        if self.first_pot.has_key(c3) and self.first_pot[c3] != group:
            return True
        return False

    def Europe_constraints(self, group_number, Europe):
        s = ""
        for n in range(0, group_number):
            for i in range(0, len(Europe)):
                for j in range(i + 1, len(Europe)):
                    for k in range(j + 1, len(Europe)):
                        if self.is_valid(Europe[i], Europe[j], Europe[k], n) is True:
                            continue


                        s1 = ~X(Europe[i] + str(n)) | ~X(Europe[j] + str(n))
                        s2 = ~X(Europe[i] + str(n)) | ~X(Europe[k] + str(n))
                        s3 = ~X(Europe[j] + str(n)) | ~X(Europe[k] + str(n))
                        s = s1 | s2 | s3
                        self.clauses.add(s)
        print len(self.clauses)
        return self.clauses


    def get_clauses(self):
        self.optimize()
        self.Europe_constraints(self.group_number, self.UEFA)
        self.same_pots_diff_group(self.group_number, self.pots)
        self.same_pots_diff_group(self.group_number, self.normal)
        self.one_country_at_most_one_group(self.group_number, self.pots)
        self.one_country_at_least_one_group(self.group_number, self.pots)

        print "finish clauses"


    def pl_resolution(self):
        clauses = self.clauses
        new = set()
        while True:
            n = len(clauses)
            pairs = [(clauses[i], clauses[j])
                     for i in range(n) for j in range(i + 1, n)]
            for (ci, cj) in pairs:
                resolvents = self.pl_resolve(ci, cj)
                if False in resolvents:
                    return False
                new = new.union(set(resolvents))
            if new.issubset(set(clauses)):
                return True
            for c in new:
                if c not in clauses:
                    clauses.add(c)

    def pl_resolve(self, ci, cj):
        clauses = []
        for di in self.disjuncts(ci):
            for dj in self.disjuncts(cj):
                if di == ~dj or ~di == dj:
                    seqi = self.removeall(di, self.disjuncts(ci))
                    seqj = self.removeall(dj, self.disjuncts(cj))
                    dnew = self.unique(seqi + seqj)
                    if self.check_invert(dnew):
                        continue
                    tmp = self.associate('|', dnew)
                    clauses.add(tmp)
        return clauses

    def check_invert(self, seq):
        n = len(seq)
        if n <= 1:
            return False
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if seq[i] == ~seq[j] or seq[j] == ~seq[i]:
                        return True
            return False

    def disjuncts(self, s):
        return self.dissociate('|', [s])

    def dissociate(self, op, args):
        result = []

        def collect(subargs):
            for arg in subargs:
                if arg.op == op:
                    collect(arg.args)
                else:
                    result.append(arg)

        collect(args)
        return result

    def unique(self, seq):
        return list(set(seq))

    def removeall(self, item, seq):
        if isinstance(seq, str):
            return seq.replace(item, '')
        else:
            tmp = []
            for x in seq:
                # if x != item:
                #     tmp.add(x)
                if x.op == item.op and x.args == item.args:
                    continue
                else:
                    tmp.append(x)
            return tmp

    def associate(self, op, args):
        args = self.dissociate(op, args)
        if len(args) == 0:
            return self._op_identity[op]
        elif len(args) == 1:
            return args[0]
        else:
            return X(op, *args)

    _op_identity = {'&': True, '|': False, '+': 0, '*': 1}
#---------------------------------------------------------------------------------------

    # def conjuncts(self, s):
    #     return self.dissociate('&', [s])

    def find_pure_symbol(self, symbols, clauses):
        for s in symbols:
            found_pos, found_neg = False, False
            for c in clauses:
                if not found_pos and s in self.disjuncts(c):
                    found_pos = True
                if not found_neg and ~s in self.disjuncts(c):
                    found_neg = True
            if found_pos != found_neg:
                return s, found_pos
        return None, None

    def find_unit_clause(self, clauses, model):
        for clause in clauses:
            P, value = self.unit_clause_assign(clause, model)
            if P:
                return P, value
        return None, None

    def unit_clause_assign(self, clause, model):
        P, value = None, None
        for literal in self.disjuncts(clause):
            sym, positive = self.inspect_literal(literal)
            if sym in model:
                if model[sym] == positive:
                    return None, None  # clause already True
            elif P:
                return None, None  # more than 1 unbound variable
            else:
                P, value = sym, positive
        return P, value

    def inspect_literal(self, literal):
        if literal.op == '~':
            return literal.args[0], False
        else:
            return literal, True

    def extend(self, s, var, val):
        s2 = s.copy()
        s2[var] = val
        return s2

    def dpll_satisfiable(self):
        s = copy.deepcopy(self.clauses)
        clauses = s
        # symbols = list(set(sym for clause in s for sym in self.prop_symbols(clause)))
        tmp = self.associate('&', s)
        symbols = self.prop_symbols(tmp)
        return self.dpll(clauses, symbols, {})

    def dpll(self, clauses, symbols, model):
        unknown_clauses = []  # clauses with an unknown truth value
        for c in clauses:
            val = self.pl_true(c, model)
            if val is False:
                return False
            if val is not True:
                unknown_clauses.append(c)

        if not unknown_clauses:
            return model
        P, value = self.find_unit_clause(clauses, model)
        if P:
            new_symbols = self.removeall(P, symbols)
            new_model = self.extend(model, P, value)
            return self.dpll(clauses, new_symbols, new_model)
        P, value = self.find_pure_symbol(symbols, unknown_clauses)
        if P:
            new_symbols = self.removeall(P, symbols)
            # new_symbols = symbols.remove(P)
            new_model = self.extend(model, P, value)
            return self.dpll(clauses, new_symbols, new_model)

        if not symbols:
            raise TypeError("Argument should be of the type Expr.")
        P, symbols = symbols[0], symbols[1:]
        return (self.dpll(clauses, symbols, self.extend(model, P, True)) or
                self.dpll(clauses, symbols, self.extend(model, P, False)))
#-------------------------------------------------------------------------------------------------------



    def prop_symbols(self, x):
        if not isinstance(x, X):
            return []
        elif self.is_prop_symbol(x.op):
            return [x]
        else:
            return list(set(symbol for arg in x.args for symbol in self.prop_symbols(arg)))

    def pl_true(self, exp, model={}):
        if exp in (True, False):
            return exp
        op, args = exp.op, exp.args
        if self.is_prop_symbol(op):
            return model.get(exp)
        elif op == '~':
            p = self.pl_true(args[0], model)
            if p is None:
                return None
            else:
                return not p
        elif op == '|':
            result = False
            for arg in args:
                p = self.pl_true(arg, model)
                if p is True:
                    return True
                if p is None:
                    result = None
            return result
        elif op == '&':
            result = True
            for arg in args:
                p = self.pl_true(arg, model)
                if p is False:
                    return False
                if p is None:
                    result = None
            return result
    #!!!!!!!!!!!!!
    def is_prop_symbol(self, s):
        return self.is_symbol(s)

    def is_symbol(self, s):
        return isinstance(s, str) and s[0].isalpha()

    def probability(self, p):
        return p > random.uniform(0.0, 1.0)
    def corner_case(self):
        if len(self.pots[0]) > self.group_number or len(self.UEFA) > self.group_number * 2:
            return True
        for line in self.pots:
            if len(line) > self.group_number:
                return True
        return False
    def solver(self, filename, filenumber):

        self.get_file(filename)
        output = "output" + str(filenumber) + ".txt";
        if self.corner_case() is True:
            with open(output, 'w') as f:
                f.write('No\n')
            return
        else:
            self.get_clauses()
            tag = self.dpll_satisfiable()
            print "finish self.dpll_satisfiable()"
            if not tag:
                self.write_file(output, False, '')
            else:
                # ans = self.WalkSAT(p=0.8)
                self.write_file(output, True, tag)

    def write_file(self, filename, tag, args):
        res = self.group_number * [""] ;
        if tag == False:
            with open(filename, 'w') as f:
                f.write('No\n')
            return
        else:
            with open(filename, 'w') as f:
                f.write('Yes\n')
                keys = args.keys()
                for key in keys:
                    if args[key] == True:
                        ss = key.op
                        country = ss[0:len(ss) - 1]
                        group = int(ss[len(ss) - 1])
                        res[group] = res[group] + country + ','
                for i in res:
                    f.write(i[0:len(i) - 1] +'\n')


def main():
    for i in range(1, 8):
        print "input: " + str(i)
        start = time()
        filename = "input" + str(i) + ".txt"
        agent = Agent()
        agent.solver(filename, i)
        stop = time()
        print("finish input" + str(i) + ", spend " + str(stop - start) + "s")

        print "begin to verify the answers: "



if __name__ == '__main__':
    main()