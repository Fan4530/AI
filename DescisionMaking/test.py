x1 = (1, 0)
x2 = (0, 1)
print x1[0]

import operator
def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))
def vector_multiply(a, b):
    return (a[0] * b, a[1] * b);

print vector_multiply(x1, 4)

step = 3
for i in range(1, step):
    print i