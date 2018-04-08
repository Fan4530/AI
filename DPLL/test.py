def a():
    return "a"
def b():
    return "b"
def c():
    return "c"

def decorator(func):
    print 'start %s' % func()
def main():
    decorator(a)
    decorator(b)
    decorator(c)
main()
