i = 0
def prints(x):
    global i
    try:
        print("Shape %i : %s, %s" % (i, x.shape, type(x)))
        i = i + 1
    except:
        print("exception")
