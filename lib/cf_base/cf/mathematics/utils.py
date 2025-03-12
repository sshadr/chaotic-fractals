import numpy as np

# the next power of 2
def next_power_of_2(x):
    assert x > 0
    logx = np.log2(x)
    if logx == int(logx):
        logx += 1
    return np.exp2(np.ceil(logx))


# the previous power of 2
def prev_power_of_2(x):
    assert x > 0
    logx = np.log2(x)
    if logx == int(logx):
        logx -= 1
    return np.exp2(np.floor(logx))