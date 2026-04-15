def masksCompatible(m1, m2):
    cnt = 0
    for i in range(10):
        if m1[i] + m2[i] > 1:
            return False
        if m1[i] + m2[i] == 1:
            cnt += 1
    if cnt == 9:
        return False
    return True

def addMasks(m1, m2):
    for i in range(10):
        m1[i] += m2[i]
    return m1

def removeMask(point, mask):
    for i in range(10):
        if mask[i] == 1:
            point.mask[i] = 0
