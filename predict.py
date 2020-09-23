from SimAHNnD import SimAHNnD

def predict(ahn, x, y):
    [ysim, Ss, ahn] = SimAHNnD(ahn, x, y)
    return ysim, Ss, ahn