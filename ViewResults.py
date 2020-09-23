import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ViewResults(Yestimates, y, mode='2D', x=0):

    if mode == '2D':
        err = (y - Yestimates)
        plt.figure(1, figsize=(10,5))
        plt.subplot(211)
        plt.plot(y, color='orange', label='Target Function "Y"', alpha=0.7)
        plt.plot(Yestimates, color='green', label='Function Approximation "Ysim"', linewidth=1.0)
        plt.legend()
        plt.subplot(212)
        plt.plot(err, color='red', label='Relative Error')
        plt.legend()
        plt.show()

    if mode == '3D':
        plt.figure()
        ax = plt.axes(projection='3d')
        Axes3D.plot(ax,xs=x[:,0], ys=x[:,1], zs=y.flatten(), color='orange', label='Target Function "Y"', alpha=0.7)
        Axes3D.plot(ax,xs=x[:,0], ys=x[:,1], zs=Yestimates.flatten(), color='green', label='Function Approximation "Ysim"', linewidth=1.0)
        plt.legend()
        plt.show()
    return