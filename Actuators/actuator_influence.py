# Actuator Influence Functions

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    node_coord = np.loadtxt('node_coord.txt')
    for k in range(9):
        disp = np.loadtxt('disp_000%d.txt' %(k+1))

        plt.figure()
        plt.tricontourf(node_coord[:,0], node_coord[:,1], disp)
        plt.colorbar()
        plt.show()