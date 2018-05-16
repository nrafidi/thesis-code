import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    x = np.linspace(0.0, 20.0, 1000)

    sig = np.sin(x) + np.cos(3*x) + np.sin(10*x)

    fig, ax = plt.subplots()
    ax.plot(x, sig)

    ax.plot(x[::10], sig[::10])
    plt.show()