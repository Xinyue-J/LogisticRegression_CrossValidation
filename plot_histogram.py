from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import Binarizer
import numpy as np


def readfile(filename):
    csv_data = pd.read_csv(filename, header=None)
    return csv_data


def bi(data):
    names = data.columns
    std = Binarizer().fit(data)
    test = pd.DataFrame(std.transform(data), columns=names)

    return test


def plot3d(x, y):
    plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.array(x), np.array(y)
    hist, xedges, yedges = np.histogram2d(x, y)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 1, yedges[:-1] + 1, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = 2.5 * np.ones_like(zpos)
    dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.show()


if __name__ == '__main__':

    xtest = readfile('H2W3 data files/Xtest.csv')
    ytest = readfile('H2W3 data files/Ytest.csv')

    xtest = bi(xtest)

    x = np.array(xtest.iloc[:, :48].sum(axis=1))
    y = np.array(xtest.iloc[:, 48:54].sum(axis=1))
    ytest = np.array(ytest)

    plt.figure()
    plt.scatter(x, y, c=ytest + 1, s=(ytest + 0.5) * 30)

    # 3d
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(x)):
        if ytest[i] == 0:
            type1_x.append(x[i])
            type1_y.append(y[i])
        elif ytest[i] == 1:
            type2_x.append(x[i])
            type2_y.append(y[i])
    plot3d(type1_x, type1_y)
    plot3d(type2_x, type2_y)
