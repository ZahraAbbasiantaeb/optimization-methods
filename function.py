import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dfx1 = lambda x: 400 * (x[0] ** 3) - 400 * x[0] * x[1] + 2 * x[0] - 2

dfx2 = lambda x: 200 * (x[1] - x[0] ** 2)

f =  lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def gradient(x):

    return np.array([dfx1(x), dfx2(x)])


def hessiyan(x):

    df11 = 1200 * np.square(x[0]) - 400 * x[1] + 2

    df12 = -400 * x[0]

    df21 = -400 * x[0]

    df22 = 200

    hess = np.array([[df11, df12], [df21, df22]])

    return hess


def plot_points(points):

    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = []

    for i in range (0,len(X)):
        Z.append(f([X[i], Y[i]]))

    print(len(points))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.contour3D(X, Y, Z, 150, cmap='binary')

    # plt.show()

    x = []
    y = []
    z = []

    for elem in points:
        x.append(elem[0])
        y.append(elem[1])

    for i in range(0, len(x)):
        z.append(f([x[i], y[i]]))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    ax.scatter(x, y, z, c = 'red', marker='o', s=500)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()

    return


def showResult(points, alphas, steps):

    print('reached the optimum point in following steps: ' + str(steps))

    x = np.arange(1, len(alphas)+1, 1)
    plt.scatter(x, alphas, s =2)

    plt.show()

    x = []
    y = []
    z = []

    for elem in points:
        x.append(elem[0])
        y.append(elem[1])

    for i in range(0, len(x)):
        z.append(f([x[i], y[i]]))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c='red', marker='o', s=50)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()

    return