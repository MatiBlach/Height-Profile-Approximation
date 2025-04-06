from pandas import read_csv
from functools import reduce
import matplotlib.pyplot as plt
import operator
import math

DATA_SIZE = 510

def get_data(filename):
    records = read_csv("paths/" + filename + ".csv").values
    distances = [record[0] for record in records]
    elevations = [record[1] for record in records]
    return distances, elevations


def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i

def norm(vec: list):
    return sum([y[0]**2 for y in vec])**0.5

def matrixsub(mat1: list, mat2: list) -> list:
    mat = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
    for y in range(len(mat1)):
        for x in range(len(mat1[0])):
            mat[y][x] = mat1[y][x] - mat2[y][x]
    return mat


def matrixmul(mat1: list, mat2: list):
    y1 = len(mat1)
    x1 = len(mat1[0])
    x2 = len(mat2[0])
    mat = [[0 for _ in range(x2)] for _ in range(y1)]

    for y in range(len(mat)):
        for x in range(len(mat[0])):
            mat[y][x] = sum([mat1[y][l] * mat2[l][x] for l in range(x1)])

    return mat

def res(mat_a: list, b: list, x: list) -> list:
    return matrixsub(matrixmul(mat_a, x), b)


def lagrange(X, Y, nodes=9, indexes=None):

    if indexes is None:
        indexes = [int(i) for i in linspace(0, len(X) - 1, nodes)]

    def F(x):
        return sum(
            reduce(
                operator.mul,
                [(x - X[j]) / (X[i] - X[j]) for j in indexes if i != j],
                1) * Y[i]
            for i in indexes)


    interpolated_X = list(linspace(X[0], X[-1], 1000))
    interpolated_Y = [F(x) for x in interpolated_X]

    return interpolated_X, interpolated_Y, indexes

def splines(X, Y, nodes=15,indexes=None):

    if indexes is None:
        indexes = [int(i) for i in linspace(0, len(X) - 1, nodes)]
    
    n = len(indexes)
    a = [Y[ix] for ix in indexes]
    b, c, d = [0] * (n-1), [0] * n, [0] * (n-1)
    h = [X[indexes[i + 1]] - X[indexes[i]] for i in range(n - 1)]
    alpha = [0] * n

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (Y[indexes[i + 1]] - Y[indexes[i]]) - (3 / h[i - 1]) * (Y[indexes[i]] - Y[indexes[i - 1]])

    l = [1] + [0] * (n-1)
    mu = [0] * (n-1) + [0]
    z = [0] * n

    for i in range(1, n - 1):
        l[i] = 2 * (X[indexes[i + 1]] - X[indexes[i - 1]]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1
    z[n - 1] = 0
    c[n - 1] = 0

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (Y[indexes[j + 1]] - Y[indexes[j]]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    def f(x):
        ix = n - 2  # Default to the last interval
        for i in range(n - 1):
            if X[indexes[i]] <= x < X[indexes[i + 1]]:
                ix = i
                break
        dx = x - X[indexes[ix]]
        return a[ix] + b[ix] * dx + c[ix] * dx ** 2 + d[ix] * dx ** 3

    interpolated_X = list(linspace(X[0], X[-1], 1000))
    interpolated_Y = [f(x) for x in interpolated_X]

    return interpolated_X, interpolated_Y, indexes


def plot_evenly(filename, title, plot_filename, nodes, function,show_nodes=True):
    X, Y = get_data(filename)
    x, y, ixs = function(X, Y, nodes)
    
    plt.title(f"{title}")
    plt.xlabel("odległość [m]")
    plt.ylabel("wysokość [m]")
    plt.plot(X, Y)
    plt.plot(x, y)
    if(show_nodes):
        plt.scatter([X[i] for i in ixs], [Y[i] for i in ixs], c='g')
    plt.legend(["dane", "interpolacja", "węzły"])

    plt.savefig("plots/" + plot_filename + ".png")
    plt.show()
    
def chebyshev(N):
    def f(n):
        return DATA_SIZE/2 * math.cos((2 * n + 1)/(2 * N) * math.pi) + DATA_SIZE/2
    return [int(f(n)) for n in range(N-1, -1, -1)] + [DATA_SIZE]
    
def plot_chebyshev(filename, title, plot_filename, nodes, function, show_nodes=True):
    X, Y = get_data(filename)
    x, y, ixs = function(X, Y, indexes=chebyshev(nodes))

    plt.title(f"{title}")
    plt.xlabel("odległość [m]")
    plt.ylabel("wysokość [m]")
    plt.plot(X, Y)
    plt.plot(x, y)
    if(show_nodes):
        plt.scatter([X[i] for i in ixs], [Y[i] for i in ixs], c='g')
    plt.legend(["dane", "interpolacja", "węzły"])

    plt.savefig("plots/" + plot_filename + ".png")
    plt.show()


plot_evenly("Obiadek", "Obiadek - Interpolacja Lagrange, 5 węzłów", function=lagrange, nodes=5,plot_filename="wykres1")
plot_evenly("Obiadek", "Obiadek - Interpolacja Lagrange, 10 węzłów", function=lagrange, nodes=10,plot_filename="wykres2")
plot_evenly("Obiadek", "Obiadek - Interpolacja Lagrange, 15 węzłów", function=lagrange, nodes=15,plot_filename="wykres3")
plot_chebyshev("Obiadek", "Obiadek - Interpolacja Lagrange, 30 węzłów Chebysheva", nodes=30, function=lagrange,plot_filename="wykres4")


plot_evenly("Obiadek", "Obiadek - Interpolacja funkcjami sklejanymi, 5 węzłów", function=splines, nodes=5,plot_filename="wykres5")
plot_evenly("Obiadek", "Obiadek - Interpolacja funkcjami sklejanymi, 10 węzłów", function=splines, nodes=10,plot_filename="wykres6")
plot_evenly("Obiadek", "Obiadek - Interpolacja funkcjami sklejanymi, 100 węzłów", function=splines, nodes=100,plot_filename="wykres7",show_nodes=False)
plot_evenly("Obiadek", "Obiadek - Interpolacja funkcjami sklejanymi, 500 węzłów", function=splines, nodes=500,plot_filename="wykres8",show_nodes=False)

plot_evenly("MountEverest", "Mount Everest - Interpolacja Lagrange, 5 węzłów", function=lagrange, nodes=5,plot_filename="wykres9")
plot_evenly("MountEverest", "Mount Everest- Interpolacja Lagrange, 10 węzłów", function=lagrange, nodes=10,plot_filename="wykres10")
plot_chebyshev("MountEverest", "Mount Everest - Interpolacja Lagrange, 10 węzłów Chebysheva", nodes=10, function=lagrange,plot_filename="wykres11")
plot_chebyshev("MountEverest", "Mount Everest - Interpolacja Lagrange, 30 węzłów Chebysheva", nodes=10, function=lagrange,plot_filename="wykres12")


plot_evenly("MountEverest", "Mount Everest - Interpolacja funkcjami sklejanymi, 5 węzłów", function=splines, nodes=5,plot_filename="wykres13")
plot_evenly("MountEverest", "Mount Everest- Interpolacja funkcjami sklejanymi, 10 węzłów", function=splines, nodes=10,plot_filename="wykres14")
plot_evenly("MountEverest", "Mount Everest- Interpolacja funkcjami sklejanymi, 30 węzłów", function=splines, nodes=30,plot_filename="wykres15",show_nodes=False)
plot_chebyshev("MountEverest", "Mount Everest - Interpolacja funkcjami sklejanymi 30 węzłów Chebysheva", nodes=30, function=splines,plot_filename="wykres16",show_nodes=False)


plot_evenly("SpacerniakGdansk", "Spacerniak Gdańsk - Interpolacja Lagrange, 10 węzłów", function=lagrange, nodes=10,plot_filename="wykres17")
plot_chebyshev("SpacerniakGdansk", "Spacerniak Gdańsk - Interpolacja Lagrange, 30 węzłów Chebysheva", nodes=30, function=lagrange,plot_filename="wykres18",show_nodes=False)
plot_evenly("SpacerniakGdansk", "Spacerniak Gdańsk - Interpolacja funkcjami sklejanymi, 30 węzłów", function=splines, nodes=30,plot_filename="wykres19",show_nodes=False)
plot_chebyshev("SpacerniakGdansk", "Spacerniak Gdańsk - Interpolacja funkcjami sklejanymi, 30 węzłów Chebysheva", nodes=30, function=splines,plot_filename="wykres20",show_nodes=False)