import numpy as np
from numpy.linalg import inv
from function import showResult
from function import gradient, f, hessiyan

Ru = 0.8
C = 0.3
count = 1
epsi=10e-3
maxiter = 400


def backtrack_line_search(x0):

    alpha = 1

    while (f(x0) - (f(x0 - alpha * gradient(x0)) + alpha * C * np.dot(gradient(x0),gradient(x0)))) < 0:
        alpha *= Ru

    return alpha


def steepestDescent(x):

    alpha = backtrack_line_search(x)

    xk_s = []

    alphas = []

    step = 0

    while (f(x) > 0.1e-18):

        step += 1

        xk_s.append(x)

        alphas.append(alpha)

        x = (x - np.multiply(alpha,  gradient(x)))

        alpha = backtrack_line_search(x)

    return xk_s, alphas, step


def newton(x):

    alpha = backtrack_line_search(x)

    step = 0

    alphas = []

    xk_s = []

    while (f(x) > 0.1e-18):

        print(f(x))

        xk_s.append(x)

        alphas.append(alpha)

        x = (x - np.multiply(alpha, inv(hessiyan(x)).dot(gradient(x))))

        alpha = backtrack_line_search(x)

        step += 1

        if (f(x)==0):

            break

    return xk_s, alphas, step


def bfgs_method(x0):

    gfk = gradient(x0)
    I = np.eye(len(x0), dtype=int)
    Hk = I
    xk = x0
    xs = []
    alphas = []
    steps = 0

    while (f(xk) > 0.1e-18):

        xs.append(xk)
        steps += 1
        pk = -np.dot(Hk, gfk)
        alpha_k = backtrack_line_search(x0)
        alphas.append(alpha_k)
        xk1 = xk + alpha_k * pk
        sk = xk1 - xk
        xk = xk1

        gfk1 = gradient(xk)
        yk = gfk1 - gfk
        gfk = gfk1

        term = 1.0 / (np.dot(yk, sk))
        term1 = I - term * sk[:, np.newaxis] * yk[np.newaxis, :]
        term2 = I - term * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(term1, np.dot(Hk, term2)) + (term * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    return xs, alphas, steps


x = np.array(np.transpose(np.ones(5)))

xs, alphas, steps = bfgs_method(x)
showResult(xs, alphas, steps)

xs, alphas, steps = newton(x)
showResult(xs, alphas, steps)

xs, alphas, steps = steepestDescent(x)
showResult(xs, alphas, steps)