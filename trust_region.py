import numpy as np
from math import sqrt
from function import gradient, f, hessiyan
from function import showResult


def dogleg_method(Hk, gk, Bk, trust_radius):

    pB = -np.dot(Hk, gk)

    norm_pB = sqrt(np.dot(pB, pB))

    if norm_pB <= trust_radius:
        return pB

    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk

    norm_pU = sqrt(np.dot(pU, pU))

    if norm_pU >= trust_radius:

        return trust_radius * pU / norm_pU

    pB_minus_pU = pB - pU

    pB_dot_pU = np.dot(pB_minus_pU, pB_minus_pU)

    pU_dot_pB_minus_pU = np.dot(pU, pB_minus_pU)

    term = pU_dot_pB_minus_pU ** 2 - pB_dot_pU * (norm_pU - trust_radius ** 2)

    tau = (-pU_dot_pB_minus_pU + sqrt(term)) / pB_dot_pU

    return pU + tau * pB_minus_pU


def find_cauchy_point(gk, Bk, trust_radius):

    gk_bk_gk = np.dot(gk, np.dot(Bk, gk))

    tmp_tau = (np.sqrt(np.dot(gk, gk))**3) / (trust_radius * gk_bk_gk)

    if(gk_bk_gk <= 0):

        tau = 1

    else:

        tau = min(tmp_tau, 1)

    alpha = (-1 * tau * trust_radius) / (np.sqrt(np.dot(gk, gk)))

    pk = alpha * gk

    return pk


def trust_region_find_delta(xk, pk, gk, Bk, trust_radius):

    max_trust_radius = 100.0

    eta = 0.15

    func_red = f(xk) - f(xk + pk)

    model_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

    if model_red == 0.0:
        Ru = 1e99

    else:
        Ru = func_red / model_red

    norm_pk = sqrt(np.dot(pk, pk))

    if Ru < 0.25:
        trust_radius = 0.25 * trust_radius

    else:

        if Ru > 0.75 and norm_pk == trust_radius:
            trust_radius = min(2.0 * trust_radius, max_trust_radius)

        else:

            trust_radius = trust_radius

    if Ru > eta:
        xk = xk + pk

    else:
        xk = xk


    return xk, trust_radius, Ru


def trust_region_dogleg(xk):

    trust_radius = 1.0

    iteration = 0

    xk_s = []

    Ru_s = []

    while True:

        xk_s.append(xk)

        gk = gradient(xk)

        Bk = hessiyan(xk)

        Hk = np.linalg.inv(Bk)

        pk = dogleg_method(Hk, gk, Bk, trust_radius)

        xk, trust_radius, Ru = trust_region_find_delta(xk, pk, gk, Bk, trust_radius)

        Ru_s.append(Ru)

        if f(xk) < 0.1e-18:
            break

        iteration = iteration + 1

    showResult(xk_s, Ru_s, iteration)

    return xk


def trust_region_cauchy(xk):

    trust_radius = 1.0

    iteration = 0

    xk_s = []

    Ru_s = []

    while True:

        xk_s.append(xk)

        gk = gradient(xk)

        Bk = hessiyan(xk)

        pk = find_cauchy_point(gk, Bk, trust_radius)



        print(f(xk))

        xk, trust_radius, Ru = trust_region_find_delta(xk, pk, gk, Bk, trust_radius)

        Ru_s.append(Ru)

        if f(xk) < 0.0012310:
            break

        iteration = iteration + 1

    showResult(xk_s, Ru_s, iteration)

    return xk


# trust_region_dogleg([1.2, 1.2])
