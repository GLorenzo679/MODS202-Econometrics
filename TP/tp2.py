import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

# wage      educ      exper     tenure    nonwhite  female    married   numdep
# smsa      northcen  south     west      construc  ndurman   trcommpu  trade
# services  profserv  profocc   clerocc   servocc   lwage     expersq   tenursq


PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    df = pd.read_csv(PATH + "/data/textfiles/wage1.raw", delim_whitespace=True, header=None)

    # ex.1
    wage = df[0]
    y = wage
    const = np.ones(y.shape)
    educ = df[1]
    exper = df[2]
    tenure = df[3]
    X = np.column_stack((const, educ, exper, tenure))

    # ex.2
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(f"Ex:2\nbeta: {beta}\n")

    # ex.3
    u = y - X @ beta
    n, k = np.shape(X)
    sig2 = u.T @ u / (n - k)
    Var = sig2 * np.linalg.inv(X.T @ X)
    std = np.sqrt(np.diag(Var))

    print(f"Ex:3\nstd: {std}\n")

    # ex.4
    plt.hist(u, "auto")
    # plt.show()

    s = np.abs(u) < 3 * np.sqrt(sig2)  # 3*std
    u1 = u[s]
    plt.hist(u1, "auto")
    # plt.show()

    y = y[s]
    X = X[s, :]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(f"Ex:4\nbeta: {beta}\n")

    # ex.5
    u = y - X @ beta
    n, k = np.shape(X)
    sig2 = u.T @ u / (n - k)
    Var = sig2 * np.linalg.inv(X.T @ X)
    std = np.sqrt(np.diag(Var))

    p_values = t.sf(beta[2] / std[2], n - k) * 2  # survival function
    print("Ex:5")
    print(f"student test:\t{beta[2] / std[2]}")
    print(t.ppf(0.025, n - k))
    print(f"p_values:\t{p_values}\n")

    # ex.6
    model = sm.OLS(y, X)
    results = model.fit()
    print("Ex:6")
    print(f"{results.summary()}\n")

    # ex.7
    y = np.log(y)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    u = y - X @ beta
    n, k = np.shape(X)
    sig2 = u.T @ u / (n - k)
    Var = sig2 * np.linalg.inv(X.T @ X)
    std = np.sqrt(np.diag(Var))

    p_values = t.sf(beta[2] / std[2], n - k) * 2  # survival function
    print("Ex:7")
    print(f"student test:\t{beta[2] / std[2]}")
    print(f"p_values:\t{p_values}\n")

    # ex.8 - REVIEW
    test = (beta[1] - 0.6) / std[1]
    p_values = 2 * (1 - t.sf(test, n - k))  # survival function

    print("Ex:8")
    print(f"student test:\t{test}")
    print(f"p_values:\t{p_values}\n")

    # ex.9 - REVIEW
    toteduc = educ + exper
    X = np.column_stack((const, educ, toteduc, tenure))
    X = X[s, :]
    model = sm.OLS(y, X)
    results = model.fit()
    print("Ex:9")
    print(f"{results.summary()}\n")

    # ex.10
    y = wage - educ
    diffeduc = exper - educ
    X = np.column_stack((const, educ, diffeduc, tenure))

    y = y[s]
    X = X[s, :]
    model = sm.OLS(y, X)
    results = model.fit()
    print("Ex:10")
    print(f"{results.summary()}\n")


main()
