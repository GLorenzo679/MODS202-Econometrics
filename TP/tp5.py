import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

# year      i3        inf       rec       out       def       i3_1
# inf_1     def_1     ci3       cinf      cdef      y77


PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    df = pd.read_csv(
        PATH + "/data/textfiles/intdef.raw", delim_whitespace=True, header=None
    )

    # ex.1:
    year = df[0]
    i3 = df[1]
    inf = df[2]
    deficit = df[5]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(year, i3)
    axs[0].set_title("3 mo. T bill rate")
    axs[1].plot(year, inf)
    axs[1].set_title("Inflation rate")
    axs[2].plot(year, deficit)
    axs[2].set_title("Deficit as % GDP")
    plt.show()

    # ex.2:
    n = len(inf)
    inf_1 = inf[0 : n - 1]
    def_1 = deficit[0 : n - 1]
    y = i3[1:n]
    const = np.ones(n - 1)
    X = np.column_stack((const, inf_1, def_1))

    model = sm.OLS(y, X)
    results = model.fit()
    print("Ex.2:\n", results.summary())

    # ex.3:
    u = results.resid
    n = len(u)
    u_1 = u[0 : n - 1]
    const = np.ones(n - 1)
    X = np.column_stack((const, u_1))
    X = X[:, 1]  # take only the u values, not the const values
    y = u[1:n]

    model = sm.OLS(y, X)
    results1 = model.fit()
    print("Ex.3:\n", results1.summary())

    # ex.4:
    rho = results1.params[0]

    n = len(inf)
    y = i3[1:n].reset_index(drop=True) - rho * i3[0 : n - 1].reset_index(drop=True)
    const = np.ones(n - 1)
    X = np.column_stack((const, inf_1, def_1))
    X_1 = X[0 : n - 1, :]
    X = X - rho * X_1

    model = sm.OLS(y, X)
    results2 = model.fit()
    print("Ex.4:\n", results2.summary())

    # ex.5:
    n = len(i3)
    y = i3[2:n]
    inf_1 = inf[1 : n - 1]
    inf_2 = inf[0 : n - 2]
    def_1 = deficit[1 : n - 1]
    def_2 = deficit[0 : n - 2]
    const = np.ones(n - 2)
    X = np.column_stack((const, inf_1, inf_2, def_1, def_2))

    model = sm.OLS(y, X)
    results = model.fit()
    print("Ex.5:\n", results.summary())

    d_inf = (results.params[1], results.params[2])
    d_def = (results.params[3], results.params[4])
    x = (1, 2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].bar(x, d_inf)
    axs[0].set_title("d_inf")
    axs[1].bar(x, d_def)
    axs[1].set_title("f_def")
    plt.show()

    # ex.6:
    n = len(i3)

    # unrestricted model
    y = i3[2:n]
    y_1 = i3[1 : n - 1]
    y_2 = i3[0 : n - 2]
    inf_1 = inf[1 : n - 1]
    inf_2 = inf[0 : n - 2]
    const = np.ones(n - 2)
    X = np.column_stack((const, y_1, y_2, inf_1, inf_2))

    model = sm.OLS(y, X)
    results = model.fit()

    k_ur = X.shape[1]
    SSR_ur = results.ssr

    # restricted model
    X = np.column_stack((const, y_1, y_2))

    model = sm.OLS(y, X)
    results = model.fit()

    k_r = X.shape[1]
    SSR_r = results.ssr

    # k_ur - k_r = number of restrictions
    # n - 2 = number of observations
    # k_ur = number of parameters in unrestricted model
    # k_r = number of parameters in restricted model

    F = ((SSR_r - SSR_ur) / SSR_ur) * ((n - 2 - k_ur) / (k_ur - k_r))
    p_value = 1 - f.cdf(F, k_ur - k_r, n - 2 - k_ur)
    print("Ex.6:\n", f"F: {F}\n", f"p-value: {p_value}\n")


if __name__ == "__main__":
    main()
