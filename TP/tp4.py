import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

# year      age       agesq     nbh       cbd       inst      linst
# price    rooms     area      land      baths     dist      ldist
# lprice    y81      larea     lland     linstsq


PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    df = pd.read_csv(PATH + "/data/textfiles/hprice3.raw", delim_whitespace=True, header=None)

    # ex.1:
    price = df[7] / 100

    plt.hist(price, "auto")
    plt.show()

    # regression on price
    s = np.shape(price)
    const = np.ones(s)
    age = df[1]
    nbh = df[3]
    inst = df[5]
    rooms = df[8]
    area = df[9]
    land = df[10]
    baths = df[11]
    dist = df[12]
    y81 = df[15]

    y = price

    X = np.column_stack((const, age, nbh, inst, rooms, area, land, baths, dist, y81))
    model = sm.OLS(y, X)
    result = model.fit()
    print("Ex.1:\n", result.summary())

    s = y81 == 0
    p0 = np.mean(price[s])
    s = y81 == 1
    p1 = np.mean(price[s])
    print(f"Ex:1:\n\tp1-p0 (variation of the mean of prices 1981 vs 1978): {p1-p0}")

    # ex.2:
    u = result.resid
    u2 = u**2
    y = u2
    model = sm.OLS(y, X)
    result = model.fit()
    print("Ex.2:\n", result.summary())

    # ex.3:
    # calculate F statistic explicitly with the formula ((SSR1-SSR0)/2) / (SSR0/(n-k))
    u = result.resid
    n, k = np.shape(X)
    SSR0 = u.T @ u

    X2 = np.column_stack((const, age, nbh, inst, rooms, baths, dist, y81))
    model = sm.OLS(y, X2)
    result = model.fit()
    print("Ex.3:\n", result.summary())

    u = result.resid
    SSR1 = u.T @ u
    F = ((SSR1 - SSR0) / 2) / (SSR0 / (n - k))
    print(f"Ex:3\n\tF: {F}\n")

    print(SSR0, SSR1)

    # ex.4:
    min_baths = np.min(baths)
    max_baths = np.max(baths)
    print("Ex:4\n")
    print("min_baths:", min_baths)
    print("max_baths:", max_baths)

    baths_binary = np.eye(4)[baths - 1]

    X3 = np.column_stack((const, age, nbh, inst, rooms, area, land, baths_binary, dist, y81))

    y = price
    model = sm.OLS(y, X3)
    result = model.fit()
    print("Ex.4:\n", result.summary())

    u = result.resid
    u2 = u**2
    y = u2
    model = sm.OLS(y, X3)
    result = model.fit()
    print("Ex.4:\n", result.summary())

    # ex.5:
    log_area = np.log(area)
    log_land = np.log(land)

    X4 = np.column_stack((const, age, nbh, inst, rooms, log_area, log_land, baths_binary, dist, y81))

    y = price
    model = sm.OLS(y, X4)
    result = model.fit()

    u = result.resid
    u2 = u**2
    y = u2
    model = sm.OLS(y, X4)
    result = model.fit()
    print("Ex.5:\n", result.summary())

    # ex.6:
    y = np.log(price)
    model = sm.OLS(y, X4)
    result = model.fit()

    u = result.resid
    u2 = u**2
    y = u2
    model = sm.OLS(y, X4)
    result = model.fit()
    print("Ex.6:\n", result.summary())

    # ex.7:
    h = np.sqrt(log_land)
    y = np.log(price)
    X5 = np.column_stack((const, age, nbh, inst, rooms, log_area, log_land, baths_binary, dist, y81))

    model = sm.WLS(y, X5, weights=1 / h)
    result = model.fit()
    print("Ex.7:\n", result.summary())

    # ex.8:
    log_price = np.log(price)

    plt.scatter(log_land, log_price)
    plt.xlabel("log_land")
    plt.ylabel("log_price")
    plt.show()

    # create mask for log_land <= 10
    mask = log_land <= 10
    log_land_under_10 = log_land[mask]

    X6 = np.column_stack(
        (
            const[mask],
            age[mask],
            nbh[mask],
            inst[mask],
            rooms[mask],
            log_area[mask],
            log_land_under_10,
            baths_binary[mask],
            dist[mask],
            y81[mask],
        )
    )

    y = log_price[mask]
    model = sm.OLS(y, X6)
    result = model.fit()

    u = result.resid
    u2 = u**2
    y = u2
    model = sm.OLS(y, X6)
    result = model.fit()
    print("Ex.8.1:\n", result.summary())

    # create mask for log_land > 10
    mask = log_land > 10
    log_land_over_10 = log_land[mask]

    X7 = np.column_stack(
        (
            const[mask],
            age[mask],
            nbh[mask],
            inst[mask],
            rooms[mask],
            log_area[mask],
            log_land_over_10,
            baths_binary[mask],
            dist[mask],
            y81[mask],
        )
    )

    y = log_price[mask]
    model = sm.OLS(y, X7)
    result = model.fit()

    u = result.resid
    u2 = u**2
    y = u2
    model = sm.OLS(y, X7)
    result = model.fit()
    print("Ex.8.2:\n", result.summary())


if __name__ == "__main__":
    main()
