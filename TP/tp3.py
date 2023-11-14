import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

# wage      educ      exper     tenure    nonwhite  female    married   numdep
# smsa      northcen  south     west      construc  ndurman   trcommpu  trade
# services  profserv  profocc   clerocc   servocc   lwage     expersq   tenursq


PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    df = pd.read_csv(PATH + "/data/textfiles/wage1.raw", delim_whitespace=True, header=None)

    wage = df[0]
    y = np.log(wage)
    const = np.ones(y.shape)
    educ = df[1]
    exper = df[2]
    tenure = df[3]

    X = np.column_stack((const, educ, exper, tenure))

    # ex.1.1

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    u = results.resid
    SSR0 = u.T @ u

    print(f"Ex:1.1\n\tSSR0: {SSR0}\n")

    # ex.1.2
    X0 = X
    X = np.column_stack((const, tenure))
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    u = results.resid
    SSR1 = u.T @ u
    print(f"Ex:1.2\n\tSSR1: {SSR1}\n")

    # ex.1.3
    n, k = np.shape(X0)
    F = ((SSR1 - SSR0) / 2) / (SSR0 / (n - k))
    p_value = f.sf(F, 2, n - k)
    print(f"Ex:1.3\n\tF: {F}\n\tp-value: {p_value}\n")

    # ex.2 - REVIEW
    X = np.column_stack((const, educ, tenure))
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    u = results.resid
    SSR1 = u.T @ u

    F = ((SSR1 - SSR0) / 2) / (SSR0 / (n - k))
    p_value = f.sf(F, 1, n - k)
    print(f"Ex:2\n\tF: {F}\n\tp-value: {p_value}\n")

    # ex.3
    X = const
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    u = results.resid
    SSR1 = u.T @ u

    F = ((SSR1 - SSR0) / 3) / (SSR0 / (n - k))
    p_value = f.sf(F, 3, n - k)
    print(f"Ex:3\n\tF: {F}\n\tp-value: {p_value}\n")

    # ex.4
    X = np.column_stack((const, tenure))
    y = np.log(wage) - 0.1 * educ - 0.01 * exper
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    u = results.resid
    SSR1 = u.T @ u

    F = ((SSR1 - SSR0) / 2) / (SSR0 / (n - k))
    p_value = f.sf(F, 2, n - k)
    print(f"Ex:4\n\tF: {F}\n\tp-value: {p_value}\n")

    # ex.5
    female = df[5]
    male = 1 - female
    married = df[6]
    marrfem = married * female
    X = np.column_stack((const, male, marrfem, educ, exper, tenure))
    n, k = np.shape(X)
    y = np.log(wage)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    # ex.6
    singmale = (1 - married) * male
    singfemale = (1 - married) * female
    X = np.column_stack((const, singmale, singfemale, marrfem, educ, exper, tenure))
    n, k = np.shape(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    # ex.7
    SSR1 = SSR0
    northcen = df[9]
    south = df[10]
    west = df[11]
    X = np.column_stack((const, northcen, south, west, educ, exper, tenure))
    n, k = np.shape(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    u = results.resid
    SSR0 = u.T @ u

    F = ((SSR1 - SSR0) / 3) / (SSR0 / (n - k))
    p_value = f.sf(F, 3, n - k)
    print(f"Ex:7\n\tF: {F}\n\tp-value: {p_value}\n")

    # ex.8
    femeduc = female * educ
    X = np.column_stack((const, femeduc, educ, exper, tenure))
    n, k = np.shape(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    # ex.9
    femexper = female * exper
    femtenure = female * tenure
    X = np.column_stack((const, female, femeduc, femexper, femtenure, educ, exper, tenure))
    n, k = np.shape(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    u = results.resid
    SSR0 = u.T @ u

    F = ((SSR1 - SSR0) / 4) / (SSR0 / (n - k))
    p_value = f.sf(F, 4, n - k)
    print(f"Ex:9\n\tSSR0: {SSR0}\n\tF: {F}\n\tp-value: {p_value}\n")

    # ex.10
    s = female == 1
    y = np.log(wage)
    X = X0[s, :]
    y = y[s]
    n, k = np.shape(X)
    model = sm.OLS(y, X)
    results = model.fit()
    u = results.resid
    SSR01 = u.T @ u

    s = female == 0
    y = np.log(wage)
    X = X0[s, :]
    y = y[s]
    n, k = np.shape(X)
    model = sm.OLS(y, X)
    results = model.fit()
    u = results.resid
    SSR00 = u.T @ u

    SSR0 = SSR00 + SSR01

    F = ((SSR1 - SSR0) / 1) / (SSR0 / (n - k))
    p_value = f.sf(F, 1, n - k)
    print(f"Ex:10\n\tSSR00: {SSR00}\n\tSSR01: {SSR01}\n\tF: {F}\n\tp-value: {p_value}\n")


main()
