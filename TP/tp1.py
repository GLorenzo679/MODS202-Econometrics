import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# wage      educ      exper     tenure    nonwhite  female    married   numdep
# smsa      northcen  south     west      construc  ndurman   trcommpu  trade
# services  profserv  profocc   clerocc   servocc   lwage     expersq   tenursq


PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    # ex.1 - load dataset
    df = pd.read_csv(PATH + "/data/textfiles/wage1.raw", delim_whitespace=True, header=None)

    # ex.2 - histogram of wage
    wage = df[0]

    # plt.hist(wage, "auto")
    # plt.show()

    # ex.3 - stats of wage
    wage_mean = np.mean(wage)
    wage_std = np.std(wage)
    wage_max = np.max(wage)
    wage_min = np.min(wage)

    print("Mean:\t", wage_mean)
    print("Std:\t", wage_std)
    print("Max:\t", wage_max)
    print("Min:\t", wage_min)

    # ex.4 - compute covariance and correlation between wage and educ
    educ = df[1]
    wage_educ_cov = np.cov(wage, educ)
    wage_educ_corr = np.corrcoef(wage, educ)

    print("Covariance:\n", wage_educ_cov)
    print("Correlation:\n", wage_educ_corr)

    # ex.5 - scatter plot of wage and educ
    # plt.scatter(wage, educ)
    # plt.show()

    # ex.6 - compute average wage of men and women
    women = df[5]
    s = women == 1

    avg_women_wage = np.mean(wage[s])
    print("Average salary of women:", avg_women_wage)

    avg_man_wage = np.mean(wage[~s])
    print("Average salary of man:", avg_man_wage)

    # ex.7 - compute the average wage of women who have a wage higher than the median wage
    median_wage = np.median(wage)
    s1 = wage > median_wage
    avg_women_more_median = np.mean(wage[s][s1])
    print("Average salary of women who have a wage higher than the median wage:", avg_women_more_median)

    # ex.8 compute the 5th percentile of wage

    index_5th = math.ceil(((5 * (len(wage) / 100)) - 1))

    # first method
    np_5th = np.percentile(wage, 5)
    print("Numpy 5th percentile of wage:", np_5th)

    # second method
    pd_5th = wage.sort_values(ascending=True)
    print("Pandas 5th percentile of wage:", pd_5th.iloc[index_5th])

    # ex.9 - plot the mean of wage for each tenure year
    tenure = df[3]

    for t in tenure.unique():
        s = tenure == t
        plt.scatter(t, np.mean(wage[s]))

    # plt.show()

    # ex.10 - remove observations with wage > 10
    s = wage <= 10
    df1 = np.array(df)
    df2 = df1[s, :]
    print(f"Removed {df1.shape[0] - df2.shape[0]} observations")

    # ex.11 - add back the observations removed in ex.10
    s1 = wage > 10
    df3 = df1[s1, :]
    df_tot = np.concatenate((df2, df3), axis=0)
    print(f"Added {df3.shape[0]} observations")
    print(df_tot.shape)


main()
