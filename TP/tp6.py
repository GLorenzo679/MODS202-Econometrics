import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf

# gfr       pe        year      t         tsq       pe_1      pe_2      pe_3
# pe_4      pill      ww2       tcu       cgfr      cpe       cpe_1     cpe_2
# cpe_3     cpe_4     gfr_1     cgfr_1    cgfr_2    cgfr_3    cgfr_4    gfr_2


PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    df = pd.read_csv(
        PATH + "/data/textfiles/fertil3.raw", delim_whitespace=True, header=None
    )

    # ex.1:
    gfr = df[0]
    year = df[2]
    n = len(gfr)

    plt.plot(year, gfr)
    plt.title("GFR")
    plt.show()

    # ex.2:
    lgfr = np.log(gfr)
    dl = np.diff(lgfr)
    year = year[1:]

    plt.plot(year, dl)
    plt.title("dl")
    plt.show()

    # ex.3:
    dfl = pd.DataFrame(dl)
    dfl_1 = dfl.shift(1)
    dl_1 = dfl_1[0]

    cov = np.cov(dl[1:n], dl_1[1:n])
    corr = np.corrcoef(dl[1:n], dl_1[1:n])

    print("Ex.3:\n", "cov:\n", cov, "\ncorr:\n", corr)

    # ex.4:
    acf(dl)
    pacf(dl)
    print("\nEx.4:\n", "acf:\n", acf(dl), "\npacf:\n", pacf(dl))

    plot_acf(dl)
    plot_pacf(dl)
    plt.show()

    # ex.5:
    print("Ex.5:\n")
    mdl_1 = AutoReg(dl, 1).fit()
    print("AR(1)\n", mdl_1.params, mdl_1.aic, mdl_1.bic)

    mdl_3 = AutoReg(dl, 3).fit()
    print("AR(3)\n", mdl_3.params, mdl_3.aic, mdl_3.bic)

    best_p = 0
    best_aic = 100000
    best_bic = 100000

    for i in range(1, 11):
        mdl = AutoReg(dl, i).fit()
        # print(f"AR({i})\n", mdl.params, mdl.aic, mdl.bic)

        if mdl.aic < best_aic:
            best_aic = mdl.aic
            best_p = i

        if mdl.bic < best_bic:
            best_bic = mdl.bic
            best_p = i

    print("Best p:", best_p)


if __name__ == "__main__":
    main()
