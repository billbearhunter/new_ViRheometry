from param import Param
import matplotlib.pyplot as plt
import numpy as np
import argparse

color_list = [(255/255, 0/255, 0/255), (0/255, 0/255, 255/255), (208/255, 2/255, 4/255)]

import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 30

def calcFlowCurve(m, x):
    eta = m.eta * 0.1
    # eta = m.eta
    n = m.n
    # sigmaY = m.sigmaY
    sigmaY = m.sigmaY * 0.1
    y = eta * np.power(x, n) + sigmaY
    return y

def flowPoints(file, params, outfile, extent_y):
    df = pd.read_table(file, header=5, encoding="UTF-16")

    fig, ax = plt.subplots()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**0, 10**2)
    plt.ylim(extent_y[0], extent_y[1])

    ax.plot(df['[1/s]'], df['[Pa]'], linestyle="dotted", linewidth=3.0, color="black")

    x = np.linspace(df['[1/s]'][5], df['[1/s]'][18], 10000)

    for i, param in enumerate(params):
        y = calcFlowCurve(param, x)
        ax.plot(x, y, color=color_list[i % len(color_list)], linewidth="3.0")

    ax.set_xlabel(r'$\dot{\gamma}[s^{-1}]$', fontsize=31, labelpad=1.8)
    ax.set_ylabel(r'$\sigma_s[Pa]$', fontsize=31, labelpad=1.8)
    ax.grid()
    plt.subplots_adjust(left=0.19, right=0.95, bottom=0.235, top=0.93)
    plt.savefig(outfile)


def main():
    parser = argparse.ArgumentParser(description="Plot flow curves from rheology data.")
    parser.add_argument(
        "--file", required=True,
        help="Path to input CSV file (e.g. ./Rheo_Data/Chuno_20230114_1523_25C.csv)"
    )
    parser.add_argument(
        "--est", nargs=3, type=float, action="append",
        metavar=("eta", "n", "sigmaY"),
        help="Fit parameters (eta n sigmaY). Repeat --est for multiple curves."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output file path (e.g. ./figs/chuno.pdf)"
    )
    parser.add_argument(
        "--extent_y", nargs=2, type=float, default=[1e0, 1e2],
        metavar=("Y_MIN", "Y_MAX"),
        help="Y-axis range (default: 1e0 1e2)"
    )
    args = parser.parse_args()

    if not args.est:
        parser.error("At least one --est eta n sigmaY is required.")

    params = [Param(e[0], e[1], e[2]) for e in args.est]
    flowPoints(args.file, params, args.out, args.extent_y)


if __name__ == '__main__':
    main()