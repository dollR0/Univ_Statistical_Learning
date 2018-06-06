# coding: UTF-8

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from scipy import linalg


def preprocess(X):
    """
    preprocess for projection pursuit
    (centering and sphering)
    arg:
        X: input matrix
    return:
        X: preprocessed matrix
    """
    N = X.shape[1]
    H = np.identity(N) - np.ones((N, N)) / N
    xh = np.matmul(X, H)
    hxt = np.matmul(H, X.T)
    Xt = np.matmul(linalg.sqrtm(np.linalg.pinv((np.matmul(xh, hxt) / N))), xh)
    return Xt


def projection_pursuit(Xt, g, dg):
    """
    projection pursuit
    arg:
        Xt: input matrix(preprocessed)
        g: criterion
        dg: derivative of criterion g
    return:
        phi: vectors of non-Gaussian directions
    """

    # settings
    epsilon = 1e-8
    dim = Xt.shape[0]
    N = Xt.shape[1]
    phi = None

    for d in range(dim):
        delta = epsilon + 1

        # initialize b
        b = np.random.rand(dim)
        b = b / np.linalg.norm(b)

        while np.abs(delta) > epsilon:
            b_prev = np.copy(b)

            # step
            b = b * np.sum(dg(np.sum(b.reshape((-1, 1)) * Xt, axis=0))) / N - \
                np.sum(Xt * (g(np.sum(b.reshape((-1, 1)) * Xt, axis=0))), axis=1) / N

            # orthonormalization
            if phi is not None:
                b = b - np.sum(np.sum(b.reshape((1, -1)) * phi,
                                      axis=1).reshape((-1, 1)) * phi, axis=0)

            # normalization
            b = b / np.linalg.norm(b)

            # calculate cos simularity
            delta = np.abs(np.sum(b_prev * b) /
                           (np.linalg.norm(b_prev) * np.linalg.norm(b))) - 1

        if phi is None:
            phi = np.array([b])
        else:
            phi = np.vstack((phi, b))
    return phi


def main():
    # settings
    N = 10000
    dim = 2

    # generate data
    # data = [[data1], [data2], ...]
    Xorig = np.random.rand(N)
    for d in range(dim - 1):
        Xorig = np.vstack((Xorig, np.random.rand(N)))
    W = np.array([[0.2, 0.8], [0.6, 0.4]])
    X = np.matmul(W, Xorig)

    # preprocess
    Xt = preprocess(X)

    # projection pursuit
    phi = projection_pursuit(Xt, np.tanh, lambda x: 1 - np.tanh(x)**2)

    # plot for report
    sns.set()
    sns.set_style('whitegrid')
    plt.rcParams["savefig.dpi"] = 300
    plt.title("original data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(Xorig[0], Xorig[1], alpha=0.3, s=10, c="violet")
    plt.savefig("original_data.png")
    plt.close()

    plt.title("mixed data(preprocessed) with non-Gaussian directions")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(Xt[0], Xt[1], alpha=0.3, s=10, c="violet")
    for b in phi:
        v = b * np.arange(-2, 2, 0.001).reshape(-1, 1)
        v = v.T
        plt.plot(v[0], v[1], c="crimson")

    plt.savefig("pp_result.png")
    plt.close()


if __name__ == '__main__':
    main()
