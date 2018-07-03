# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def my_rand(n):
    x = np.zeros((n, 1))
    u = np.random.rand(n, 1)
    flag = u < 1 / 8
    x[flag] = np.sqrt(8 * u[flag])
    flag = np.logical_and(u >= 1 / 8, u < 1 / 4)
    x[flag] = 2 - np.sqrt(2 - 8 * u[flag])
    flag = np.logical_and(u >= 1 / 4, u < 1 / 2)
    x[flag] = 1 + 4 * u[flag]
    flag = np.logical_and(u >= 1 / 2, u < 3 / 4)
    x[flag] = 3 + np.sqrt(4 * u[flag] - 2)
    flag = np.logical_and(u >= 3 / 4, u <= 1)
    x[flag] = 5 - np.sqrt(4 - 4 * u[flag])
    return x


def gauss_kernel(x):
    d = x.shape[1]
    return np.exp(-(x**2) / 2) / ((2 * np.pi)**(d / 2))


def kde(kernel, X, h, xs):
    n, d = X.shape
    return np.array([np.sum(kernel((x - X) / h)) for x in xs]) / (n * h**d)


def main():
    N = 1000
    n_split = 5
    test_size = N // n_split
    X = my_rand(N)

    hs = np.arange(1e-2, 0.5, 0.01)
    LCVs = []
    for h in hs:
        lcv = 0
        for i in range(n_split):
            X_test = X[test_size * i:test_size * (i + 1)]
            X_train = np.vstack((X[:test_size * i], X[test_size * (i + 1):]))
            p = kde(gauss_kernel, X_train, h, X_test)
            lcv += np.sum(np.log(p)) / test_size
        lcv /= n_split
        LCVs.append(lcv)
        print("h={}, LCV={}".format(h, lcv))

    h = hs[np.argmax(LCVs)]
    print("use h = {}.".format(h))
    x_plot = np.arange(0, 5, 0.01)
    p = kde(gauss_kernel, X, h, x_plot)
    # plot
    sns.set_style('whitegrid')
    plt.rcParams["savefig.dpi"] = 300
    plt.xlabel("h")
    plt.ylabel("LCV")
    plt.plot(hs, LCVs, color="#f781bf")
    plt.savefig("LCV.png")
    plt.close()
    plt.xlabel("x")
    plt.ylabel("p")
    plt.hist(X[:, 0], bins=30, alpha=0.5, density=1, color="#f781bf")
    plt.plot(x_plot, p, color="#e41a1c")
    plt.savefig("kde.png")
    plt.close()


if __name__ == '__main__':
    main()
