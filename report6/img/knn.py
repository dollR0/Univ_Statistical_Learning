# coding: UTF-8
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(1)


class KNNClassifier():
    def __init__(self, X, y, k):
        self.k = k
        self.X = X
        self.y = y

    def classify(self, xp):
        if xp.ndim == 1:
            top_k = collections.Counter(self.find_top_k(xp))
            y = top_k.most_common()[0][0]
        else:
            y = np.array([])
            for x in xp:
                top_k = collections.Counter(self.find_top_k(x))
                y = np.append(y, top_k.most_common()[0][0])
        return y

    def find_top_k(self, xp):
        temp = self.X - xp
        dist = np.sum(temp**2, axis=1)
        return self.y[np.argsort(dist)][:self.k]


def main():
    # load data
    for i in range(10):
        df_train = pd.read_csv(
            "./digit/digit_train{}.csv".format(i), header=None)
        df_test = pd.read_csv(
            "./digit/digit_test{}.csv".format(i), header=None)
        if i == 0:
            X = df_train.values
            y = np.zeros(len(X), dtype=int)
            X_val = df_test.values
            y_val = np.zeros(len(X_val), dtype=int)
        else:
            X = np.vstack((X, df_train.values))
            y = np.hstack(
                (y, np.ones(len(df_train.values), dtype=int) * i))
            X_val = np.vstack((X_val, df_test.values))
            y_val = np.hstack(
                (y_val, np.ones(len(df_test.values), dtype=int) * i))

    # randomize X, y
    N = len(X)
    rand_ind = np.random.permutation(range(N))
    X = np.array([X[r] for r in rand_ind])
    y = np.array([y[r] for r in rand_ind])

    losses = []
    k_best = 1
    n_split = 10
    test_size = N // n_split

    print("{} split cross validation".format(n_split))
    print("total data : {}".format(N))
    for k in range(1, 10):
        loss_average = 0
        for i in range(n_split):
            # train/test split
            X_test = X[test_size * i:test_size * (i + 1)]
            y_test = y[test_size * i:test_size * (i + 1)]
            X_train = np.vstack((X[:test_size * i], X[test_size * (i + 1):]))
            y_train = np.hstack((y[:test_size * i], y[test_size * (i + 1):]))

            # classify
            classifier = KNNClassifier(X_train, y_train, k)
            y_pred = classifier.classify(X_test)

            # calculate loss
            loss = np.sum(y_pred != y_test) / len(y_test)
            loss_average += loss
        loss_average /= n_split
        losses.append(loss_average)

        print("k = {}, test loss = {}".format(k, loss_average))
        if loss_average < losses[k_best-1]:
            k_best = k
        loss_prev = loss_average

    # validation
    print("train size:", len(X))
    print("validation size:", len(X_val))
    classifier = KNNClassifier(X, y, k_best)
    y_pred = classifier.classify(X_val)
    print("k=", k_best)
    print("Validation accuacy:", np.sum(y_pred == y_val) / len(y_val))

    # plot
    sns.set_style('whitegrid')
    plt.rcParams["savefig.dpi"] = 300
    plt.xlabel("k")
    plt.ylabel("loss")
    plt.plot(np.arange(1, 10, 1), losses, color="#f781bf")
    plt.savefig("knn_loss.png")
    plt.close()


if __name__ == '__main__':
    main()
