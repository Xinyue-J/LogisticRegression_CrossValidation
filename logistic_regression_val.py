import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer


def readfile(filename):
    csv_data = pd.read_csv(filename, header=None)
    return csv_data


def bi(train_feature, test_feature):
    names = train_feature.columns
    std = Binarizer().fit(train_feature)
    train = pd.DataFrame(std.transform(train_feature), columns=names)
    test = pd.DataFrame(std.transform(test_feature), columns=names)

    return train, test


def log_tf(train_feature, test_feature):
    train = train_feature.apply(lambda x: np.log(x + 0.1))
    test = test_feature.apply(lambda x: np.log(x + 0.1))
    return train, test


def std_(train_feature, test_feature):
    names = train_feature.columns

    std = StandardScaler().fit(train_feature)
    std_train_feature = pd.DataFrame(std.transform(train_feature), columns=names)
    std_test_feature = pd.DataFrame(std.transform(test_feature), columns=names)

    return std_train_feature, std_test_feature


def cross_val(x, y, c):
    acc_cross = []
    clf = LogisticRegression(C=c, max_iter=10 ** 6)

    sfolder = StratifiedKFold(n_splits=5, shuffle=True)
    for tr_i, val_i in sfolder.split(x, y):
        train = pd.DataFrame(x, tr_i)
        val = pd.DataFrame(x, val_i)
        tr_label = pd.DataFrame(y, tr_i)
        val_label = pd.DataFrame(y, val_i)

        num = tr_label.shape[0]
        tr_l = np.array(tr_label).reshape(num)
        num = val_label.shape[0]
        val_l = np.array(val_label).reshape(num)

        std_train, std_val = std_(train, val)  # preprocessing
        clf.fit(std_train, tr_l)  # label should be 1-d array
        acc_cross.append(clf.score(std_val, val_l))
    mean = np.mean(acc_cross)
    std = np.std(acc_cross)

    return mean, std


def find_best(ACC, DEV, Cs):
    # inputs are all np.array
    index_maxmean = np.argwhere(ACC == np.max(ACC))
    i1 = index_maxmean[:, 0]

    if len(i1) == 1 and DEV[i1[0]] == np.min(DEV):  # clear choice
        best_c = Cs[i1[0]]
        mean = ACC[i1[0]]
        deviation = DEV[i1[0]]
    else:  # max mean -> min dev
        pair = []
        for index in range(len(i1)):
            pair.append(DEV[i1[index]])
            # pair contains the dev with max mean, then we find the min dev among this
        best_index = np.argwhere(pair == np.min(pair))  # position in pair

        best_c = Cs[i1[best_index[0][0]]]
        mean = ACC[i1[best_index[0][0]]]
        deviation = DEV[i1[best_index[0][0]]]

    return best_c, mean, deviation


if __name__ == '__main__':
    xtrain = readfile('H2W3 data files/Xtrain.csv')
    ytrain = readfile('H2W3 data files/Ytrain.csv')
    xtest = readfile('H2W3 data files/Xtest.csv')
    ytest = readfile('H2W3 data files/Ytest.csv')

    # print(xtrain.shape[0]) # 3065 rows: # of data points
    # print(xtrain.shape[1]) # 57 cols: # of features, we only need feature of 1 to 55
    # xtrain = xtrain.drop(columns=[55, 56])
    # xtest = xtest.drop(columns=[55, 56])

    # choose parameter C using cross validation
    mean_list = []
    std_list = []
    Cs = np.logspace(-5, 5, num=50)

    for c in Cs:
        mean, std = cross_val(xtrain, ytrain, c)
        mean_list.append(mean)
        std_list.append(std)

    print('log transform:')
    best_c, mean, deviation = find_best(np.array(mean_list), np.array(std_list), Cs)
    print('mean_error:', 1 - mean)
    print('standard deviation:', deviation)
    print('best_c:', best_c)

    # use the best parameter for the model, and calculate its accuracy
    clf = LogisticRegression(C=best_c, max_iter=10 ** 6)

    num = ytrain.shape[0]
    ytrain_l = np.array(ytrain).reshape(num)
    num = ytest.shape[0]
    ytest_l = np.array(ytest).reshape(num)

    p_xtrain, p_xtest = std_(xtrain, xtest)  # preprocessing
    clf.fit(p_xtrain, ytrain_l)

    print('error_train:', 1 - clf.score(p_xtrain, ytrain_l))
    print('error_test:', 1 - clf.score(p_xtest, ytest_l))
