import numpy as np
import matplotlib.pyplot as plt

# Results
# Reconstruction Errors for 2D:
# ----------------------
# Buggy PCA - 38.396359916101076
# Demean PCA - 4.097135348936227
# Norm PCA - 9.014756571421726
# DRO PCA - 4.097135348936228

# Reconstruction Errors:
# ----------------------
# Buggy PCA - 13163.110154859782
# Demean PCA - 8259.74014907716
# Norm PCA - 8268.5467161029
# DRO PCA - 8259.74014907716

def read_from_file(filename):
    data = []
    with open(filename, 'r') as rfile:
        for line in rfile:
            data.append(list(map(float, line.strip().split(','))))
    
    return np.array(data).reshape((len(data), len(data[0])))

def buggyPCA(X, d):
    n, D = X.shape
    U, S, VT = np.linalg.svd(X)

    V = VT.T
    V_d = V[:, :d]
    Z = np.matmul(X, V_d)
    Y = np.matmul(Z, V_d.T)
    return Y

def demeanPCA(X, d):
    n, D = X.shape
    mu = np.mean(X, axis=0)
    X = [x_i - mu for x_i in X]
    X = np.array(X).reshape((n, D))

    Y = buggyPCA(X, d)

    Y = [y_i + mu for y_i in Y]

    Y = np.array(Y).reshape((n, D))

    return Y

def normPCA(X, d):
    n, D = X.shape

    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X = [(x_i - mu)/std for x_i in X]
    X = np.array(X).reshape((n, D))
    Y = buggyPCA(X, d)
    Y = [y_i*std + mu for y_i in Y]
    Y = np.array(Y).reshape((n, D))

    return Y

def DRO(X, d):
    n, D = X.shape

    b = np.mean(X, axis=0).reshape((D, 1))
    one_n = np.ones((n, 1))
    U, S, VT = np.linalg.svd(X - np.matmul(one_n, b.T))

    V = VT.T
    S = np.diag(S)

    V_d = V[:, :d]
    U_d = U[:, :d]
    S_d = S[:d, :d]

    Z = np.sqrt(n) * U_d
    A = (1/np.sqrt(n)) * np.matmul(V_d, S_d)

    Y = np.matmul(Z, A.T) + np.matmul(one_n, b.T)

    return Y

if __name__ == '__main__':

    # X = read_from_file('data/data1000D.csv')
    # U, S, VT = np.linalg.svd(X)
    # S = S[1:100]

    # plt.scatter(list(range(1, len(S)+1)), S, color='green')
    # plt.show()

    X = read_from_file('data/data1000D.csv')
    Y_buggy = buggyPCA(X, 30)
    Y_demean = demeanPCA(X, 30)
    Y_norm = normPCA(X, 30)
    Y_dro = DRO(X, 30)

    # x, y = [x_i[0] for x_i in X], [x_i[1] for x_i in X]
    # plt.scatter(x, y, edgecolors='blue', facecolors='none', marker='o')
    # x, y = [y_i[0] for y_i in Y_buggy], [y_i[1] for y_i in Y_buggy]
    # plt.scatter(x, y, color='red', marker='x')
    # plt.show()

    # x, y = [x_i[0] for x_i in X], [x_i[1] for x_i in X]
    # plt.scatter(x, y, edgecolors='blue', facecolors='none', marker='o')
    # x, y = [y_i[0] for y_i in Y_demean], [y_i[1] for y_i in Y_demean]
    # plt.scatter(x, y, color='red', marker='x')
    # plt.show()

    # x, y = [x_i[0] for x_i in X], [x_i[1] for x_i in X]
    # plt.scatter(x, y, edgecolors='blue', facecolors='none', marker='o')
    # x, y = [y_i[0] for y_i in Y_norm], [y_i[1] for y_i in Y_norm]
    # plt.scatter(x, y, color='red', marker='x')
    # plt.show()

    # x, y = [x_i[0] for x_i in X], [x_i[1] for x_i in X]
    # plt.scatter(x, y, edgecolors='blue', facecolors='none', marker='o')
    # x, y = [y_i[0] for y_i in Y_dro], [y_i[1] for y_i in Y_dro]
    # plt.scatter(x, y, color='red', marker='x')
    # plt.show()

    T_buggy = X - Y_buggy
    T_demean = X - Y_demean
    T_norm = X - Y_norm
    T_dro = X - Y_dro

    print('Reconstruction Errors:')
    print('----------------------')

    T_buggy = [np.linalg.norm(t_i, ord=2) for t_i in T_buggy]
    print('Buggy PCA -', sum(T_buggy))
    T_demean = [np.linalg.norm(t_i, ord=2) for t_i in T_demean]
    print('Demean PCA -', sum(T_demean))
    T_norm = [np.linalg.norm(t_i, ord=2) for t_i in T_norm]
    print('Norm PCA -', sum(T_norm))
    T_dro = [np.linalg.norm(t_i, ord=2) for t_i in T_dro]
    print('DRO PCA -', sum(T_dro))