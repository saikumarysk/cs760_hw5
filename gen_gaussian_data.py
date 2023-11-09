from numpy.random import multivariate_normal
import numpy as np

def write_to_file(x, y, label, alpha):
    with open('data_'+str(alpha)+'.txt', 'a') as wfile:
        for i in range(len(x)):
            wfile.write(str(x[i])+' '+str(y[i])+' '+str(label)+'\n')

if __name__ == '__main__':
    variables = [(np.array([-1, -1]), np.array([2, 0.5, 0.5, 1]).reshape((2, 2))), (np.array([1, -1]), np.array([1, -0.5, -0.5, 2]).reshape((2, 2))), (np.array([0, 1]), np.array([1, 0, 0, 2]).reshape((2, 2)))]

    for alpha in [0.5, 1, 2, 4, 8]:
        count = 0
        for mean, covariance in variables:
            cv = alpha * covariance
            #print(mean, cv)
            s = multivariate_normal(mean, cv, 100)
            x, y = [j[0] for j in s], [j[1] for j in s]
            write_to_file(x, y, count+1, alpha)
            count += 1