import numpy as np

def read_from_file(filename):
    data = []
    label = []
    with open(filename, 'r') as rfile:
        for line in rfile:
            coord = list(map(float, line.strip().split(' ')))
            coord[-1] = int(coord[-1])
            label.append(coord[-1])
            data.append(tuple(coord[:-1]))
    
    return np.array(data).reshape((300, 2)), label

def euclidean_distance(x_1, x_2):
    return np.sqrt((x_1[0]-x_2[0])**2 + (x_1[1] - x_2[1])**2)

def get_max_accuracy(pred, label):
    perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

    max_accuracy = 0
    for perm in perms:
        accuracy = get_accuracy(pred, label, perm)
        max_accuracy = max(accuracy, max_accuracy)
    
    return max_accuracy

def get_accuracy(pred, label, perm):
    correct = 0
    for i in range(len(pred)):
        if perm[pred[i]-1] == label[i]: correct += 1
    
    return correct / len(pred)

class Gaussian_Mixture:

    def __init__(self, data, max_iter):
        self.data = data
        self.n, self.D = data.shape
        self.max_iter = max_iter
    
    def classify(self):
        self.phi = [1/3, 1/3, 1/3]
        self.mu = [self.data[r, :] for r in np.random.randint(0, 300, size=3)] # Random centroids
        self.sigma = [np.cov(self.data.T) for _ in range(3)] # Identity covariance

        for _ in range(self.max_iter):
            self.expectation()
            self.maximization()
    
    def expectation(self):

        self.weights = np.zeros((self.n, 3))
        for i in range(self.n):
            nums = [self.phi[k]*self.multinormal(self.data[i], self.mu[k], self.sigma[k]) for k in range(3)]
            sum_ = sum(nums)
            self.weights[i] = np.array([n/sum_ for n in nums]).reshape((1, 3))
        
        self.N = np.sum(self.weights, axis=0)

    def maximization(self):
        self.mu = np.zeros((3, 2))
        for j in range(3):
            for i in range(self.n):
                self.mu[j] += self.weights[i][j] * self.data[i]
        
        self.mu = [self.mu[k]/self.N[k] for k in range(3)]

        self.sigma = [np.zeros((2, 2)) for _ in range(3)]
        for k in range(3):
            self.sigma[k] = np.cov(self.data.T, aweights=(self.weights[:, k]), ddof=0)
        
        self.sigma = [self.sigma[k]/self.N[k] for k in range(3)]

        self.phi = [self.N[k]/self.n for k in range(3)]

    def evaluate(self):
        probs = []
        for i in range(self.n):
            probs.append(np.argmax([self.multinormal(self.data[i], self.mu[j], self.sigma[j]) for j in range(3)]) + 1)
        
        return probs
    
    def multinormal(self, x, mu, Sigma):
        constant = 1/(2*np.pi)
        constant *= (np.linalg.det(Sigma))**(-0.5)

        exponent = -0.5
        exponent *= np.matmul((x - mu).T, np.matmul(np.linalg.inv(Sigma), (x - mu)))

        return constant * np.exp(exponent)

    def objective(self, pred):
        return sum([euclidean_distance(self.data[i], self.mu[pred[i]-1])**2 for i in range(len(self.data))])

if __name__ == '__main__':
    files = ['data_0.5.txt', 'data_1.txt', 'data_2.txt', 'data_4.txt', 'data_8.txt']

    for file in files:
        print('Filename:', file)
        data, label = read_from_file(file)
        cluster = Gaussian_Mixture(data, max_iter=100)
        print('Beginning Clustering!')
        cluster.classify()
        print('Done Clustering!')
        pred = cluster.evaluate()

        print('Clustering Accuracy:', get_max_accuracy(pred, label))
        print('Clustering Objective:', cluster.objective(pred))
        print('#####################################')