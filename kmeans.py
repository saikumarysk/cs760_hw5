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
    
    return data, label

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

class kMeans:
    
    def __init__(self, data, x, y, max_iter):
        self.data = data
        self.x = x
        self.y = y
        self.max_iter = max_iter
        self.centroids = self.find_centroids()
    
    def classify(self):
        prev_centroids = None
        iter = 0
        while prev_centroids != self.centroids and iter <= self.max_iter:
            sorted_points = [[], [], []]
            for i in range(300):
                dists = [euclidean_distance(self.data[i], self.centroids[j]) for j in range(3)]
                cluster = np.argmin(dists)
                sorted_points[cluster].append(self.data[i])
            
            prev_centroids = self.centroids
            self.centroids = self.get_new_centroids(sorted_points)
            iter += 1
            print(f'Iteration {iter}/{self.max_iter} Done!', end='\r')
    
    def evaluate(self, x):
        dists = [euclidean_distance(x, self.centroids[j]) for j in range(3)]
        cluster = np.argmin(dists)
        return cluster
    
    def get_new_centroids(self, sorted_points):
        cluster_1, cluster_2, cluster_3 = sorted_points

        cum_sum_x, cum_sum_y = 0, 0
        for i in range(len(cluster_1)):
            cum_sum_x += cluster_1[i][0]
            cum_sum_y += cluster_1[i][1]
        
        c_1 = (cum_sum_x/len(cluster_1), cum_sum_y/len(cluster_1))

        cum_sum_x, cum_sum_y = 0, 0
        for i in range(len(cluster_2)):
            cum_sum_x += cluster_2[i][0]
            cum_sum_y += cluster_2[i][1]
        
        c_2 = (cum_sum_x/len(cluster_2), cum_sum_y/len(cluster_2))

        cum_sum_x, cum_sum_y = 0, 0
        for i in range(len(cluster_3)):
            cum_sum_x += cluster_3[i][0]
            cum_sum_y += cluster_3[i][1]
        
        c_3 = (cum_sum_x/len(cluster_3), cum_sum_y/len(cluster_3))

        return [c_1, c_2, c_3]

    def find_centroids(self):
        c_1 = np.random.choice(300)
        c_1 = self.data[c_1]

        d = [euclidean_distance(data[i], c_1)**2 for i in range(300)]
        sum_ = sum(d)
        p = [dist/sum_ for dist in d]
        c_2 = np.random.choice(300, 1, True, p)
        c_2 = self.data[c_2[0]]

        d = [min(euclidean_distance(data[i], c_1), euclidean_distance(data[i], c_2))**2 for i in range(300)]
        sum_ = sum(d)
        p = [dist/sum_ for dist in d]
        c_3 = np.random.choice(300, 1, True, p)
        c_3 = self.data[c_3[0]]

        return [c_1, c_2, c_3]

    def objective(self, pred):
        return sum([euclidean_distance(self.data[i], self.centroids[pred[i]-1])**2 for i in range(len(self.data))])

if __name__ == '__main__':
    files = ['data_0.5.txt', 'data_1.txt', 'data_2.txt', 'data_4.txt', 'data_8.txt']

    for file in files:
        print('Filename:', file)
        data, label = read_from_file(file)
        x, y = [i[0] for i in data], [i[1] for i in data]
        print('Finding Centroids')
        cluster = kMeans(data, x, y, 300)
        print('Centroid Finding Done!')
        cluster.classify()
        correct = 0
        pred = []
        for i, point in enumerate(data):
            pred.append(cluster.evaluate(point) + 1)
        
        print('Clustering Objective -', cluster.objective(pred))
        print('Clustering Accuracy -', get_max_accuracy(pred, label))
        print("#####################################################")