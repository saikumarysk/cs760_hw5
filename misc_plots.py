import matplotlib.pyplot as plt

sigma = [0.5, 1, 2, 4, 8]
# objectives = [303.029643910935, 558.9918988336558, 904.9963963305394, 1603.2153274028226, 3587.3489027017526]
# accuracies = [0.78, 0.7366666666666667, 0.6166666666666667, 0.59, 0.53]

objectives = [310.3459641366143, 558.9918988336558, 904.9963963305394, 1603.2153274028226, 3587.3489027017526]
accuracies = [0.79, 0.7366666666666667, 0.6166666666666667, 0.59, 0.53]

#plt.plot(sigma, objectives, label='Clustering Objectives', color='red')
plt.plot(sigma, accuracies, label='Clustering Accuracies', color='blue')
plt.legend()
plt.title('Clustering Accuracy vs. Sigma')
plt.show()