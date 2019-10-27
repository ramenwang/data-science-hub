from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 
import statistics as stats
import matplotlib.pyplot as plt
import itertools

class knn():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def euclidean_dist(x1, x2):
        '''
        x1 and x2 should be n-dimension vectors
        '''
        t = x1 - x2
        return np.sqrt(np.dot(t, np.transpose(t)))

    def knn_classify(self, x_input, k):
        if k % 2 != 1:
            return 'Error!'
        distances = [self.euclidean_dist(x_input, x2) for x2 in self.x]
        kn_idxs = np.argsort(distances)[:k]
        return stats.mode(self.y[kn_idxs])

    def predict(self, x_input, k):
        return [self.knn_classify(x, k) for x in x_input]



if __name__ == "__main__":
    # read data from sklearn
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=42)
    # set up hyperparameters
    k = 5
    knn_classifier = knn(x_train, y_train)
    prediction = knn_classifier.predict(x_test, k)
    accuracy = np.sum(y_test == prediction)/len(y_test)
    print(f"KNN with k = {k} has an accuracy of {accuracy*100:.3f}%")

    # visualize different k setup
    n_points = 100
    x1 = np.random.multivariate_normal([1,50],[[1,0],[0,10]], n_points)
    x2 = np.random.multivariate_normal([2,50],[[1,0],[0,10]], n_points)
    x_input = np.concatenate([x1,x2])
    y_input = np.array([0]*n_points + [1]*n_points)
    print("x has the shape of ", x_input.shape)
    print("y has the shape of ", y_input.shape)

    prediction_list = []
    k_list = [3,5,7,9,11,13,15,17,19]
    knn_classifier = knn(x_input, y_input)
    for k_i in k_list:
        prediction_list.append(knn_classifier.predict(x_input, k_i))

    x_axis = [x_input[:,0].min()-1, x_input[:,0].max()+1]
    y_axis = [x_input[:,1].min()-1, x_input[:,1].max()+1]

    xx, yy = np.meshgrid(np.arange(x_axis[0], x_axis[1], 0.1),\
        np.arange(y_axis[0], y_axis[1], 0.1))

    f, axarr = plt.subplots(3,3, sharex='col', sharey='row', figsize = (15, 12))
    
    for idx, predict, tt in zip(itertools.product([0,1,2],[0,1,2]),\
        prediction_list, [f'KNN (k = {k_i})' for k_i in k_list]):

        z = np.c_[xx.ravel(), yy.ravel()]
        z = z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, z, alpha = 0.4)
        axarr[idx[0], idx[1]].scatter(x_input[:,0], x_input[:,1],\
            c=y_input, s = 20, edgecolor='k')
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()





