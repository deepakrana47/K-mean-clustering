import pickle
import numpy as np

def eucl_dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)

def k_mean(x, k):

    #initalizing cluster variable
    cluster = np.zeros(x.shape[0])

    # calculation min and max for every dimension of data
    minv = np.min(x,axis=0)
    maxv = np.max(x,axis=0)

    # for k in range(2,11):
    error = 0

    # initalizing centroids of k clusters
    center = np.zeros((k, x.shape[1]))
    for i in range(k):
        for j in x.shape[1]:
            center[i,j] = np.random.randint(minv, maxv)

    # assigining zeros to old centroids value
    center_old = np.zeros(center.shape)

    # initial error
    err = eucl_dist(center, center_old, None)

    while err != 0:

        # calculatin distance of data points from centroids and assiging min distance cluster centroid as data point cluster
        for i in range(len(x)):
            distances = eucl_dist(x[i], center)
            clust = np.argmin(distances)
            cluster[i] = clust

        # changing old centroids value
        center_old = np.copy(center)

        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [x[j] for j in range(len(x)) if cluster[j] == i]
            if points:
                center[i] = np.mean(points, axis=0)

        # calculation difference between new centroid and old centroid values
        err = eucl_dist(center, center_old, None)

    # calculation total difference between cluster centroids and cluster data points
    for i in range(k):
        d = [eucl_dist(x[j],center[i],None) for j in range(len(x)) if cluster[j] == i]
        error += np.sum(d)

    # counting data points in all clusters
    count = {key: 0.0 for key in range(k)}
    for i in range(len(x)):
        count[cluster[i]] += 1

    # displaying cluster number, average distance between centroids and data points and cluster count
    print k, error/len(x), count

    return cluster

if __name__ == '__main__':

    # loading dataset of form [[data1],[data2], ....]
    # inp = pickle.load(open('/media/zero/41FF48D81730BD9B/all-the-news/4/temp_tokenize_dec_dataset/test.pickle', 'rb'))
    x = np.array([i[0] for i in inp])

    # return cluster number for every data
    cluster = k_mean(x)
