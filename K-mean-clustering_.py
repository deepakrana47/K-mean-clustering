import pickle, json
import numpy as np

def eucl_dist(a, b, f_size):
    esum = 0
    for i in range(f_size):
        if a[i] and b[i]:
            esum += (a[i] - b[i])**2
    return np.sqrt(esum)

def get_diff(c1, c2):
    row, col = c1.shape
    csum = 0
    for i in row:
        csum += np.linalg.norm(c1[i], c2[i])
    return csum

def get_min(X, f_size):
    vmin = np.zeros(f_size)
    for x in X:
        for j in x:
            j = int(j)
            if vmin[j] > x[j]:
                vmin[j] = x[j]
    return vmin

def get_max(X, f_size):
    vmax = np.zeros(f_size)
    for x in X:
        for j in x:
            j = int(j)
            if vmax[j] < x[j]:
                vmax[j] = x[j]
    return vmax

def get_mean(points, f_size):
    cmean = np.zeros(f_size)
    for i in points:
        for j in range(f_size):
            if j in points[i]:
                cmean += points[i][j]
    cmean = cmean / len(points)
    return cmean

def k_mean(x, d_size, f_size, k):

    #initalizing cluster variable
    cluster = np.zeros(d_size)

    # calculation min and max for every dimension of data
    minv = get_min(x, f_size)
    maxv = get_max(x, f_size)

    # for k in range(2,11):
    error = 0

    # initalizing centroids of k clusters
    center = np.zeros((k, f_size))
    for i in range(k):
        for j in range(f_size):
            center[i,j] = minv[j] + np.random.random() * (maxv[j] - minv[j])

    # assigining zeros to old centroids value
    center_old = np.zeros(center.shape)

    # initial error
    err = get_diff(center, center_old)

    while err != 0:

        # calculatin distance of data points from centroids and assiging min distance cluster centroid as data point cluster
        for i in range(len(x)):
            distances = []
            for c in range(center.shape[0]):
                distances.append(eucl_dist(x[i], center[c], f_size))
            clust = np.argmin(distances)
            cluster[i] = clust

        # changing old centroids value
        center_old = np.copy(center)

        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [x[j] for j in range(len(x)) if cluster[j] == i]
            if points:
                center[i] = get_mean(points, f_size)

        # calculation difference between new centroid and old centroid values
        err = get_diff(center, center_old)

    # calculation total difference between cluster centroids and cluster data points
    for i in range(k):

        d = [eucl_dist(x[j],center[i],f_size) for j in range(len(x)) if cluster[j] == i]
        error += np.sum(d)

    # counting data points in all clusters
    count = {key: 0.0 for key in range(k)}
    for i in range(len(x)):
        count[cluster[i]] += 1

    # displaying cluster number, average distance between centroids and data points and cluster count
    print(k, error/len(x), count)

    return cluster

if __name__ == '__main__':

    # loading dataset of form [[data1],[data2], ....]
    # inp = pickle.load(open('test.pickle', 'rb'))
    fd = open("tfidf_vec.json",'r')
    inp = []
    for line in fd:
        inp.append(json.loads(line))
    p = json.load(open("term_2_index.json","r"))
    # x = np.array([i[0] for i in inp])

    # return cluster number for every data
    cluster = k_mean(inp, len(inp), len(p), 20)
