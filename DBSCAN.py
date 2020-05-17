# Reference: https://github.com/chrisjmccormick/dbscan
from scipy.spatial.distance import euclidean

def DBSCAN(dataset, epsilon, min_points):
    '''
    Cluster dataset based on 'epsilon' distance and 'min_points' using the DBSCAN Algorithm
    0 - cluster not assigned yet
    -1 - Noise (doesn't belong to any cluster)
    '''
    # initialize the labels
    labels = [0] * len(dataset)

    # Initialize C - ID of current cluster
    C = 0

    # seed point - point which has not been considered yet
    # first we iterate through every seed point and once a valid seed point is found, we expand the cluster
    for P in range(len(dataset)):
         
         # if a point has already been allocated to a cluster, we don't consider it
         if labels[P] != 0:
             continue

        # find P's neighbouring points within 'epsion' distance
        neighbour_points = regionQuery(P, D, epsilon)

        # if neighbouring points is less than min_points, we label it as a noise
        # else we grow a cluster from the seed point
        if len(neighbour_points) < min_points:
            labels[P] = -1
        else:
            C += 1
            growCluster(dataset, labels, P, neighbour_points, C, epsilon, min_points)

    return labels

def growCluster(dataset, labels, P, neighbour_points, C, epsilon, min_points):
    '''Grow a new cluster with label `C` from the seed point `P`.'''
    # assign label to seed point
    labels[P] = C

    # iterate through the neighbour points
    i = 0
    while i < len(neighbour_points):

        # get neighbour point
        pn = neighbour_points[i]

        # if pn was assigned -1, it means it's a leaf
        if labels[pn] == -1:
            labels[pn] = C

        # if it was not assigned anything, we assign it label C
        elif labels[pn] == 0:
            labels[pn] = C

            # now we find it's neighbours
            pn_neighbours = regionQuery(pn, dataset, epsilon)

            # if the number of neighbours cross the threshold
            # it's a branch point and we add the neighbours to the queue
            if len(pn_neighbours) >= min_points:
                neighbour_points += pn_neighbours

        i += 1

def regionQuery(dataset, P, epsilon):
    neighbours = []
    for pn in range(len(dataset)):
        if euclidean(dataset[pn], dataset[P]) < epsilon:
            neighbours.append(pn)

    return neighbours