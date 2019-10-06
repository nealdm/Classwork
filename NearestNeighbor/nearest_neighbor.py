# nearest_neighbor.py
"""Nearest Neighbor Search.
<Neal Munson>
<10/24/2018>

"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree as skdt
from scipy import stats as st
from matplotlib import pyplot as plt

def exhaustive_search(X, z):
    """Solves the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    # vect represents the closest vector in X to z
    vect = np.argmin(la.norm(X-z, axis = 1))
    # dist represents the distance from vect to z
    dist = la.norm(X[vect] - z)
    return X[vect], dist


class KDTNode:
    """A node class for k-d trees. Contains an array value, a
    reference to the pivot value, and references to two child nodes.
    """
    def __init__(self, x):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        if type(x) is not np.ndarray:
            raise TypeError("Data type must be an np.ndarray")
        self.value = x
        self.pivot = None        # This node's pivot value
        self.left = None        # self.left.value @ pivot < self.value @ pivot
        self.right = None       # self.value @ pivot < self.right.value @ pivot

class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        # set n to be a node with the desired Data
        n = KDTNode(data)
        # if tree is empty, create a new node, set its pivot to 0, and make it
        # the root.  Also set the k attribute to the length of x.
        if self.root is None:
            n.pivot = 0
            self.root = n
            self.k = len(data)
            return
        # if length of data is not equal to k, we raise a ValueError
        if len(data) != self.k:
            raise ValueError("length of vector needs to be {} to match the root length".format(self.k))

        # if tree is nonempty, find the node that should become the parent of n.
        # determine if n is the parent's left or right child, link it accordingly
        # and set the pivot value of n.  Raise a ValueError if the tree contians n.
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if np.allclose(data, current.value):               # this value is in the tree
                raise ValueError("There is already a node in this k-d tree with this value.")
            # if this node is a leaf, return it
            if current.left is None and current.right is None:
                return current
            if data[current.pivot] < current.value[current.pivot]:# Recursively search left.
                if current.left is None:            # If you need to branch to the left
                    return current
                else:
                    return _step(current.left)
            else:                                   # Recursively search right.
                if current.right is None:            # If you need to branch to the right
                    return current
                else:
                    return _step(current.right)

        parent = _step(self.root)
        # if n is to the left
        if data[parent.pivot] < parent.value[parent.pivot]:
            parent.left = n
        else:
            parent.right = n
        if parent.pivot == self.k-1:
            n.pivot = 0
        else:
            n.pivot = parent.pivot + 1
        return


    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest, d):
            if current is None:
                return nearest, d
            x = current.value
            i = current.pivot
            dist = la.norm(x-z)
            if dist < d:
                nearest = current
                d = dist
            if z[i] < x[i]:  # search to the left
                nearest, d = KDSearch(current.left, nearest, d)
                if z[i] + d >= x[i]: # search right if needed
                    nearest, d = KDSearch(current.right, nearest, d)
            else:  # search to the right
                nearest, d = KDSearch(current.right, nearest, d)
                if z[i] - d <= x[i]: # search left if needed
                    nearest, d = KDSearch(current.left, nearest, d)
            return nearest, d
        # start the recursive searching process
        node, d = KDSearch(self.root, self.root, la.norm(self.root.value-z) )
        return node.value, d


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


class KNeighborsClassifier:
    """A k-Neighbors classifier for solving nearest neighbor problems.

    Attributes:
        n (int): the number of neighbors included in a vote.
        tree (KDTree): the produced KDTree from which to search
        labels (np.ndarray): the values of the labels for each element of the tree

    """
    def __init__(self, n_neighbors):
        """Accepts an integer n_neighbors, the number of neighbors to include in the vote."""
        self.n = n_neighbors # should be straightforward enough.
        self.tree = None # these two are initialized in 'fit()'
        self.labels = None

    def fit(self, X, y):
        '''takes an m * k numpy array X (training set) and
        1-dim numpy array y (training labels) with m entries'''
        self.tree = skdt(X) # creates a KDTree with each of the m rows of X
        self.labels = y

    def predict(self, z):
        '''accept 1-dim Numpy array z with k entries.  Query tree for n_neighbors elements
        of X that are nearest to z.

        Returns: the most common label.
        If there is a tie, we choose the alphanumerically smallest label.'''
        distances, indices = self.tree.query(z, k = self.n) # finds the k-nearest neighbors
        if self.n == 1:
            return self.labels[indices]
        mode,count = st.mode(self.labels[list(indices)]) # finds the mode of the lables from
                                                         # the k-nearest neighbors
        return mode[0]



def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # imports the Data
    data = np.load(filename)
    X_train = data["X_train"].astype(np.float) # Training data
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float) # Training data
    y_test = data["y_test"]

    # Runs the test, and returns the classification accuracy as a float
    knc = KNeighborsClassifier(n_neighbors)
    knc.fit(X_train,y_train)
    values = [knc.predict(x) for x in X_test]
    samearray = values - y_test
    number_correct = sum(sum([samearray == 0])) # this gives the total number of
                                                # indices where the 'values' and 'y_test'
                                                # arrays had the same value
    return int(number_correct)/int(len(values)) # here the int() function is used to
                                                # make the percent come out cleanly

    # run the following to see what one of the numbers from this
    # data set looks like 'on paper'
    # plt.imshow(X_test[0].reshape((28,28)),cmap="gray")
    # plt.show()
