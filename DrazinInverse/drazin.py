# drazin.py
"""Volume 1: The Drazin Inverse.
<Neal Munson>
<Section 2>
<April 2, 2019>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import csgraph
import csv


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    # check that A @ (A^D) = (A^D) @ A
    if not np.allclose(A@Ad,Ad@A):
        return False
    # check that (A^(k+1)) @ (A^D) = A^k
    if not np.allclose(np.linalg.matrix_power(A,k+1)@Ad,np.linalg.matrix_power(A,k)):
        return False
    # check that (A^D) @ A @ (A^D) = A^D
    if not np.allclose(Ad @ A @ Ad, Ad):
        return False
    # if all the previous are true, return True
    return True



# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    f = lambda x: abs(x) > tol # this is used to sort the row order
    g = lambda x: abs(x) <= tol # also used to sort the row order for the rest of the values
    n,n = A.shape
    Q1,S,k1 = la.schur(A, sort=f) # k represents the number of nonzero eigenvalues
    Q2,T,k2 = la.schur(A, sort=g)
    U = np.hstack((S[:,:k1],T[:,:n-k1]))
    Uinv = np.linalg.inv(U)
    V = Uinv @ A @ U
    Z = np.zeros((n,n),dtype='float64') # special care is taken that the zeros are floats

    if k1 != 0:
        Minv = np.linalg.inv(V[:k1,:k1])
        Z[:k1,:k1] = Minv

    return U @ Z @ Uinv


# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    # gets the shape of the matrix and establishes the array to use
    # to create the resistancy matrix
    n,n = A.shape
    M = np.zeros((n,n))

    # create the Laplacian matrix from the adjacency matrix
    D = np.diag(A.sum(axis = 0))
    L = D-A

    # follow the algorithm to turn the Laplacian values into resistancy values
    for j in range(n):
        Ltild = L.copy()
        Ltild[j,:] = np.eye(n)[j]
        LtD = drazin_inverse(Ltild)
        M[:,j] = np.diag(LtD)

    # makes sure that each effective resistance from a node to itself is 0
    for i in range(n):
        M[i,i] = 0

    return M


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.

        Attributes:
            Names: (list) The ordered list of people in the graph network
            AMatrix: (ndarray) The adjacency matrix for the network
            ER: (ndarray) The effective resistance Matrix of the adjacency matrix
        """
        # establish the names of each of the people in the graph
        # by reading in the names and creating an array
        self.Names = []
        with open(filename, newline='') as csvfile:
            file = csv.reader(csvfile,delimiter=',')
            for row in file:
                if row[0] not in self.Names:
                    self.Names.append(row[0])
                if row[1] not in self.Names:
                    self.Names.append(row[1])
        # get the number of people in the graph
        n = len(self.Names)
        # create the graph by re-reading in the people and creating the correct
        # connections in the adjacency matrix
        self.AMatrix = np.zeros((n,n))
        with open(filename, newline='') as csvfile:
            file = csv.reader(csvfile,delimiter=',')
            for row in file:
                per1, per2 = row[0], row[1]
                ind1, ind2 = self.Names.index(per1), self.Names.index(per2)
                self.AMatrix[ind1,ind2] = 1
                self.AMatrix[ind2,ind1] = 1
        self.ER = effective_resistance(self.AMatrix)

        return


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        # creates a mask called ERbool that holds True where things need to be ignored
        ERbool = self.ER == 0 # this ignores a node making a connection with itself
        n = len(self.Names)
        # cycles through the nodes and if they are connected, sets the mask to 'True'
        for r in range(n):
            for c in range(r,n):
                if self.AMatrix[r,c] == 1:
                    ERbool[r,c] = True
                    ERbool[c,r] = True

        if node == None:
            # return tuple with names of the nodes between which the next
            # link should occur
            minval = np.min(self.ER[ERbool != True])
            xloc,yloc = np.where(self.ER == minval)
            xind,yind = xloc[0],yloc[0]  # the xind and yind hold the indicies for the Names of
                                         # the two people most likely to connect next
            return (self.Names[xind],self.Names[yind])

        elif node in self.Names:
            # return the name of the node which should be connect to the node
            # next out of all the other nodes in the network
            nodeID = self.Names.index(node)
            minval = np.min(self.ER[nodeID][ERbool[nodeID] != True])
            xloc,yloc = np.where(self.ER == minval)
            xind,yind = nodeID,yloc[0]  # the xind and yind hold the indicies for the Names of
                                         # the two people most likely to connect next
            return(self.Names[yind])
        else:
            raise ValueError("node needs to be None or the name of someone in the Network")


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        if node1 not in self.Names or node2 not in self.Names:
            raise ValueError("One or both of the nodes is not in the graph.")
        # gets the indicies of the two names given
        ind1, ind2 = self.Names.index(node1), self.Names.index(node2)
        # updates the adjacency matrix
        self.AMatrix[ind1,ind2] = 1
        self.AMatrix[ind2,ind1] = 1
        # updates the effective resistance
        self.ER = effective_resistance(self.AMatrix)
        return
