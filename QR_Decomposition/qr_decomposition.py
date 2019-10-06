# qr_decomposition.py
"""The QR Decomposition.
<Neal Munson>
<10/29/2018>
"""
import numpy as np
from scipy import linalg as la


def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n,n))
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]     # normalize the ith column of Q
        for j in range(i+1,n):
            R[i,j] = Q[:,j].T @ (Q[:,i]) # note that the multiplication here is vector
            Q[:,j] = Q[:,j]-R[i,j]*Q[:,i]    # orthogonalize the jth column of Q
    return Q,R


def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    # takes the product of the diagonal of R from the QR factorization
    return np.prod(np.diag(qr_gram_schmidt(A)[1]))


def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    n = len(A[0]) # size of n (recall A is n by n)
    Q,R = qr_gram_schmidt(A)
    y = Q.T @ b
    # use backsubstituion to solve Rx=y for x
    x = np.empty(n)
    x[n-1] = (1/R[n-1,n-1])*y[n-1]
    for k in range(n-2,-1,-1):
        summation = 0
        for j in range(k+1,n):
            summation += R[k,j]*x[j]
        x[k] = (1/R[k,k])*(y[k] - summation)
    return x


def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    sign = lambda x: 1 if x>= 0 else -1 # to be used in calculating 'u' values
    m,n = np.shape(A)
    R = np.copy(A)
    Q = np.eye(m)  # the m by m identity matrix
    for k in range(n):
        u = np.copy(R[k:,k])
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*np.outer(u,np.dot(u.T,R[k:,k:]))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u,np.dot(u.T,Q[k:,:]))
    return Q.T, R

def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x: 1 if x>= 0 else -1 # to be used in calculating 'u' values
    m,n = np.shape(A)
    H = np.copy(A)
    Q = np.eye(m)
    for k in range(n-2):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        np.outer(u.T,H[k+1:,k:])
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,np.dot(u.T,H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2*np.outer(np.dot(H[:,k+1:],u),u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,np.dot(u.T,Q[k+1:,:]))
    return H, Q.T
