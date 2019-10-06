# markov_chains.py
"""Markov Chains.
<Neal Munson>
<November 2, 2018>
"""

import numpy as np
from scipy import linalg as la


def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    A = np.random.random((n,n))
    return A/A.sum(axis=0)


def forecast(days):
    """Forecast 'day' number of days weather in the future given that today is hot.
    Returns a list with 'day' number of entries, indicating whether the day was
    cold (0) or hot (1).  Doesn't include today.
    """
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    weather = 0
    prediction = []
    # Sample from a binomial distribution to choose a new state.
    for i in range(days):
        weather = np.random.binomial(1,transition[1,weather])
        prediction.append(weather)
        weather = prediction[-1]
    return prediction # keep in mind the first day is on the left of the list

def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    transition = np.array([[0.5,0.3,0.1,0.0],
                           [0.3,0.3,0.3,0.3],
                           [0.2,0.3,0.4,0.5],
                           [0.0,0.1,0.2,0.2]])
    weather = 1
    translation = np.diag(range(4))
    prediction = []
    # Sample from a multinomial distribution to choose a new state.
    for i in range(days):
        weather = np.random.multinomial(1,transition[:,weather])
        weather = sum(translation @ weather)
        prediction.append(weather)
        weather = prediction[-1]
    return prediction # keep in mind the first day is on the left of the list



def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    n = len(A)
    done = False
    x0 = np.random.random(n)
    k = 0
    while not done:
        x1 = A @ x0
        if la.norm(x1-x0) < tol: # returns if the vector is close enough to converging
            return x1
        if k > N:
            raise ValueError("A**k does not converge in {} steps.".format(N))
        x0 = x1
        k += 1


class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        contents: (list) contains the list of lines from the file
        tmat: (np.array) the transition matrix
        states: (np.array) the 'translation' of vector index to word
        stop_index: (int) Index of the "$top" element

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # open up the file and read in the lines
        with open(filename, 'r') as openedfile:
            self.contents = openedfile.readlines()
        states = ["$tart"] # states will contain the 'vector' representation of the
                           # values that each word can take on
        for line in self.contents:
            line = line.strip().split(' ')
            for word in line:
                if word not in states:
                    states.append(word)
        states.append("$top")
        # tmat represents the transition matrix from one word to another
        self.tmat = np.zeros((len(states),len(states)))
        for line in self.contents:
            line = line.strip().split(' ')
            for i,word in enumerate(line):
                if i == 0:
                    self.tmat[states.index(word),states.index("$tart")] += 1
                else:
                    self.tmat[states.index(word),states.index(w)] += 1
                w = word
            self.tmat[states.index("$top"),states.index(w)] += 1
        self.tmat[states.index("$top"),states.index("$top")] += 1
        self.states = states
        self.tmat = self.tmat/self.tmat.sum(axis=0)
        self.stop_index = len(states) - 1
        return

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        transition = self.tmat
        word_index = 0
        num_sentence = [] # num sentence calculates the sentence by numbers, using
        # the transition matrix to see the number's order.
        # Sample from a multinomial distribution to choose a new state.
        while word_index != self.stop_index:
            word_vector = np.random.multinomial(1,transition[:,word_index])
            word_index = np.argmax(word_vector)
            num_sentence.append(word_index)
        word_sentence = [] # word sentence represents the sentence in strings
        for i in num_sentence:
            word_sentence.append(self.states[i])
        word_sentence.remove("$top")
        sentence = ' '.join(word_sentence)
        return sentence
