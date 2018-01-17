"""modelDefinition.py"""

# Package imports
import numpy as np
from math import sqrt

def genesis(X):
    """
    Create the very first node
    :param X:
    :return:
    """
    np.random.seed(2)                               # So we can get consistent results when debugging

    L0 = X.shape[0]
    model = {'L0': L0, 'L1':1}
    #model['W1'] = np.arange(X.shape[0])
    #model['W1'] = model['W1'].reshape(1,L0)
    model['W1'] = np.random.randn(1, X.shape[0]) * sqrt(2.0/L0)
    model['W1'] = model['W1'].reshape((1, model['L0']))
    model['b1'] = -np.dot(model['W1'], X[...,0])      # Set b so that Z1[0,0] = 0, ie memorize this input
    model['b1'] = model['b1'].reshape((1,1))
    model['G1'] = []
    model['G1'].append(X[...,0])                         # Record the 'genesis memory' input vector
    model['G1'][0] = model['G1'][0].reshape((1,L0))

    return model


def add_unit(model, n, memory=False, Am=False, d_index=False):
    """
    Arguments:
        model   - Model dict
        n       - Layer new unit goes onto
        memory  - Add memory unit if True
        Am       - np.array(Ln,1) - Activation from previous level; A=X when building in L1
                  the pattern being 'memorized' is A[...,m_index], all rows, one col from m
        m_index - The column from A to use
    Returns:
        Updated model
    """
    if not model:                                   # Making very first unit?
        model = genesis(Am)
        return model

    assert n>=1                                     # Don't create units on input layer, L0
    np.random.seed(2)                               # So we can get consistent results when debugging

    Ln_name = 'L' + str(n)                          # This layer
    Lm_name = 'L' + str(n-1)                        # Previous layer
    Wn_name = 'W' + str(n)
    bn_name = 'b' + str(n)

    if Ln_name not in model:
        model[Ln_name] = 0                          # New layer, this will be the first unit
    Ln = model[Ln_name]                             # # of units in Ln
    Lm = model[Lm_name]

    Ln += 1
    W = np.random.randn(1, Lm) * sqrt(2.0/Lm)
    b = np.random.randn(1) * sqrt(2/Lm)             # Adding only one bias weight
    b = b.reshape((1,1))
    # Now add memory - set b so that Z=0 for this unit and this input sample
    if memory:
        d = Am.shape[1]
        assert Am.shape == (Lm, d)                 # Activity coming to this layer
        assert d > d_index
        A = Am[...,d_index]
        b = -np.dot(W, Am[..., d_index].T)          # Set b so that Z = 0, ie memorize the d'th item in A
        b = np.reshape(b, (1,1))
        Gn_name = 'G' + str(n)
        if Gn_name not in model:
            model[Gn_name] = []

        model[Gn_name].append(Am[...,d_index])  # Record the 'genesis memory' input vector
        model[Gn_name][0].reshape((1,Lm))

    if Wn_name not in model:
        model[Wn_name] = W
        model[bn_name] = b
    else:
        model[Wn_name] = np.vstack((model[Wn_name], W))       # Stack the new row onto Wn
        model[bn_name] = np.append(model[bn_name], b)
        model[bn_name] = np.reshape(model[bn_name], (1,Ln))

    model[Ln_name] = Ln

    return model


def add_memory_unit(model, n, A, d_index):
    """
    Arguments:
        model   dictionary of model parameters:
                - L     - np.array(L) of layer structures {'L0': #units in layer0 (input), 'L1': ...
                - G     - np.array(L?) of 'genisis memories', one for each unit built as a 'one-shot'; Only layer1 and above
                - Wn    - np.array(Ln, Ln-1) weight matrix; for W0, L(n-1) is input
                - bn    - np.array(Ln, 1) bias vector
        n       Layer unit will be added to
        A       - np.array(Ln,1) - Activation from previous level
                  the pattern being 'memorized' is A[...,m_index], all rows, one col from m
        d_index - The column from A to use
    Returns:
        Updated model
    """
    memory = True
    model = add_unit(model, n, memory, A, d_index)

    return model

