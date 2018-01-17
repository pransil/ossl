"""modelMods.py - Adding new units to top layer

"""
import numpy as np
from scipy import spatial

def add_unit(model, cache, Xi, m_index):
    # model -- weights and biases
    # cache -- need for A1i and A2
    # Xi    -- The input this unit is created to 'memorize'

    # First check to see if a unit already exists for this input
    """if 'W2' in model:                   # Always create if l2 == 0
        A1 = cache['A1']
        error, _ = compute_error(cache['A2'], margin)
        Z2i = np.dot(model["W2"], A1i) + model["b2"]
        assert Z2.shape == (l2, m)
        # A2 = functions.sigmoid(Z2)
        A2 = np.tanh(Z2)  # tanh is zero-centered
    else: """

    if 'W2' in model:
        W2 = model["W2"]
        (l2, l1) = W2.shape                 # Current shape. Will be (l2+1, l1) at exit
        b2 = model["b2"]
    else:
        W1 = model['W1']
        l1 = W1.shape[0]
        l2 = 0                              # Will +=1 a few lines down
    model['l' + str(l2) + '_initial_memory'] = Xi         # May want for diag, tracking of behavior

    l2 += 1                                 # Creating new unit
    # create weights and bias for new unit so that they add to zero
    W = np.random.randn(1, l1) / ((l2+1)*(l2+1))    # One new row of weights
    A1 = cache['A1']
    A1i = A1[..., m_index]  # l1 by 1
    b = -np.dot(W, A1i)                     # Set b so that Z2(A1i) = 0, ie memorize this inpu

    if 'W2' in model:
        W2 = np.vstack((W2, W))
        b2 = np.append(b2, b)
        b2 = b2.reshape((l2, 1))
    else:
        W2 = W
        b2 = b
        b2 = b2.reshape((l2, 1))

    Z2 = np.dot(W2, A1) + b2
    #assert Z2.shape == (l2, m)
    A2 = np.tanh(Z2)  # tanh is zero-centered

    model["W2"] = W2
    model["b2"] = b2
    cache['A2'] = A2
    cache['Z2'] = Z2

    assert W2.shape == (l2, l1)
    assert b2.shape == (l2, 1)

    return model, cache

def find_dups(Am, An, margin):
    dups = []
    rows, cols = An.shape
    m_dist = spatial.distance.pdist(Am.T)
    n_dist = spatial.distance.pdist(An.T)
    for r in range(rows-1):
        for c in range(r, rows):
            if dist[0] < margin:
                dups.append((r,c+1))
            dist = dist[1:]

    return dups


def remove_one_dup(model, cache, index, layer):
    Wn_name = 'W' + str(layer)
    bn_name = 'b' + str(layer)
    model[Wn_name] = np.delete(model[Wn_name], index, axis=0)
    model[bn_name] = np.delete(model[bn_name], index, axis=0)

    mem = 'G' + str(layer)
    del model[mem[index]]

    Zn_name = 'Z' + str(layer)
    An_name = 'a' + str(layer)
    cache[Zn_name] = np.delete(cache[Zn_name], index, axis=1)
    cache[An_name] = np.delete(cache[An_name], index, axis=1)

    return model, cache


def remove_duplicate_units(model, cache, layer, margin):
    """
    layer - Layer number
    """

    An_name = 'A' + str(layer)
    An = cache[An_name]
    Am_name = 'A' + str(layer-1)
    Am = cache[Am_name]
    dups = find_dups(Am, An, margin)
    if len(dups) < 0:
        model, cache = remove_one_dup(model, cache, index, layer)
        model, cache = remove_duplicate_units(model, cache, layer, margin)

    return model, cache
