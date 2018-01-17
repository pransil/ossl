"""backProp.py"""

import numpy as np

def make_target(A, margin):
    Ln, m = A.shape
    target = np.sign(A)                            # loser target is -1, +1; winner target=0
    target_mask = np.abs(A) > margin               # Mostly to handle numerical precision for zero
    assert target_mask.shape == (Ln, m)
    target = target * target_mask
    assert target.shape == (Ln, m)

    return target


def find_winners_simple(target):
    winners = target == 0                           # Really? Is it that simple?
    return winners


def compute_error(A, margin):

    """Arguments:
    A       -- The tanh output of the activation, of shape (categories, number of examples)
    margin  -- How close the output needs to be to 0 to be considered  the 'winner'
    Returns:
    error   -- 0 - A for winners, +/-1 - A for losers
    target  -- [-1, 0, 1, ...]
    winners -- [True, False, ...]
    """

    (Ln, m) = A.shape
    # error pushes losers away from zero and winner toward zero
    target = make_target(A, margin)                 # target[] is (Ln by m)
    error = target - A                              # find distance from -1 or +1 (for losers)
    winners = find_winners_simple(target)
    return error, target, winners


def back_prop(model, cache, layer, error, margin):
    """
    Arguments:
    model   -- dictionary containing weights, biases...
    cache   -- dictionary containing "Z1", "A1", "Z2", "A2"
    layer   -- from this layer (Ln) to prev (Lm)

    Returns:
    grads   -- dictionary containing gradients wrt weights, biases;
               grads are zeroed out for non-winners and multiple units can 'win' an input
    """
    d = float(cache['A0'].shape[1])
    Am_name = 'A' + str(layer-1)
    Wn_name = 'W' + str(layer)
    bn_name = 'b' + str(layer)
    An_name = 'A' + str(layer)
    Zn_name = 'Z' + str(layer)
    Am = cache[Am_name]                     # This is X (A0) if layer=1
    Wn = model[Wn_name]
    An = cache[An_name]
    Zn = cache[Zn_name]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZn = An - error
    target = make_target(An, margin)
    winners = find_winners_simple(target)
    dZn_boosted = boost_dZ(dZn, target)

    #dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))        SAVE THIS --- WILL NEED LATER
    dW = (1 / d) * np.dot(dZn, Am.T)
    db = (1 / d) * np.sum(dZn, axis=1, keepdims=True)
    db = db.T

    assert dW.shape == model[Wn_name].shape
    assert db.shape == model[bn_name].shape
    assert dZn_boosted.shape == cache[Zn_name].shape

    grads = {"dW1": dW_boosted,
             "db1": db }

    return grads


# For each winner, boost the contribution to dW. Hack needed because learning is moving
# units too far from their initial memory
def boost_dZ(dZn, target):
    winners = find_winners_simple(target)

    lr_boost = 8000                                # ToDo - should pass in. Hack!!!
    dZ_winners = dZn * winners                     # zero except in winner locations that have been 'knocked off' center
    dZ_winners_boosted = dZ_winners * lr_boost
    dZ_boosted = dZn + dZ_winners_boosted

    #return dZ_boosted
    return dZn


def update_parameters(model, grads, d, learning_rate=0.0002):
    """
    Update weights and biases to be closer to center of cluster
    Arguments:
    model -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    d     --    Number of data samples

    Returns:
    model -- wth updated W and b
    """

    W = model['W1']             # Todo - generalize
    b = model['b1']
    dW = grads['dW1']
    db = grads['db1']

    # Update rule for each parameter
    W = W - learning_rate * dW
    b = b - (learning_rate * db)        #/(d*100)

    model['W1'] = W
    model['b1'] = b

    return model
