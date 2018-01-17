"""main.py for OSSL - Oneshot SelfSupervised Learning
    A variant on k-means clustering except that cluster-overlap should be common, providing a
    'coarse coding' of inputs.
    For each input, if no unit 'wins' (ie input is in its cluster), create a new unit (cluster)
    Margin is a hyper param defining the radius of each cluster. Currently it is the same for all dimensions.

    Uses Error/Target where output of 0 is a 'win' and losers are -1 or +1. Not sure this is a good idea.
        Might go back to normal win=1 and lose=0
"""

import numpy as np
import dataUtils
import modelDefinition
import fwdProp
import backProp


if __name__ == '__main__':
    d = 10               # Number of data samples
    L0 = 10             # Width of input vector - X[L0, m]
    margin = 0.002
    learning_rate = 0.0000001
    epochs = 100

    # Load the data and define the network
    X = dataUtils.load_simple_data(d, L0)
    d = X.shape[1]
    cache = {'A0': X}                           # Input is also A0
    model = False
    layer = 0
    for e in range(epochs):
        model, cache = fwdProp.forward_prop(model, cache, layer)
        # error = target - A; target(winner) = 0 = A; target(loser-)= -1 = -1 -A; target(loser+)= 1 = 1 -A;
        # for loser-, want A more neg; for loser+, want A more+; for winner, want |A| smaller
        error, target, winners = backProp.compute_error(cache['A1'], margin)

        # For every sample with no winner (target !=0) create new unit (cluster)
        Ln = cache['A1'].shape[0]
        num_winners = np.sum(winners, axis=0)       # Sum each column
        for index in range(d):
            if num_winners[index] < 1:
                model = modelDefinition.add_memory_unit(model, layer+1, X, index)

        # Now that we have all clusters, re-run, re-calculate error,
        model, cache = fwdProp.forward_prop(model, cache, layer)
        error, target, winners = backProp.compute_error(cache['A1'], margin)

        grads = backProp.back_prop(model, cache, layer+1, error, margin)
        model = backProp.update_parameters(model, grads, d, learning_rate)

        # Print the cost every 10 iterations
        if e % 10 ==0 :
            print (dataUtils.winner_error(error), dataUtils.total_err(error))
            #print('A1', An)
            #print ('Target in main\n',target)

