"""fwdProp.py"""

# Package imports
import numpy as np
import modelDefinition

def forward_prop(model, cache, layer):
    # Make name strings
    Lm_name = 'L' + str(layer)                          # This layer
    Ln_name = 'L' + str(layer+1)                        # Previous layer
    Wn_name = 'W' + str(layer+1)
    bn_name = 'b' + str(layer+1)
    Am_name = 'A' + str(layer)

    Am = cache[Am_name]
    if not model:                                       # Start building from zero!
        model = modelDefinition.genesis(Am)

    if Ln_name not in model:
        model[Ln_name] = 0                              # New layer, make the first unit and continue
        model = modelDefinition.genesis(Am)

    Ln = model[Ln_name]                                 # # of units in Ln
    Lm = model[Lm_name]

    W = model[Wn_name]
    assert W.shape == (Ln, Lm)
    b = model[bn_name]

    # Forward Prop, calculate An (probabilities)
    Zn = np.dot(W, Am) + b.T
    An = np.tanh(Zn)

    # Save results in cache
    Zn_name = 'Z' + str(layer+1)
    An_name = 'A' + str(layer+1)
    cache[Zn_name] = Zn
    cache[An_name] = An
    m = Am.shape[1]
    assert Zn.shape == (Ln, m)
    assert An.shape == (Ln, m)

    return model, cache


