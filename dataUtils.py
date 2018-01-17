""" dataUtils.py"""

import numpy as np
from numpy import linalg as LA


def write_data(fname, array):
    np.set_printoptions(precision=8, suppress=True)
    d = array.astype(np.int32)
    #print(d)
    np.savetxt(fname, d)


def read_data(fname):
    d = np.loadtxt(fname)
    return d


def load_simple_data(m, x_dim):
    d = np.zeros((m, x_dim), dtype=np.int8)
    step = int(x_dim/m)
    d = d.reshape((m, int(x_dim/step), step))

    for i in range(m):
        d[[i],[i]] = 1

    d = d.reshape((m, x_dim))
    # Stacking one more so I won't confuse m with l2. Without this they'd be equal
    d = np.vstack((d, d[-1]))
    d = d.T
    assert d.shape == (x_dim, m+1)

    write_data("d1.txt", d)
    d = read_data("d2.txt")
    print(d.T)
    return d.T

def load_labels():
    l = [0,1,2,3,4,5,6,7]
    return l

def make_1D_gaussian_data(mean, std_dev, length):
    a = np.random.randn(length) * std_dev + mean
    return (np.stack([a]))



def make_nD_gaussian_data(params, length):
    """ Generate 'length' samples of n-dim gaussian data
            with specified mean, std_dev in each dim
        params = [mean0, std_dev0, ... mean_n, std_dev_n, ...] for n features """
    for p in params:
        a = np.random.randn(length) * p[1] + p[0]
        if 'X' in locals():
            X = np.block([[X], [a]])
        else:
            X = np.stack([a])     # Convert to rank1, never return rank0
    return np.transpose(X)

def make_n_nD_gaussian_data(num, mean, std, length):
    """ Generate 'length' samples of n-dim gaussian data
            with specified mean, std_dev in each dim
        params = [mean0, std_dev0, ... mean_n, std_dev_n, ...] for n features """
    for n in range(num):
        a = np.random.randn(length) * std + mean * neg_pos_flag
        if 'X' in locals():
            X = np.block([[X], [a]])
        else:
            X = np.stack([a])     # Convert to rank1, never return rank0
    return np.transpose(X)


""" make num gaussians all with length rows and num columns (data) and num columns 'one-hot' labels
       same mean, std_dev except that:
   g0 mean = mean for all cols
   g1 mean[1] = -mean, all other mean[n] = mean
   g2 mean[2] = -mean, all other mean[n] = mean
 ...
   returns X[length][2*num] which includes data in first num cols, 'one-hot' labels in 2nd num cols; not shuffled
        For length=3, num=2
        g0 g0 1 0       sample 0 from g0
        g0 g0 1 0       sample 1 from g0
        g0 g0 1 0       sample 2 from g0
        g1 g1 0 1       sample 0 from g1
        g1 g1 0 1       sample 1 from g1
        g1 g1 0 1       sample 2 from g1
"""
def make_N_gaussians_and_labels(num, mean, std, length):

    y_ones = np.ones((1, length))
    y_zeros = np.zeros((1, length))

    for n in range(num):
        x = np.random.randn(num, length) * std + mean
        # negate the mean on the right dimension (but not for [0])
        #if n > 0:
        x[n] *= -1
        # make 'one-hot' labels
        for nn in range(num):
            if nn == 0:
                if n == 0:
                    y = y_ones
                else:
                    y = y_zeros
            else:
                if n == nn:
                    y = np.concatenate((y, y_ones), axis=0)
                else:
                    y = np.concatenate((y, y_zeros), axis=0)
        # concat g0, g1, ...
        if n == 0:
            xx = x
            yy = y
        else:
            xx = np.concatenate((xx, x), axis=1)
            yy = np.concatenate((yy, y), axis=1)

    fname = '/Users/pat/Workspace/ml/scaleInvariance/labels_' + str(num) + '_classes_by_' + str(length) + '_samples.txt'
    with open(fname, 'w') as labels_file:
        #labels_file.write("%s\n" % 'label')
        for label in range(num):
            for item in range(length):
                labels_file.write("%d\n" % label)

    return xx.T, yy.T


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_data(num, mean, std, length):
    x, y = make_N_gaussians_and_labels(num, mean, std, length)
    x, y = unison_shuffled_copies(x, y)
    return x,y

# make
def make_3_3D_gaussians():
    length = 100
    y_ones = np.ones((length,1))
    y_zeros = np.zeros((length,1))
    params = [[4, 1], [4, 1], [4, 1]]
    x1 = make_nD_gaussian_data(params, length)
    xy1 = np.concatenate((x1, y_ones), axis=1)
    xyy1 = np.concatenate((xy1, y_zeros), axis=1)
    xyyy1 = np.concatenate((xyy1, y_zeros), axis=1)

    params = [[-4, 1], [-4, 1], [-4, 1]]
    x2 = make_nD_gaussian_data(params, length)
    xy2 = np.concatenate((x2, y_zeros), axis=1)
    xyy2 = np.concatenate((xy2, y_ones), axis=1)
    xyyy2 = np.concatenate((xyy2, y_zeros), axis=1)

    params = [[0, 2], [0, 2], [0, 2]]
    x3 = make_nD_gaussian_data(params, length)
    xy3 = np.concatenate((x3, y_zeros), axis=1)
    xyy3 = np.concatenate((xy3, y_zeros), axis=1)
    xyyy3 = np.concatenate((xyy3, y_ones), axis=1)

    xyyy12  = np.concatenate((xyyy1, xyyy2), axis=0)
    xyyy123 = np.concatenate((xyyy12, xyyy3), axis=0)
    np.random.shuffle(xyyy123)
    xy = np.split(xyyy123, np.array([3,]), axis=1)
    x = xy[0]
    y = xy[1]
    return x, y


def total_err(error):
    total = LA.norm(error)
    return total

def winner_error(error):
    i, j = error.shape
    mask = np.eye(i, j, dtype=float)
    win_err = LA.norm(mask * error)
    return win_err

