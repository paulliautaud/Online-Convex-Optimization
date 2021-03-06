from Algorithms.Projector import *
import numpy as np


def gd(model, X, y, epoch, l, verbose=0, lr=1):
    """
    Unconstrained GD
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param epoch: (int) maximum number of iteration of the algorithmnp.array(wts)
    :param lr: (float) learning rate
    :param l:  (float) regularization parameter (lambda)
    :param verbose: (int) print epoch results every n epochs
    """
    losses = []
    wts = [model.w]
    for i in range(epoch):
        t = i + 1
        dlr = lr / (l * t)
        new_wts = wts[-1] - dlr * model.gradLoss(X, y, l)
        wts.append(new_wts)
        model.w = new_wts
        current_loss = model.loss(X, y, l)
        losses += [current_loss]
        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))
    return losses, np.array(wts)


def projected_gd(model, X, y, epoch, l, z=1, verbose=0, lr=1):
    """
    Constrained GD with projection on B1(z)
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius for projection on the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    assert z > 0, 'L1-Ball radius should be positive'
    losses = []
    wts = [model.w]
    for i in range(epoch):
        t = i + 1
        lr = lr / (l * t)
        new_wts = wts[-1] - lr * model.gradLoss(X, y, l)
        new_wts = proj_l1(new_wts, z)
        wts.append(new_wts)
        model.w = new_wts
        current_loss = model.loss(X, y, l)
        losses += [current_loss]
        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)
