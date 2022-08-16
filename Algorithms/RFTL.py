import random as rd
from Algorithms.Projector import *
import numpy as np
from utils import mask

def optimistic_smd(model, weak_learners, eta, X, y, epoch, l, z=1, lr=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses the gradloss function to update parameters
    :param model: the first svm on which we will aggregate next weak learners with "boosting" process
    :param n_earners: number of weak learners to be considered sequentially
    :param eta: proportion we will consider for aggregation
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs

    --> One improvement would be to try adding weak learners if needed, here the cardinality is fixed !
    """

    #initialization
    n_learner = len(weak_learners)
    losses = []
    n, _ = X.shape # number of data
    wts = [np.zeros(len(model.w))] # initial weights of the predictor

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility
        t = i + 1
        dlr = lr / np.sqrt(t)

        for k in range(n_learner):

            # update the last weight xt
            # first descent
            new_weak_wts = weak_learners[k].w - dlr * weak_learners[k].gradLoss(mask(sample_x, k, n_learner),
                                                                                sample_y - (1-eta)*model.predict(sample_x), l) # OMD step on each w.l.
            new_weak_wts = proj_l1(new_weak_wts, z)  # projection on the l1-ball (stability)

            weak_learners[k].w = (t * weak_learners[k].w + new_weak_wts)/(t+1)


            # second descent
            #new_weak_wts = weak_learners[k].w - dlr * weak_learners[k].gradLoss(mask())


            # agregation step (boosting)

            model.w = (1-eta) * model.w + weak_learners[k].w # eta is already included in the learning step of the k-th weak learner
            wts.append(model.w)

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)


def smd(model, X, y, epoch, l, z=1, lr=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    losses = []
    wts = [np.zeros(len(model.w))]
    n, _ = X.shape # number of data

    for i in range(epoch):

        # sample of one "fly-data"
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        dlr = lr / np.sqrt(t)
        new_wts = wts[-1] - dlr * model.gradLoss(sample_x, sample_y, l)
        new_wts = proj_l1(new_wts, z)
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    # update wts:
    model.w = np.mean(wts, axis=0)
    return losses, np.array(wts)


def seg(model, X, y, epoch, l, z=1, lr=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses the gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    tetatp = np.zeros(d)
    tetatm = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        etat = lr * np.sqrt(1 / t)
        tetatm -= etat * model.gradLoss(sample_x, sample_y, l)
        tetatp += etat * model.gradLoss(sample_x, sample_y, l)
        tetat = np.r_[tetatm, tetatp]
        new_wts = np.exp(tetat)/np.sum(np.exp(tetat))
        new_wts = z * (new_wts[0:d] - new_wts[d:])
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    # update wts:
    model.w = np.mean(wts, axis=0)
    return losses, np.array(wts)


def adagrad(model, X, y, epoch, l, z=1, lr=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    Sts = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        Sts += model.gradLoss(sample_x, sample_y, l)**2
        Dt = np.diag(np.sqrt(Sts))
        yts = wts[-1] - lr * np.linalg.inv(Dt).dot(model.gradLoss(sample_x, sample_y, l))
        new_wts = weighted_proj_l1(yts, np.diag(Dt), z)
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    # update wts:
    model.w = np.mean(wts, axis=0)
    return losses, np.array(wts)
