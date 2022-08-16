"""
Sorbonne University
Master M2A
Convex sequential Optimization

Liautaud Paul

Main file for Optimistic Rakhlin algos
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns

from Algorithms.Adam import adamax, adamax_temporal, adam, adam_p, adam_temporal, adam_proj
from Algorithms.Explo import sbeg, sreg
from Algorithms.GD import gd, projected_gd
from Algorithms.SGD import sgd, projected_sgd
from Algorithms.RFTL import adagrad, seg, smd, optimistic_smd
from Algorithms.ONS import ons
from Models.LinearSVM import LinearSVM
from utils import *

# --- PARAMETERS ---
np.random.seed(123)

lr = 0.1
nepoch = 1000
lbd = 1 / 3  # or change to 1/5 for sbeg and sreg to get better results
Z = [100]
gamma = 1 / 8
verbose = 10

N_weak = [10, 20] #number of weak learners
eta = 0.2

alg_to_run = [
    'sgd',
    'c_sgd',
    'smd',
    'optimistic_smd',
    'seg',
    'adagrad',
    'ons',
    'sreg',
    'sbeg',
    'adam',
    'adam_fixlr',
    'adam_proj',
    'adamp',
    'adamax',
    'adamtemp',
    'adamaxtemp']

alg_to_run = ['optimistic_smd']


############################### Read and prepare data ####################

mnist_train = pd.read_csv('mnist_train.csv', sep=',', header=None)   # Reading
# Extract data
train_data = mnist_train.values[:, 1:]
# Normalize data
train_data = train_data / np.max(train_data)
train_data = np.c_[train_data, np.ones(train_data.shape[0])]         # Add intercept
# Extract labels
train_labels = mnist_train.values[:, 0]
# if labels is not 0 ==> -1 (Convention chosen)
train_labels[np.where(train_labels != 0)] = -1
# if label is 0 ==> 1
train_labels[np.where(train_labels == 0)] = 1

mnist_test = pd.read_csv('mnist_test.csv', sep=',', header=None)
test_data = mnist_test.values[:, 1:]
test_data = test_data / np.max(test_data)
test_data = np.c_[test_data, np.ones(test_data.shape[0])]
test_labels = mnist_test.values[:, 0]
test_labels[np.where(test_labels != 0)] = -1
test_labels[np.where(test_labels == 0)] = 1

test = test_data[0,:].reshape(1,-1)
test_sample = mask(test,0,10,0.85)

n, m = train_data.shape
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

############################### Test algorithms ###############################

time_dict = {}

if 'smd' in alg_to_run:
    for z in Z:
        print("-----------SMD  - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        SMDprojloss, wts = smd(
            model, train_data, train_labels, nepoch, lbd, z, lr, verbose)
        time_dict['smd z=' + str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained SMD (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, SMDprojloss[-1], acc))
        ax[0].plot(np.arange(nepoch), SMDprojloss, label='smd z=' + str(z))
        SMDprojaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(SMDprojaccuracies, label='smd z=' + str(z))
        SMDprojerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(SMDprojerrors, label='smd z=' + str(z))

if 'optimistic_smd' in alg_to_run:
    for z in Z:
        for N in N_weak :
            print("-----------Optimistic-SMD - z=" + str(z) + ' N ='+ str(N) + "----------- \n")
            model = LinearSVM(m)
            weak_learners = [LinearSVM(m)]*N
            tic = time.time()
            Opt_SMDprojloss, wts = optimistic_smd(model, weak_learners, eta, train_data, train_labels, nepoch, lbd, z, lr, verbose)
            time_dict['Optimistic smd z=' + str(z) + ' N ='+ str(N) + ' weak learners'] = (time.time() - tic)
            pred_test_labels = model.predict(test_data)
            acc = accuracy(test_labels, pred_test_labels)
            print(
                'After {:3d} epoch, constrained Opt SMD (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
                    nepoch, z, Opt_SMDprojloss[-1], acc))
            ax[0].plot(np.arange(nepoch), Opt_SMDprojloss, label='z=' + str(z) + ' N ='+ str(N))
            Opt_SMDprojaccuracies = compute_accuracies(wts, test_data, test_labels)
            ax[1].plot(Opt_SMDprojaccuracies, label='z=' + str(z) + ' N ='+ str(N))
            Opt_SMDprojerrors = compute_errors(wts, test_data, test_labels)
            ax[2].plot(Opt_SMDprojerrors, label='z=' + str(z) + ' N ='+ str(N))



# Log scale
ax[0].set_xscale('log')
ax[0].set_yscale('logit')
ax[1].set_xscale('log')
ax[1].set_yscale('logit')
ax[2].set_xscale('log')
ax[2].set_yscale('logit')

# Legend
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[2].set_title('Error')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
ax[2].set_xlabel('Epochs')
plt.savefig('LossAccuraciesErrors.png')
plt.show()

plt.clf()
keys = list(time_dict.keys())
sns.barplot(x=keys, y=[time_dict[k] for k in keys])
plt.savefig('execution_time.png')
plt.show()
