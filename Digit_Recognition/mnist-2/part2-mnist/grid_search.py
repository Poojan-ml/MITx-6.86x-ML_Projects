#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def train(batch_size=32, hidden_size=10, lr=0.1, momentum=0, activation=nn.ReLU):
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = batch_size
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, hidden_size),
              activation(),
              nn.Linear(hidden_size, 10),
            )
    lr=lr
    momentum=momentum
    ##################################

    val_acc = train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))
    return val_acc, accuracy


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility

    results_10 = OrderedDict()
    hidden_size=10
    results_10["baseline"] = train(hidden_size=hidden_size)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_10["batch_size=64"] = train(hidden_size=hidden_size, batch_size=64)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_10["lr=0.01"] = train(hidden_size=hidden_size, lr=0.01)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_10["momentum=0.9"] = train(hidden_size=hidden_size, momentum=0.9)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_10["activation=LeakyReLU"] = train(hidden_size=hidden_size, activation=nn.LeakyReLU) # default pytorch params

    #results_10["activation=ELU"] = train(hidden_size=hidden_size, activation=nn.ELU) # default pytorch params

    results_128 = OrderedDict()
    hidden_size=128
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_128["baseline"] = train(hidden_size=hidden_size)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_128["batch_size=64"] = train(hidden_size=hidden_size, batch_size=64)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_128["lr=0.01"] = train(hidden_size=hidden_size, lr=0.01)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_128["momentum=0.9"] = train(hidden_size=hidden_size, momentum=0.9)
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    results_128["activation=LeakyReLU"] = train(hidden_size=hidden_size, activation=nn.LeakyReLU) # default pytorch params

    print("results for 10 hidden (val, test):")
    print(results_10)

    print("results for 128 hidden (val, test):")
    print(results_128)
