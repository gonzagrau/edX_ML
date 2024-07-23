#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
from typing import Callable
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model


def grid_search_iter(X_train: np.ndarray, y_train: np.ndarray,
                     X_dev: np.ndarray, y_dev: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     batch_size: int,
                     hidden_size: int,
                     lr: float,
                     momentum: float,
                     activation: torch.nn) -> float:
    """
    Trains a model with certain specs, returns its testing accuracy
    """
    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification
    model = nn.Sequential(
        nn.Linear(784, hidden_size),
        activation,
        nn.Linear(hidden_size, 10),
    )

    ##################################

    val_acc = train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)
    print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))

    return val_acc

def main():
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
    X_train = np.array([X_train[i] for i in permutation])
    y_train = np.array([y_train[i] for i in permutation])

    baseline_args = {'batch_size':32, 'hidden_size':128, 'lr':0.1, 'momentum':0, 'activation':nn.ReLU()}
    batch_64_args = {'batch_size':64, 'hidden_size':128, 'lr':0.1, 'momentum':0, 'activation':nn.ReLU()}
    lr_small_args = {'batch_size':32, 'hidden_size':128, 'lr':0.01, 'momentum':0, 'activation':nn.ReLU()}
    momentum_args = {'batch_size':32, 'hidden_size':128, 'lr':0.1, 'momentum':0.9, 'activation':nn.ReLU()}
    leakReLu_args = {'batch_size':32, 'hidden_size':128, 'lr':0.1, 'momentum':0, 'activation':nn.LeakyReLU(negative_slope=0.01)}

    names = ['Baseline', 'Batch of 64', 'Learning rate 0.01', 'Momentum 0.9', 'Leaky RelU']
    args_list = [baseline_args, batch_64_args, lr_small_args, momentum_args, leakReLu_args]
    perf_dict = {}
    best_perf = 0

    # print('Training on batch of 64...')
    # grid_search_iter(X_train, y_train, X_dev, y_dev, X_test, y_test, **batch_64_args)

    for name, kwargs in zip(names, args_list):
        print(f"Training with {name}")
        perf = grid_search_iter(X_train, y_train, X_dev, y_dev, X_test, y_test, **kwargs)
        perf_dict[perf] = name
        if perf > best_perf:
            best_perf = perf
        print('='*64, '\n\n')

    print(f"Best perf: {best_perf}, with params. '{perf_dict[best_perf]}'")


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
