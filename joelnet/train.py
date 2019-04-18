"""
Here's a function that can train a neural net
"""
from typing import List, Callable

import numpy as np

from joelnet.tensor import Tensor
from joelnet.nn import NeuralNet
from joelnet.loss import Loss, TSE
from joelnet.optim import Optimizer, SGD
from joelnet.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = TSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)

def test(net: NeuralNet,
         inputs: Tensor,
         targets: Tensor,
         labels: List,
         input_decoder: Callable) -> None:
    correct = 0
    for i in range(1, len(inputs)):
        predicted = net.forward(inputs[i])
        predicted_idx = np.argmax(predicted)
        actual_idx = np.argmax(targets[i])
        print(input_decoder(inputs[i]), inputs[i], labels[predicted_idx], labels[actual_idx])
        if predicted_idx == actual_idx:
            correct += 1
    print(correct / len(inputs))