"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number
"""
from typing import List

import numpy as np

from joelnet.train import train, test
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh
from joelnet.optim import SGD
from joelnet.loss import MSE, TSE

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]

def binary_decode(bitlist: List) -> int:
    pass

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=20,
      loss=MSE(),
      optimizer=SGD(lr=0.001))

inputs = np.array([
    binary_encode(x)
    for x in range(1, 101)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(1, 101)
])

labels = ["x", "fizz", "buzz", "fizzbuzz"]

test(net,
     inputs,
     targets,
     labels,
     binary_decode)
