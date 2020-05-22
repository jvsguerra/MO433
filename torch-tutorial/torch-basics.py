from __future__ import print_function
import torch

### 1) TENSOR BASICS

# Construct and unintialized matrix
x = torch.empty(5,3)
# print(x)

# Construct a randomly initialized matrix
x = torch.rand(5,3)
# print(x)

# Construct a zero matrix of dtype long
x = torch.zeros(5, 3,  dtype=torch.long)
# print(x)

# Construct a tensor directly from data
x = torch.tensor([5.5, 3])
# print(x)

# Create a tensor based on an existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
x = torch.randn_like(x, dtype=torch.float)  
# print(x)

# Get tensor size
# print(x.size()) # torch.Size() outputs a Tuple, so it supports Tuple operations


### 2) TENSOR OPERATIONS

x = torch.rand(5, 3)
y = torch.rand(5, 3)

# Addition syntax 1
# print(x + y)

# Addition syntax 2
# print(torch.add(x, y))

# Addition providing an output tensor as argument
results = torch.empty(5, 3)
torch.add(x, y, out=results)
# print(results)

# Addition in place
y.add_(x)
# print(y)
# Note: Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.

# Indexing
# print(x)
# print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# print(x)
# print(y)
# print(z)
# print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
# print(x)
# print(x.item())

### 3) NumPy Bridge

# Converting Torch Tensor to a NumPy array
a = torch.ones(5)
# print(a)

b = a.numpy()
# print(b)

a.add_(1)
# print(a)
# print(b)

# Converting NumPy array to a Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a) # See how changing the np array changed the Torch Tensor automatically
np.add(a, 1, out=a)
# print(a)
# print(b)