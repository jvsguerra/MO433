# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Define the Neural Network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b -> Linear transformation of the incoming data
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6 x 6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Conv 1
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # Conv 2
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x)) # Linear transform 1
        x = F.relu(self.fc2(x)) # Linear transform 2
        x = self.fc3(x) # Linear transform 3
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# The learned parameters
params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

# Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32. To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.
input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(out)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# 2) Loss Function
output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()  # computes the mean-squared error between the input and the target.

loss = criterion(output, target)
# print(loss)

# Now, if you follow loss in the backward direction, using its .grad_fn attribute, you will see a graph of computations that looks like this:
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
    #   -> view -> linear -> relu -> linear -> relu -> linear
    #   -> MSELoss
    #   -> loss

# So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, and all Tensors in the graph that has requires_grad=True will have their .grad Tensor accumulated with the gradient.
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# 3) Back Propagation

# To backpropagate the error all we have to do is to loss.backward(). You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.

# Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.
net.zero_grad()     # zeroes the gradient buffers of all parameters

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# 4) Update the weights
# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
#|    weight = weight - learning_rate * gradient

# We can implement this using simple Python code:
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# However, as you use neural networks, you want to use various different update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this, we built a small package: torch.optim that implements all these methods. Using it is very simple:
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update