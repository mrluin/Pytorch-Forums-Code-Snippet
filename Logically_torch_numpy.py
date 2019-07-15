import torch
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

x_copy = x.copy()
y_copy = y.copy()
w1_copy = w1.copy()
w2_copy = w2.copy()

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x_pth = torch.from_numpy(x_copy).clone()
y_pth = torch.from_numpy(y_copy).clone()

# Randomly initialize weights
w1_pth = torch.from_numpy(w1_copy).clone()
w2_pth = torch.from_numpy(w2_copy).clone()

learning_rate = 1e-6
for t in range(500):
    # numpy
    print('###########ITER{}#############'.format(t))

    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    # print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    # PyTorch
    # Forward pass: compute predicted y
    h_pth = x_pth.mm(w1_pth)
    h_relu_pth = h_pth.clamp(min=0)
    y_pred_pth = h_relu_pth.mm(w2_pth)

    # Compute and print loss
    loss_pth = (y_pred_pth - y_pth).pow(2).sum().item()
    # print(t, loss_pth)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred_pth = 2.0 * (y_pred_pth - y_pth)
    grad_w2_pth = h_relu_pth.t().mm(grad_y_pred_pth)
    grad_h_relu_pth = grad_y_pred_pth.mm(w2_pth.t())
    grad_h_pth = grad_h_relu_pth.clone()
    grad_h_pth[h_pth < 0] = 0
    grad_w1_pth = x_pth.t().mm(grad_h_pth)

    # Update weights using gradient descent
    w1_pth -= learning_rate * grad_w1_pth
    w2_pth -= learning_rate * grad_w2_pth

    print('h diff {}'.format(np.abs((h - h_pth.numpy())).sum()))
    print('h_relu diff {}'.format(np.abs((h_relu - h_relu_pth.numpy())).sum()))
    print('y_pred diff {}'.format(np.abs((y_pred - y_pred_pth.numpy())).sum()))
    print('loss diff {}'.format(np.abs((loss - loss_pth)).sum()))
    print('grad_w1 diff {}'.format(np.abs((grad_w1 - grad_w1_pth.numpy())).sum()))
    print('grad_w2 diff {}'.format(np.abs((grad_w2 - grad_w2_pth.numpy())).sum()))
    print('w1 diff {}'.format(np.abs((w1 - w1_pth.numpy())).sum()))
    print('w2 diff {}'.format(np.abs((w2 - w2_pth.numpy())).sum()))