import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
# Create random input and output data
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


# Randomly initialize weights
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

# hold loss over iterations for plotting
xs = []
ys = []

learning_rate = 1e-6
for t in range(500):
	# Forward pass: compute predicted y
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)
	
	# compute and print loss
	loss = (y_pred - y).pow(2).sum().item()
	print(t, loss)
	
	# append to plot lists
	xs.append(t)
	ys.append(loss)

	# backprop to compute gradients of w1 and w2 with respect to loss
	grad_y_pred = 2.0  * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)
	
	# uodate weights using gradient descent
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2
	
print("plotting")

import matplotlib.pyplot as plt
plt.plot(xs, ys)
plt.title("loss over iterations")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("loss_over_iters.png")
	
