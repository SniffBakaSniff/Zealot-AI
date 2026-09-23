# Learning Rate and Optimizers

### What is learning rate?

For my first models I used this equation:

```python
parameter -= learning_rate * gradient
```

The gradient tells the model which direction a parameter should move and how strongly it affects the loss. The `learning_rate` controls how large of a step to take when adjusting the model's parameters.

For example, a learning rate of `0.1` makes the model take larger steps during training, while a learning rate of `0.001` makes it take much smaller steps.

### Optimizers

The simplest optimizer is **Stochastic Gradient Descent (SGD)**. It is very basic and uses:

```python
parameter -= learning_rate * gradient
```

It's implemented in PyTorch like this:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01  # This is the learning rate
)
```

So every time `optimizer.step()` is called, PyTorch uses the gradients to update all of the model's parameters.

In my first proper PyTorch model I'm using **Adam**, or **Adaptive Moment Estimation**.

Adam still uses gradients, but it keeps information about previous gradients to adapt the updates for each parameter.

Adam keeps track of two key values:

* An estimate of the average direction of recent gradients
* An estimate of the magnitude/scale of recent gradients

This allows Adam to adjust the update size for individual parameters.

Adam is commonly used because it can make training easier and often requires less manual tuning than basic SGD.
