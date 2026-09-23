Generated with ChatGpt because i was tired and dint wanna do it myself cuz id already spent 12 hours studying :P
iv learned nothing :O

# Custom Neural Networks with PyTorch

So far, we have used `nn.Sequential` to create neural networks. PyTorch also allows us to define our own neural-network classes.

## `nn.Module`

A custom neural network is created by inheriting from `nn.Module`.

```python
import torch
import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)

        return x
```

`nn.Module` is the base class used by PyTorch for neural-network models.

When layers are assigned to `self`, PyTorch knows they belong to the model and automatically tracks their parameters.

```python
model = MyNetwork()
```

## `forward()`

The `forward()` method defines how data flows through the network.

For our model:

```text
input
  ↓
layer1
  ↓
ReLU
  ↓
layer2
  ↓
output
```

For example:

```python
x = torch.tensor([[3.0]])
prediction = model(x)
```

The value `3.0` is passed through:

```text
3.0
 ↓
Linear(1 → 10)
 ↓
10 values
 ↓
ReLU
 ↓
10 values
 ↓
Linear(10 → 1)
 ↓
1 prediction
```

`forward()` does not specifically mean "process one input." The `x` represents whatever data is passed to the model.

## Model Parameters

Our model has two layers:

```python
self.layer1 = nn.Linear(1, 10)
self.layer2 = nn.Linear(10, 1)
```

### First layer

```text
1 input → 10 neurons
```

Each of the 10 neurons needs:

* 1 weight
* 1 bias

Therefore:

```text
10 weights + 10 biases = 20 parameters
```

### Second layer

```text
10 inputs → 1 neuron
```

The output neuron needs:

* 10 weights
* 1 bias

Therefore:

```text
10 weights + 1 bias = 11 parameters
```

Total:

```text
20 + 11 = 31 parameters
```

So this tiny model has 31 learnable parameters.

## Tensor Shapes

PyTorch tensors have shapes describing their dimensions.

For example:

```python
x = torch.tensor([[3.0]])
```

has shape:

```text
[1, 1]
```

This means:

```text
1 example × 1 feature
```

Our network transforms the shape like this:

```text
[1, 1]
   ↓
Linear(1 → 10)
   ↓
[1, 10]
   ↓
ReLU
   ↓
[1, 10]
   ↓
Linear(10 → 1)
   ↓
[1, 1]
```

The ReLU activation changes the values but does not change their shape.

## Batches

The first dimension can represent multiple examples.

For example:

```text
[1, 1]     → 1 example, 1 feature
[5, 1]     → 5 examples, 1 feature
[100, 1]   → 100 examples, 1 feature
```

The same model can process all of them.

For five examples:

```text
[5, 1]
  ↓
[5, 10]
  ↓
[5, 10]
  ↓
[5, 1]
```

The model still has exactly **31 parameters**. It is simply processing five examples at once.

## Sequential vs Custom Models

`nn.Sequential` lets us quickly describe a simple chain:

```python
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
```

A custom `nn.Module` gives us more control:

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)

        return x
```

Both can represent the same architecture.

The custom version becomes useful when we need more complicated behavior, such as:

* branching paths
* skip connections
* attention
* custom calculations
* multiple inputs or outputs
* transformer architectures

## Important Idea

A neural network has two separate concepts:

**Architecture**

How the model is constructed and how information flows.

```text
input → layer → activation → layer → output
```

**Parameters**

The learned weights and biases inside those layers.

The architecture determines **what the model is capable of doing**.

The parameters determine **what the model has learned to do**.
