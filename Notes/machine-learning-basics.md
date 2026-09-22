# My LLM Notes

# Machine Learning Basics

## Model

A model is a mathematical system with parameters that can be adjusted during training.

## Training

Training is the process of giving the model input data that it then attempts to predict the expected response. We then take the prediction and calculate the error/loss value of the prediction compared to the expected response. Taking this new error/loss value, we adjust the parameters and try again.

## Weights

The weights are values that the model changes during training. The model uses the weights against/with the input values to make a prediction.

Example:

```py
prediction = x * weight
```

The model starts with a random weight that adjusts with each step of the training until the model's predictions are as close to the expected value as possible.

## Loss / Error

We measure how wrong the model's predictions were.

My first model `single_weight_model.py` uses:

```py
error = prediction - expected
```

and

```py
total_error = error ** 2
```

For `total_error`, we square the error value to make worse predictions score higher and also to make sure there are no negative values that would cancel out positive values.

## Learning Rate

The learning rate is a value that controls how much the weights change during each step of training.

In my model I used:

```py
weight -= learning_rate * error * x
```

with a `learning_rate = 0.01`.

The `0.01` controls how large the adjustment to the weight is. A smaller learning rate means smaller adjustments, while a larger learning rate means larger adjustments.


### Multiple Weights

Models can have multiple weight values to increase the amount of tasks it can do. A model with 2 weight values can have 2 input values for a prediction `((x1, x2), expected)`. We can use multiple weights per prediction like this:

```py
prediction = x1 * weight1 + x2 * weight2
```


### Training Data
Training data is data that the is used to train a model. The training data must have a relationship the model can represent.

Example:
for a relation of `((x1 * 2, x2 * 3), expected)`

we can have training data like this:
```py
training_data = [
    ((1, 2), 8),
    ((2, 1), 7),
    ((3, 2), 12),
    ((4, 2), 14),
    ((5, 3), 19),
]
```

this training data works as it follows the relationship.
1 * 2 + 2 * 3 = 8


### Neurons

A neuron is simply a function that takes the inputs and returns an output.

### Bias

A bias is another parameter that the model can adjust during training.

It is added to the weighted inputs:
```py
output = (input * weight) + bias
```

The bias allows a neuron to shift its output independently of the input.

For example:

```py
input = 0
weight = 10
bias = 7

output = 0 * 10 + 7
output = 7
```

Without a bias, an input of 0 would always produce an output of 0.

### Activation Functions

An activation function is applied to a neurons output after the weighted inputs and bias have been calculated.
Activation function introduce non-linearity, allowing neural networks to represent more complex relationships.

### ReLu

ReLU (Rectified Linear Unit) returns the input if it is positive, otherwise it returns 0.

```py
def relu(x):
    return max(0, x)

#Examples:

relu(5) = 5
relu(0) = 0
relu(-5) = 0
```