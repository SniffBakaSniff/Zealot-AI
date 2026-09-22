import random


# Here we are creating a reusable function that hanbdles the the prediction
# it does the same thing as `prediction = x1 * weight1 + x2 * weight2`
# Thats it. nothing fancy it just a function that takes inputs and weights and solves.
def neuron(inputs, weights, bias):
    # If our output has a abse value of zero then if the input was zero the output would also be zero,
    # however with bias we can have a non-zero value even if the prediciton would have otherwise been zero.
    output = bias

    for x, weight in zip(inputs, weights):
        output += x * weight

    return output

def relu(x):
    return max(0, x)

inputs = [5]
weights = [10]
bias = -20

value = 12

output = relu(value)

print(output)

#output = neuron(inputs, weights, bias)

print(f"Inputs:  {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print(f"Output:  {output}")