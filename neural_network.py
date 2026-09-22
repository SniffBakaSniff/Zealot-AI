import random

def neuron(inputs, weights, bias):
    output = bias
    for x, weight in zip(inputs, weights):
        output += x * weight

    return output

def layer(inputs, weights, biases):
    outputs = []

    for neuron_weights, bias in zip(weights, biases):
        outputs.append(
            neuron(inputs, neuron_weights, bias)
        )

    return outputs

# Our training data with an underlying relationship of `output = (x1 * 2 + x2 * 3) + 1`
training_data = [
    ([1, 2], 9),
    ([2, 1], 8),
    ([3, 2], 13),
    ([4, 1], 12),
    ([5, 3], 20),
]

# Our training data has to input values therefore we need 1 weights
weights = [
    random.random(),
    random.random()
]

bias = random.random()

learning_rate = 0.01

for step in range(1000):
    total_error = 0

    for inputs, expected in training_data:

        prediction = (
            inputs[0] * weights[0]
            + inputs[1] * weights[1]
            + bias
        )

        # Get the error
        error = prediction - expected

        weight1_gradient = error * inputs[0]
        weight2_gradient = error * inputs[1]
        bias_gradient = error

        # Update Parameters
        weights[0] -= learning_rate * weight1_gradient
        weights[1] -= learning_rate * weight2_gradient
        bias -= learning_rate * bias_gradient

        # Increase error score.
        total_error += error ** 2

    if step % 100 == 0:
        print(f"Step {step}: error={total_error:.4f}")

print()
print(f"Weight 1: {weights[0]:.4f}")
print(f"Weight 2: {weights[1]:.4f}")
print(f"Bias:     {bias:.4f}")


