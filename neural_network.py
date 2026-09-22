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

# Our training data with an underlying relationship of `output = x * 2 + 1`
training_data = [
    (1, 3),
    (2, 5),
    (3, 7),
    (4, 9),
    (5, 11),
]

# Our training data has to input values therefore we need 1 weights
weight = random.random()

bias = random.random()

learning_rate = 0.01

for step in range(1000):
    total_error = 0

    for input, expected in training_data:

        prediction = input * weight + bias

        # Get the error
        error = prediction - expected

        # Get the gradients
        weight_gradient = error * input
        bias_gradient = error

        # Update Parameters
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

        # Increase error score.
        total_error += error ** 2

    if step % 100 == 0:
        print(f"Step {step}: error={total_error:.4f}")

print()
print(f"Weight: {weight:.4f}")
print(f"Bias:   {bias:.4f}")

print()
print("Predictions:")

for input, expected in training_data:
    prediction = input * weight + bias

    print(
        f"{input} -> {prediction:.2f} "
        f"(expected {expected})"
    )

print()
print("New predictions:")

for input in [6, 7, 10, 20, 100]:
    prediction = input * weight + bias
    print(f"{input} -> {prediction:.2f}")


