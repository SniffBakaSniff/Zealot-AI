def neuron(inputs, weights, bias):
    output = bias
    for x, weight in zip(inputs, weights):
        output += x * weight

    return max(0, output)

def layer(inputs, weights, biases):
    outputs = []

    for neuron_weights, bias in zip(weights, biases):
        outputs.append(
            neuron(inputs, neuron_weights, bias)
        )

    return outputs

inputs = [2, 3]

weights = [
    [1, 2],
    [2, 1],
    [3, 3]
]

biases = [
    1, 
    -1,
    0
]
# inputs [2, 3] and outputs [9, 6, 15]
# we have 3 output values as we have 3 sets of weights.
layer1 = layer(inputs, weights, biases)
print(layer1) 

layer2_weights = [
    [1, 1, 1],
    [1, -1, 1]
]

layer2_biases = [
    0,
    0,
]

# inputs [9, 6, 15] and outputs [3, 18]
# we have 2 output values as we have 2 sets of weights.
layer2 = layer(layer1, layer2_weights, layer2_biases)
print(layer2)

# Layer 1 gets our initial inputs
# neuron(inputs, weights, bias)
# we have the same amount of weights as inputs.
# layer1 = [
#     neuron(inputs, [1, 2], 1),
#     neuron(inputs, [2, 1], -1),
#     neuron(inputs, [3, 3], 0)
# ]

# In this case layer one outputs [9, 6, 15]
# This is using our equation `output = (input1 * weight1 + input2 * weight2) + bias`
# so in this case we get our first output from `output = (2 * 1 + 3 * 2) + 1`
# print("Layer 1:", layer1)

# Layer 2 is a bit different as it receives layer1's output as its input.
# Layer 1 has 3 neurons so it outputs 3 numbers so therefore we have 3 weights.
# for layer 2 our equation is `output = (input1 * weight1 + input2 * weight2 + input3 + weight3) + bias`
# layer2 = [
#     neuron(layer1, [1, 1, 1], 0),
#     neuron(layer1, [1, -1, 1], 0)
# ]

# print("Layer 2:", layer2)