# How a Weight Affects the Error

### Part 1
In these notes im going to wright down how weights affect the error and how to get the change in weight.

Lets say we have :
```
input = 2
weight = 0.5
bias = 0
```

our neuron calculates `2 * 0.5 + 0 = 0.5` but lets say our expected value is `5`.
```
prediction = 1
expected = 5
error = -4 or prediction - expected
```
We now know that the prediction is too low.
But how much do we change the weight?

### Part 2
Currently we have `weight = 0.5` and the prediction is `2 * 0.5 = 1`. What if we increased the weight to `1`?
We would then get `2 * 1 = 2` so by increasing the weight by `0.5` we increase the prediction by `1`. What if we increased the weight to `2`? If we increased the weight to `2` then the equation becomes `2 * 2 = 4` which is much closer to out expected number of `5`. If the weight is `2.5` then we get `2 * 2.5 = 5` which is our expected output.

We can use this using this formula `change in prediction / change in weight = input` so in or prior examples we have `2 * 0.5 = 1` and we adjusted the weight to be `2 * 2.5 = 5` so using this formula we can calculate `4 / 2 = 2` so we know what the input was just form how much our prediction and weights change.

### Part 3 - Weight Gradient
The weight gradient tells us in which direction and how strongly the weight should change. For simple neurons we do `weight_gradient = error * input` so for example of `2 * 0.5 = 1` we get:
```
input = 2
expected = 5
weight = 0.5
prediction = 2 * 0.5 # input * weight
error = 1 - 5 # prediction - expected

weight_gradient = -4 * 2 # error * input
```
so our weight gradient is `-8` and if we use `weight -= learning_rate * weight_gradient` with `learning_rate = 0.1` we get:
```
0.5 -= 0.1 * -8

# Simplified
0.5 -= -0.8

# Simplified
0.5 += 0.8
```

this changes our weight of `0.5` to `1.3` this changes the prediction from `1` to `2 * 1.3 = 2.6`.
This still isn't our expected output but it is closer and by repeating this process we will eventually get out expected response.

### Part 4 Bias Gradient 
This is works similarly to the the weight_gradient. we use `bias -= learning_rate * gradient_bias`. The difference is that changing the bias by `1` also changes the prediction by exactly `1`. We get the `gradient_bias` but using this equation `gradient_bias = error * 1` which simplifies to `gradient_bias = error`. We can therefore update the bias with `bias -= learning_rate * gradient_bias`.