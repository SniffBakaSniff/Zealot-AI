import random


# Our training data is for 2 weights
# so x1 * weight1 + x2 * weight2 == expected
# the tranining in this case has to figure out what two weights work.
# in this data set the expected values are x1 * 2 and x2 * 3
training_data = [
    ((1, 2), 8),
    ((2, 1), 7),
    ((3, 2), 12),
    ((4, 2), 14),
    ((5, 3), 19),
]

weight1 = random.random()
weight2 = random.random()

learning_rate = 0.01

# This training is more difficult for the model as it has to 
# find 2 weight values that when multiplied by their corisponding input value
# then added togeather equals the expected value.
# so lets say we have ((2, 4), 16) the weigths could be weight1 = 0 and weight2 = 4
# which returns solves for the expected value of 16. however if the input values 
# are ((5, 2) 16) with weights of (0, 4) then that would predict 8 instead of 16.
for step in range(1000):

    total_error = 0

    for (x1, x2), expexcted in training_data:

        # Same as with the single weight expect we add the 2 predictions
        # creating a final prediction that should be the expected value.
        prediction = x1 * weight1 + x2 * weight2

        # this operated exactly the same as the single weight model.
        error = prediction - expexcted

        # this also works the same as the single weight model except
        # we have 1 for each weight.
        weight1 -= learning_rate * error * x1
        weight2 -= learning_rate * error * x2

        # Same as single weight model.
        total_error += error ** 2

    if step % 100 == 0:
        print(f"Step: {step}: Error: {total_error:.4f}")

print()
print(f"Weight 1: {weight1:.4f}")
print(f"Weight 2: {weight2:.4f}")

print()
print("Predictions:")

for (x1, x2), expexcted in training_data:
    prediction = x1 * weight1 + x2 * weight2
    print(
        f"{x1}, {x2} -> "
        f"{prediction:.2f} "
        F"(Expected {expexcted})"
    )