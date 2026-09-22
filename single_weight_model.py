import random


#This is training data!!
# we have the (value, expected_value)
# so basically we have a list of tuples of training data with what we 
# #give to the "Ai" and what we expect to get back if the "ai" works right.
training_data = [
    (0, 0),
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
    (5, 10)
]

# Very basic random starting weight to get things started.
weight = random.random()

# i believe this is how much we change the weight on each failed answer.
learning_rate = 0.01

# This is the training loop (limited to 1000 cycles).
# We run this training cycle for as long as we need to (preferably a shorter duration)
# in order to get our weight to whatever value it needs to be in order to get our expected response.
for step in range (50):

    # This is a score for how well the model performed in training with a higher value
    # meaning the model made more/greater mistakes during training and a lower score meaning
    # that the model made less/smaller mistakes during training.
    total_error = 0

    # this just gets the values from the training data as x and expected
    # we run a training cycle for each value in the training data
    # i will each iteration fo this for loop a step and each iteration of the prior
    # for loop a training cycle.
    for x, expected in training_data:

        # this is where the "ai" makes an prediction
        prediction = x * weight

        # This is how far off the predication is
        #e.g. if the training data is (3, 6) and the model predicted 4
        # then error is -2
        error = prediction - expected

        # This is where we adjust our weight after a training step
        # so if out training data was (3, 6) and our weight was 1.33 
        # you would get a prediction of 4. that in turn returns a error of -2.
        # which returns a new weight of learning_rate * error * x or in this case
        # 1.33 -= 0.01 * -2 * 3 or a weight value of 1.39
        ##if step % 10 == 0:
        ##    print(f"Weight: {weight:.4f} -= {learning_rate:.4f} * {error:.4f} * {x}")
        ##    print(f"We got a weight of {weight:.4f} which is {error:.4f} away from our expected value of {expected}")
        weight -= learning_rate * error * x
        ##if step % 10 == 0:
        ##    print(f"We subtract the learning_rate * error * x from our weight to get our new weight of {weight:.4f}")

        # here we score our models prediction by squaring it to both make it positive and 
        # to make larger mistakes score higher.
        total_error += error ** 2

    if step % 100 == 0:
        print(f"Step {step}: error={total_error:.4f}")

print()
print(f"Learned weight: {weight:.4f}")

print()
print("Predictions:")

for x in range(10):
    prediction = x * weight
    print(f"{x} -> {prediction:.2f}")