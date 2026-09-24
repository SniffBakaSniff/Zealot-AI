import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

device = torch.device(
    "cpu"
)

model = MyNetwork().to(device)

best_loss = float("inf")
best_state = None

# Training Data
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
    [10.0],
])

y = torch.tensor([
    [1.0],
    [4.0],
    [9.0],
    [16.0],
    [25.0],
    [36.0],
    [49.0],
    [64.0],
    [81.0],
    [100.0],
])

# Scaling the data
x_min = x.min()
x_max = x.max()

y_min = y.min()
y_max = y.max()

x_scaled = (x - x_min) / (x_max - x_min)
y_scaled = (y - y_min) / (y_max - y_min)

# Making the dataset and mini batching
dataset = TensorDataset(x_scaled, y_scaled)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

loss_function = nn.MSELoss()

# The optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# Training Montage
for epoch in range(5000):

    total_loss = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        prediction = model(batch_x)

        loss = loss_function(
            prediction,
            batch_y
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Average loss across all batches
    epoch_loss = total_loss / len(loader)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}\n"
            f"Loss={epoch_loss:.10f}"
        )

    # Save the best model
    if epoch_loss < best_loss:

        best_loss = epoch_loss

        best_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
        }

# Restore the best model
model.load_state_dict(best_state)

print("Best loss:", best_loss)

print()
print("Final predictions:")

with torch.no_grad():

    prediction_scaled = model(
        x_scaled.to(device)
    )

    final_loss = loss_function(
        prediction_scaled,
        y_scaled.to(device)
    )

    prediction_original = (
        prediction_scaled * (y_max - y_min)
        + y_min
    )

    print("Scaled predictions:")
    print(prediction_scaled)

    print()
    print("Final scaled loss:", final_loss.item())

print()
print("Predictions:")

for input_value, prediction in zip(
    x,
    prediction_original
):
    print(
        f"{input_value.item():.0f} -> "
        f"{prediction.item():.2f}"
    )