import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
import ray
import timeit

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training function
def train_tune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    for epoch in range(10):
        # Dummy data
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to(device)
        labels = torch.tensor([[2.0], [3.0]], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ray.train.report(dict(loss=loss.item()))

# Define the search space
search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
}

start = timeit.default_timer()
# Configure Ray to use 2 GPUs in total, 1 GPU per trial
analysis = tune.run(
    train_tune,
    config=search_space,
    num_samples=10,
    stop={"training_iteration": 10},
    resources_per_trial={"cpu": 1, "gpu":1},  # Allocate 1 GPU per trial
    local_dir="ray_results",  # Optional: specify a directory for results
    verbose=1  # Set verbosity level
)

print("Best hyperparameters found were: ", analysis.get_best_config(metric= 'loss', mode = 'min'))
print("Total time:", timeit.default_timer() - start)