import torch
import torch.nn as nn
import torch.optim as optim
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

# @ray.remote
def train_tune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    for epoch in range(100):
        # Dummy data
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to(device)
        labels = torch.tensor([[2.0], [3.0]], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return loss

test_remote = ray.remote(train_tune)

if __name__ == "__main__":
    ray.init()
    start_time = timeit.default_timer()
    lr_dict = {"lr":0.01}

    futures = [test_remote.remote(lr_dict) for i in range(10)]
    print(ray.get(futures))
    print(timeit.default_timer() - start_time)


