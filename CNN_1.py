import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#turn into CNN

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#stride=2 to replace max pooling, lower the accuracy by 0.6%

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        X = self.conv1(X) #->28*28
        X = F.relu(X)
        X = self.pool(X) #->14*14

        X = self.conv2(X) #->14*14
        X = F.relu(X)
        X = self.pool(X) #->7*7

        X = torch.flatten(X, 1)

        X = self.fc1(X)
        X = F.relu(X)

        #X = self.dropout(X) #increase accuracy by about 0.2%

        logits = self.fc2(X)

        return logits

class Trainer:
    def __init__(self, network_2, device):
        self.network = network_2
        self.device = device

        self.batch_size = 64
        self.learning_rate = 0.1

        train_images = torch.tensor(load_data.load_images("train-images-idx3-ubyte.gz"), dtype=torch.float32, device=self.device)
        train_labels = torch.tensor(load_data.load_labels("train-labels-idx1-ubyte.gz"), dtype=torch.long, device=self.device)

        test_images = torch.tensor(load_data.load_images("t10k-images-idx3-ubyte.gz"), dtype=torch.float32, device=self.device)
        test_labels = torch.tensor(load_data.load_labels("t10k-labels-idx1-ubyte.gz"), dtype=torch.long, device=self.device)

        train_images = train_images.unsqueeze(1) #to prepare for conventional layer
        test_images = test_images.unsqueeze(1)

        self.train_dataset = TensorDataset(train_images, train_labels)
        self.test_dataset = TensorDataset(test_images, test_labels)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def train_one_batch(self, X, labels):
        X = X.to(self.device)
        labels = labels.to(self.device)

        logits = self.network(X)

        #dont apply softmax before cross_entropy
        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_one_epoch(self):
        self.network.train()

        total_loss = 0
        total_samples = 0

        for batch_number, (X, labels) in enumerate(self.train_loader):
            loss = self.train_one_batch(X, labels)

            current_batch_size = X.shape[0]

            total_loss += loss * current_batch_size
            total_samples += current_batch_size

            if batch_number % 100 == 0:
                trained_samples = min(batch_number * self.batch_size, len(self.train_dataset))
                print(f"Trained {trained_samples}/{len(self.train_dataset)} samples.")

        average_loss = total_loss / total_samples

        return average_loss

    def evaluate(self, max_samples=None):
        self.network.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X, labels in self.test_loader:
                X = X.to(self.device)
                labels = labels.to(self.device)

                logits = self.network(X)
                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.shape[0]

                if max_samples is not None and total >= max_samples:
                    break

        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    # def evaluate_external_data(self, external_X):
    #     self.network.eval()
    #
    #     # Accept NumPy array or PyTorch tensor
    #     if not isinstance(external_X, torch.Tensor):
    #         external_X = torch.tensor(external_X, dtype=torch.float32)
    #     else:
    #         external_X = external_X.to(dtype=torch.float32)
    #
    #     # Accept: (28, 28), (784, 1), (784,)
    #     if external_X.numel() != 784:
    #         raise ValueError(f"Expected 784 values, but got {external_X.numel()}")
    #
    #     # CNN requires: [batch, channel, height, width]
    #     external_X = external_X.reshape(1, 1, 28, 28)
    #     external_X = external_X.to(self.device)
    #
    #     with torch.no_grad():
    #         logits = self.network(external_X)
    #
    #         probabilities = torch.softmax(logits, dim=1)
    #
    #         prediction = torch.argmax(logits, dim=1).item()
    #
    #     return prediction, probabilities

    def train(self, epochs):
        for epoch in range(epochs):
            average_loss = self.train_one_epoch()
            print(f"Epoch {epoch + 1}/{epochs}, loss: {average_loss:.4f}")
            self.evaluate()

if __name__ == "__main__":
    torch.manual_seed(1202)

    device = torch.device("cuda")

    network_3 = Network().to(device)
    trainer3 = Trainer(network_3, device)

    trainer3.train(50)
