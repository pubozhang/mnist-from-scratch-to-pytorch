import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#include data loader from torch

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        #3 more layers will increase the accuracy by 0.4%

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        a1 = F.relu(self.layer1(X))
        #a1 = self.dropout(a1)

        a2 = F.relu(self.layer2(a1))
        a2 = self.dropout(a2)

        logits = self.layer3(a2)
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

        train_images = train_images.reshape(-1, 784) #-1 for automatically let torch to reshape the matrix
        test_images = test_images.reshape(-1, 784)

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

    def train(self, epochs):
        for epoch in range(epochs):
            average_loss = self.train_one_epoch()
            print(f"Epoch {epoch + 1}/{epochs}, loss: {average_loss:.4f}")
            self.evaluate()

if __name__ == "__main__":
    #torch.manual_seed(42)

    device = torch.device("cuda")

    network_3 = Network().to(device)
    trainer3 = Trainer(network_3, device)

    trainer3.train(50)
