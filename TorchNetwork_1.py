import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn

#start to use many torch features, like nn.Linear

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, X):
        a1 = F.relu(self.layer1(X))
        a2 = F.relu(self.layer2(a1))
        logits = self.layer3(a2)
        return logits

    def predict_batch(self, X):
        logits = self(X)
        return torch.argmax(logits, dim=1) #standard pytorch shape (batch_size, classes)

class Trainer:
    def __init__(self, network_2, device):
        self.network = network_2
        self.device = device

        self.train_images = torch.tensor(load_data.load_images("train-images-idx3-ubyte.gz"), dtype=torch.float32, device=self.device)
        self.train_labels = torch.tensor(load_data.load_labels("train-labels-idx1-ubyte.gz"), dtype=torch.long, device=self.device)

        self.test_images = torch.tensor(load_data.load_images("t10k-images-idx3-ubyte.gz"), dtype=torch.float32, device=self.device)
        self.test_labels = torch.tensor(load_data.load_labels("t10k-labels-idx1-ubyte.gz"), dtype=torch.long, device=self.device)

        self.batch_size = 16
        self.learning_rate = 0.1

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def prepare_train_batch(self, indexes):
        real_batch_size = indexes.shape[0]

        X = self.train_images[indexes].reshape(real_batch_size, 784) #no need for .T since we are using standard pytorch shape
        labels = self.train_labels[indexes]

        return X, labels

    def train_one_batch(self, data_indexes):
        X, labels = self.prepare_train_batch(data_indexes)

        #logits = self.network.feed_forward(X)
        logits = self.network(X)

        #dont apply softmax before cross_entropy
        loss = F.cross_entropy(logits, labels)#no need for logits.T, same reason

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_one_epoch(self):
        self.network.train()

        train_data_size = len(self.train_images)
        shuffled_indexes = torch.randperm(train_data_size, device=self.device)
        for batch_start_index in range(0, train_data_size, self.batch_size):
            batch_indexes = shuffled_indexes[batch_start_index : (batch_start_index + self.batch_size)]
            self.train_one_batch(batch_indexes)

            if batch_start_index % 1000 == 0:
                print(f"Trained {batch_start_index}/{train_data_size} samples.")

    def prepare_test_batch(self, start, batch_size):
        images = self.test_images[start:(start + batch_size)]
        labels = self.test_labels[start:(start + batch_size)]

        actual_batch_size = images.shape[0]

        X = images.reshape(actual_batch_size, 784) #no need for .T here too for same reason

        return X, labels

    def evaluate(self, max_samples=None):
        self.network.eval()

        correct = 0
        total = 0
        if max_samples is None:
            max_samples = len(self.test_images)

        with torch.no_grad():
            for start in range(0, max_samples, self.batch_size):
                X, labels = self.prepare_test_batch(start, self.batch_size)

                predictions = self.network.predict_batch(X)

                correct += torch.sum(predictions == labels).item()

                total += labels.shape[0]

        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def train(self, times):
        for i in range(times):
            print(f"Training epoch {i+1}/{times}")
            self.train_one_epoch()
            self.evaluate()

if __name__ == "__main__":
    #torch.manual_seed(42)

    device = torch.device("cuda")

    network_3 = Network().to(device)
    trainer3 = Trainer(network_3, device)

    trainer3.train(50)
