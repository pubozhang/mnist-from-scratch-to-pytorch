import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn

#now use torch.nn, but not nn.Linear

def relu(x):
    #return torch.maximum(x, torch.tensor(0.0))
    return torch.clamp(x, min=0) #wont fail when using gpu

def relu_derivative(z):
    return (z > 0).float()

def softmax(z):
    z = z - torch.max(z, dim=0, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / torch.sum(exp_z, dim=0, keepdim=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 784
        self.hidden_1_size = 128
        self.hidden_2_size = 64
        self.output_size = 10

        self.W1 = nn.Parameter(torch.randn(self.hidden_1_size, self.input_size) * torch.sqrt(torch.tensor(2.0 / self.input_size)))
        self.b1 = nn.Parameter(torch.zeros(self.hidden_1_size, 1))

        self.W2 = nn.Parameter(torch.randn(self.hidden_2_size, self.hidden_1_size) * torch.sqrt(torch.tensor(2.0 / self.hidden_1_size)))
        self.b2 = nn.Parameter(torch.zeros(self.hidden_2_size, 1))

        self.W3 = nn.Parameter(torch.randn(self.output_size, self.hidden_2_size) * torch.sqrt(torch.tensor(1.0 / self.hidden_2_size)))
        self.b3 = nn.Parameter(torch.zeros(self.output_size, 1))

        self.x = None

        self.z1 = None
        self.a1 = None

        self.z2 = None
        self.a2 = None

        self.z3 = None
        self.a3 = None

    def forward(self, X):
        self.x = X

        self.z1 = self.W1 @ self.x + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.W3 @ self.a2 + self.b3

        return self.z3

    def predict_batch(self, X):
        logits = self.forward(X)
        return torch.argmax(logits, dim=0)

class Trainer:
    def __init__(self, network_2, device):
        self.network = network_2
        self.device = device

        self.train_images = torch.tensor(load_data.load_images("train-images-idx3-ubyte.gz"), dtype=torch.float32, device=self.device)
        self.train_labels = torch.tensor(load_data.load_labels("train-labels-idx1-ubyte.gz"), dtype=torch.long, device=self.device)

        self.test_images = torch.tensor(load_data.load_images("t10k-images-idx3-ubyte.gz"), dtype=torch.float32, device=self.device)
        self.test_labels = torch.tensor(load_data.load_labels("t10k-labels-idx1-ubyte.gz"), dtype=torch.long, device=self.device)

        self.batch_size = 64
        self.learning_rate = 0.1

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def prepare_train_batch(self, indexes):
        real_batch_size = indexes.shape[0]

        X = self.train_images[indexes].reshape(real_batch_size, 784).T
        labels = self.train_labels[indexes]

        return X, labels

    def train_one_batch(self, data_indexes):
        X, labels = self.prepare_train_batch(data_indexes)

        #logits = self.network.feed_forward(X)
        logits = self.network(X)

        loss = F.cross_entropy(logits.T, labels)#logits.T because cross_entropy in torch need shape: (batch_size, 10)

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

        X = images.reshape(actual_batch_size, 784).T

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
    device = torch.device("cuda")

    network_3 = Network().to(device)
    trainer3 = Trainer(network_3, device)

    trainer3.train(50)
