import load_data
import torch

#same as network 3, but use auto grad, delete backward_prop function, add new loss function

def relu(x):
    #return torch.maximum(x, torch.tensor(0.0))
    return torch.clamp(x, min=0) #wont fail when using gpu

def relu_derivative(z):
    return (z > 0).float()

def softmax(z):
    z = z - torch.max(z, dim=0, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / torch.sum(exp_z, dim=0, keepdim=True)

def cross_entropy_from_logits(logits, Y):
    batch_size = Y.shape[1]

    logits = logits - torch.max(logits, dim=0, keepdim=True).values
    exp_logits = torch.exp(logits)

    probability = exp_logits / torch.sum(exp_logits, dim=0, keepdim=True)

    log_probability = torch.log(probability + 1e-10)#1e-10 is to prevent log(0)

    loss = -torch.sum(Y * log_probability) / batch_size

    return loss

class Network:
    def __init__(self, device):
        self.device = device

        self.input_size = 784
        self.hidden_1_size = 128
        self.hidden_2_size = 64
        self.output_size = 10

        self.W1 = (torch.randn(self.hidden_1_size, self.input_size, device=self.device) * torch.sqrt(torch.tensor(2.0 / self.input_size, device=self.device))).requires_grad_()
        self.b1 = torch.zeros(self.hidden_1_size, 1, device=self.device, requires_grad=True)

        self.W2 = (torch.randn(self.hidden_2_size, self.hidden_1_size, device=self.device) * torch.sqrt(torch.tensor(2.0 / self.hidden_1_size, device=self.device))).requires_grad_()
        self.b2 = torch.zeros(self.hidden_2_size, 1, device=self.device, requires_grad=True)

        self.W3 = (torch.randn(self.output_size, self.hidden_2_size, device=self.device) * torch.sqrt(torch.tensor(1.0 / self.hidden_2_size, device=self.device))).requires_grad_()
        self.b3 = torch.zeros(self.output_size, 1, device=self.device, requires_grad=True)

        self.x = None

        self.z1 = None
        self.a1 = None

        self.z2 = None
        self.a2 = None

        self.z3 = None
        self.a3 = None

    def feed_forward(self, X):
        self.x = X

        self.z1 = self.W1 @ self.x + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.W3 @ self.a2 + self.b3
        #not do softmax here

        return self.z3

    def apply_gradients(self, learning_rate):
        with torch.no_grad():
            self.W1 -= learning_rate * self.W1.grad
            self.b1 -= learning_rate * self.b1.grad

            self.W2 -= learning_rate * self.W2.grad
            self.b2 -= learning_rate * self.b2.grad

            self.W3 -= learning_rate * self.W3.grad
            self.b3 -= learning_rate * self.b3.grad

    def zero_grad(self):
        self.W1.grad.zero_()
        self.b1.grad.zero_()

        self.W2.grad.zero_()
        self.b2.grad.zero_()

        self.W3.grad.zero_()
        self.b3.grad.zero_()

    def predict_batch(self, X):
        logits = self.feed_forward(X)
        return torch.argmax(logits, dim=0)

    def save_model(self, path):
        torch.save({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)

        self.W1 = checkpoint["W1"].to(self.device)
        self.b1 = checkpoint["b1"].to(self.device)

        self.W2 = checkpoint["W2"].to(self.device)
        self.b2 = checkpoint["b2"].to(self.device)

        self.W3 = checkpoint["W3"].to(self.device)
        self.b3 = checkpoint["b3"].to(self.device)

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

    def prepare_train_batch(self, indexes):
        real_batch_size = indexes.shape[0]

        X = self.train_images[indexes].reshape(real_batch_size, 784).T
        Y = torch.zeros(10, real_batch_size, device=self.device)

        labels = self.train_labels[indexes]
        columns = torch.arange(real_batch_size, device=self.device)#list of 0 to batch size

        Y[labels, columns] = 1.0

        return X, Y

    def train_one_batch(self, data_indexes):
        X, Y = self.prepare_train_batch(data_indexes)

        logits = self.network.feed_forward(X)

        loss = cross_entropy_from_logits(logits, Y)

        loss.backward()

        self.network.apply_gradients(self.learning_rate)

        self.network.zero_grad()

        return loss.item()

    def train_one_epoch(self):
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
        correct = 0
        total = 0
        if max_samples is None:
            max_samples = len(self.test_images)
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

    network_3 = Network(device)
    trainer3 = Trainer(network_3, device)

    trainer3.train(50)
