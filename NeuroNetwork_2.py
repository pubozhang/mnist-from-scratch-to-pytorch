import load_data
import torch
import time

#second version, only use tensor, softmax + cross entropy, two hidden layer, split into network, trainer

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative_from_activation(a):
    return a * (1 - a)

def relu(x):
    #return torch.maximum(x, torch.tensor(0.0))
    return torch.clamp(x, min=0) #wont fail when using gpu

def relu_derivative(z):
    return (z > 0).float()

def softmax(z):
    # z = z - torch.max(z)
    # exp_z = torch.exp(z)
    # return exp_z / torch.sum(exp_z)

    #for real mini batches
    z = z - torch.max(z, dim=0, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / torch.sum(exp_z, dim=0, keepdim=True)

class Network:
    def __init__(self, device):
        self.device = device

        self.input_size = 784
        self.hidden_1_size = 128
        self.hidden_2_size = 64
        self.output_size = 10

        self.W1 = torch.randn(self.hidden_1_size, self.input_size, device=self.device) * torch.sqrt(torch.tensor(2.0 / self.input_size, device=self.device))
        self.b1 = torch.zeros(self.hidden_1_size, 1, device=self.device)

        self.W2 = torch.randn(self.hidden_2_size, self.hidden_1_size, device=self.device) * torch.sqrt(torch.tensor(2.0 / self.hidden_1_size, device=self.device))
        self.b2 = torch.zeros(self.hidden_2_size, 1, device=self.device)

        self.W3 = torch.randn(self.output_size, self.hidden_2_size, device=self.device) * torch.sqrt(torch.tensor(1.0 / self.hidden_2_size, device=self.device))
        self.b3 = torch.zeros(self.output_size, 1, device=self.device)

        self.x = None

        self.z1 = None
        self.a1 = None

        self.z2 = None
        self.a2 = None

        self.z3 = None
        self.a3 = None

    def feed_forward(self, x):
        self.x = x

        self.z1 = self.W1 @ self.x + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.W3 @ self.a2 + self.b3
        #self.a3 = sigmoid(self.z3)
        self.a3 = softmax(self.z3)

    def backward_prop(self, y):
        #output layer, 64 -> 10
        #dC_da3 = 2 * (self.a3 - y)
        #da3_dz3 = sigmoid_derivative_from_activation(self.a3)
        dz3_dW3 = self.a2

        #delta3 = dC_da3 * da3_dz3#dC_dz3
        delta3 = self.a3 - y

        dC_dW3 = delta3 @ dz3_dW3.T
        dC_db3 = delta3

        #layer 2
        dC_da2 = self.W3.T @ delta3
        da2_dz2 = relu_derivative(self.z2)
        dz2_dW2 = self.a1

        delta2 = dC_da2 * da2_dz2#dC_dz2

        dC_dW2 = delta2 @ dz2_dW2.T
        dC_db2 = delta2

        #layer 1
        dC_da1 = self.W2.T @ delta2
        da1_dz1 = relu_derivative(self.z1)
        dz1_dW1 = self.x

        delta1 = dC_da1 * da1_dz1#dC_dz1

        dC_dW1 = delta1 @ dz1_dW1.T
        dC_db1 = delta1

        return [dC_dW1, dC_db1, dC_dW2, dC_db2, dC_dW3, dC_db3]

    def apply_gradients(self, grads, learning_rate):
        self.W1 -= learning_rate * grads[0]
        self.b1 -= learning_rate * grads[1]

        self.W2 -= learning_rate * grads[2]
        self.b2 -= learning_rate * grads[3]

        self.W3 -= learning_rate * grads[4]
        self.b3 -= learning_rate * grads[5]

    def predict(self, x):
        self.feed_forward(x)
        return torch.argmax(self.a3).item()

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

    def prepare_train_sample(self, index):
        x = self.train_images[index].reshape(784, 1)
        y = torch.zeros(10, 1, device=self.device)
        y[self.train_labels[index]] = 1
        return x, y

    def train_one_batch(self, data_indexes):
        gradients = None
        for index in data_indexes:
            x, y = self.prepare_train_sample(index)
            self.network.feed_forward(x)

            grad = self.network.backward_prop(y)

            if gradients is None:
                gradients = [torch.zeros_like(g) for g in grad]

            for i in range(len(gradients)):
                gradients[i] += grad[i]

        for i in range(len(gradients)):
            gradients[i] /= self.batch_size

        self.network.apply_gradients(gradients, self.learning_rate)

    def train_one_epoch(self):
        train_data_size = len(self.train_images)
        shuffled_indexes = torch.randperm(train_data_size, device=self.device)
        for batch_start_index in range(0, train_data_size, self.batch_size):
            batch_indexes = shuffled_indexes[batch_start_index : (batch_start_index + self.batch_size)]
            self.train_one_batch(batch_indexes)

            if batch_start_index % 1000 == 0:
                print(f"Trained {batch_start_index}/{train_data_size} samples.")

    def evaluate(self, max_samples=None):
        correct = 0
        if max_samples is None:
            max_samples = len(self.test_images)
        for i in range(max_samples):
            x = self.test_images[i].reshape(784, 1)
            label = self.test_labels[i].item()

            prediction = self.network.predict(x)

            if label == prediction:
                correct += 1
        accuracy = correct / max_samples
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def train(self, times):
        for i in range(times):
            print(f"Training epoch {i+1}/{times}")
            self.train_one_epoch()

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    device = torch.device("cuda")

    network2 = Network(device)
    trainer2 = Trainer(network2, device)

    start_time = time.perf_counter()

    trainer2.train(1)

    end_time = time.perf_counter()

    running_time = end_time - start_time
    print(f"Trained for {running_time:.6f} seconds")

    # network2.save_model("models/last_model")
    trainer2.evaluate()

    # network2.load_model("models/last_model")
    # trainer2.evaluate()


