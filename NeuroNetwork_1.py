import numpy as np
import load_data

#First version, no pytorch, one hidden layer, only one class

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

class Network:
    def __init__(self):
        self.train_images = load_data.load_images(
            "train-images-idx3-ubyte.gz"
        )

        self.train_labels = load_data.load_labels(
            "train-labels-idx1-ubyte.gz"
        )

        self.hidden_layer_length = 128

        self.data_set_size = len(self.train_images)
        self.batch_size = 10
        self.learning_rate = 0.1 #0.5 for sigmoid

        self.current_data_index = 0

        self.a_1 = np.random.randn(784, 1)
        self.weights_1 = np.random.randn(self.hidden_layer_length, 784) * np.sqrt(2.0 / 784)
        #self.bais_1 = np.random.randn(self.hidden_layer_length, 1)
        self.bais_1 = np.zeros((self.hidden_layer_length, 1))

        self.z_1 = None
        self.a_2 = None

        self.weights_2 = np.random.randn(10, self.hidden_layer_length) * np.sqrt(1.0 / self.hidden_layer_length)
        #self.bais_2 = np.random.rand(10, 1)
        self.bais_2 = np.zeros((10, 1))

        self.z_2 = None
        self.result_layer = None

        self.desire_output = np.zeros((10, 1))

        self.update_values()

    def update_values(self):
        self.z_1 = self.weights_1 @ self.a_1 + self.bais_1
        #self.a_2 = sigmoid(self.z_1)
        self.a_2 = relu(self.z_1)

        self.z_2 = self.weights_2 @ self.a_2 + self.bais_2
        self.result_layer = sigmoid(self.z_2)
        #self.result_layer = relu(self.z_2)
        #self.result_layer = softmax(self.z_2)

    def calculate_gradient(self):
        self.update_values()

        dC_da2 = 2 * (self.result_layer - self.desire_output) #10, 1
        da2_dz2 = sigmoid_derivative(self.z_2) #10, 1

        delta_2 = da2_dz2 * dC_da2 #10, 1
        #delta_2 = self.result_layer - self.desire_output #cross entropy

        dz2_dw2 = self.a_2 #16, 1
        dz2_db2 = 1

        dC_dw2 = delta_2 @ dz2_dw2.T
        dC_db2 = delta_2 * dz2_db2

        dC_da1 = self.weights_2.T @ delta_2 #16, 10 @ 10, 1 = 16, 1
        #da1_dz1 = sigmoid_derivative(self.z_1) #16, 1
        da1_dz1 = relu_derivative(self.z_1)

        delta_1 = da1_dz1 * dC_da1 #16, 1

        dz1_dw1 = self.a_1 #784, 1
        dz1_db1 = 1

        dC_dw1 = delta_1 @ dz1_dw1.T
        dC_db1 = delta_1 * dz1_db1

        return dC_dw1, dC_db1, dC_dw2, dC_db2

    def load_in_data_of_index(self, train_data_index):
        self.a_1 = np.reshape(self.train_images[train_data_index], (784, 1))
        self.desire_output = np.zeros((10, 1))
        self.desire_output[self.train_labels[train_data_index]] = 1

    def train_one_batch(self):
        w1_fix = []
        b1_fix = []
        w2_fix = []
        b2_fix = []

        for i in range(self.current_data_index, self.current_data_index + self.batch_size):
            self.load_in_data_of_index(i)
            a, b, c, d = self.calculate_gradient()
            w1_fix.append(a)
            b1_fix.append(b)
            w2_fix.append(c)
            b2_fix.append(d)

        self.current_data_index += self.batch_size

        w1_fix_avg = np.average(w1_fix, axis=0)
        b1_fix_avg = np.average(b1_fix, axis=0)
        w2_fix_avg = np.average(w2_fix, axis=0)
        b2_fix_avg = np.average(b2_fix, axis=0)

        self.weights_1 -= self.learning_rate * w1_fix_avg
        self.bais_1 -= self.learning_rate * b1_fix_avg
        self.weights_2 -= self.learning_rate * w2_fix_avg
        self.bais_2 -= self.learning_rate * b2_fix_avg

        print("Done training " + str(self.batch_size) + " data, current at " + str(self.current_data_index) + " data.")

    def train_i_batch(self, batch_number):
        for i in range(batch_number):
            self.train_one_batch()

    def train_all(self):
        for i in range(self.data_set_size // self.batch_size):
            self.train_one_batch()

    def judge_image(self, image):
        self.a_1 = np.reshape(image, (784, 1))
        self.update_values()
        return self.result_layer

    def judge_image_given_answer(self, image, answer):
        self.a_1 = np.reshape(image, (784, 1))
        self.update_values()
        if self.get_current_result() == answer:
            return True
        else:
            return False

    def get_current_result(self):
        return np.argmax(self.result_layer)

if __name__ == '__main__':
    nw = Network()
    nw.train_all()


