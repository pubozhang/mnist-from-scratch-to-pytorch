# MNIST Neural Networks: From Scratch to PyTorch

This project explores handwritten-digit classification on MNIST by building neural networks from a low-level NumPy implementation to a convolutional neural network (CNN) in PyTorch. The intermediate implementations are intentionally preserved to show what I learned at each stage. The final CNN achieves over 99% MNIST test accuracy and can be tried through an interactive drawing board.

## Major Features

- Loads the original gzip-compressed MNIST IDX files without `torchvision`.
- Builds the classifier step by step, from manual backpropagation to a two-block CNN.
- Provides a Pygame drawing board that centers a handwritten digit and predicts it with the CNN.
- Includes a custom 3 x 3 convolution control for experimenting with the drawing.

## Learning Progression

| Stage | Files | Main idea |
| --- | --- | --- |
| 1 | `NeuroNetwork_1.py` | NumPy forward propagation and manual backpropagation in a 784 -> 128 -> 10 network. |
| 2 | `NeuroNetwork_2.py` | PyTorch tensors with manually calculated, per-sample gradients in a 784 -> 128 -> 64 -> 10 network. |
| 3 | `NeuroNetwork_3.py` | Vectorized mini-batch training and an accuracy plot. |
| 4 | `NeuroNetwork_4.py` | Autograd with a custom cross-entropy calculation and explicit parameter updates. |
| 5 | `NeuroNetwork_5.py` | PyTorch's built-in cross-entropy loss and SGD optimizer. |
| 6 | `NeuroNetwork_6.py` | Custom parameters registered inside `nn.Module`. |
| 7 | `TorchNetwork_1.py` | Standard `nn.Linear` layers and PyTorch batch shapes. |
| 8 | `TorchNetwork_2.py` | `TensorDataset`, `DataLoader`, and dropout. |
| 9 | `CNN_1.py` | Convolution, ReLU, and max pooling followed by fully connected classification layers. |

The overlap between files is intentional: each version keeps the previous stage recognizable while replacing a small part of the training pipeline with a higher-level abstraction.

## Final CNN and Interactive Board

`CNN_1.Network` takes a normalized 28 x 28 grayscale image and applies:

```text
Conv2d(1, 32, 3 x 3) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, 3 x 3) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(3136, 256) -> ReLU -> Linear(256, 10)
```

`board.py` downsamples and centers a mouse-drawn digit, then passes it to `inference.py` for prediction with `CNN_1.Network`. Earlier models are separate learning stages, not drawing-board backends.

## Results and Reproducibility

| Model | MNIST test accuracy |
| --- | ---: |
| Best saved `CNN_1.Network` | **99.41%** |

The included pretrained CNN state dict is stored at `checkpoints/cnn_1_best.pt`.

The best saved model reached 99.41% test accuracy at epoch 18 using random seed 0. Exact results may vary across different PyTorch, CUDA, and hardware environments.

## Quick Start

```bash
git clone https://github.com/pubozhang/mnist-from-scratch-to-pytorch.git
cd mnist-from-scratch-to-pytorch
python -m pip install -r requirements.txt
python board.py
```

The four MNIST data files are included in `data/`, and the pretrained CNN checkpoint is included at `checkpoints/cnn_1_best.pt`. The drawing board loads this checkpoint by default without retraining. `inference.py` selects CUDA when available and otherwise uses CPU.

## Using the Board

- Hold the left mouse button to draw.
- Press `R` to clear the canvas.
- Press `L` to apply the custom convolution; it can be pressed repeatedly.
- Press `Space` to predict. The digit is printed in the terminal.

To train a fresh model instead of loading weights, change `MODEL_LOADER` in `inference.py` to `train_model`. The learning-stage scripts are also independently runnable; for example, `python NeuroNetwork_1.py` runs the NumPy model, while `python CNN_1.py` runs the original CNN training script (which currently requires CUDA).

## Technologies

Python, NumPy, PyTorch, Matplotlib, Pygame, and Python's built-in `gzip` module. The unpinned external dependencies are listed in `requirements.txt`.

## What I Learned

Implementing the same task at each abstraction level helped me understand tensor shapes, backpropagation, mini-batch training, and evaluation before relying on PyTorch's higher-level tools. Building the drawing board also showed me how preprocessing and model inference connect to an interactive interface.

## Attribution

The bundled dataset is the [MNIST database of handwritten digits](https://yann.lecun.org/exdb/mnist/) by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges, derived from NIST data. This project uses [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), [Matplotlib](https://matplotlib.org/), and [Pygame](https://www.pygame.org/).
