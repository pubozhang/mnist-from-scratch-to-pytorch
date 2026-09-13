from pathlib import Path
import numpy as np
import torch

from CNN_1 import Network, Trainer

CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "cnn_1_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPOCHS = 18
TRAINING_SEED = 0

def load_checkpoint_model():
    """Load the saved model weights."""
    model = Network().to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_model():
    """Train the choosing model and use it immediately without saving a file."""
    print(f"Training on {DEVICE} for {TRAINING_EPOCHS} epoch(s)...")
    torch.manual_seed(TRAINING_SEED)
    model = Network().to(DEVICE)
    trainer = Trainer(model, DEVICE)
    trainer.train(TRAINING_EPOCHS)
    model.eval()
    return model


def predict_digit(image_28x28, model):
    """Predict one digit with the model already loaded by the board."""
    if model is None:
        raise RuntimeError("model is not available")

    image = np.asarray(image_28x28, dtype=np.float32)
    if image.shape != (28, 28):
        raise ValueError(f"expected image shape (28, 28), received {image.shape}")
    if image.max() > 1:
        image = image / 255.0

    X = torch.tensor(image, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities, torch.argmax(logits, dim=1).item()


# Change this line to load_checkpoint_model to use models in models/
# Change this line to train_model to train a model instead.
MODEL_LOADER = load_checkpoint_model
