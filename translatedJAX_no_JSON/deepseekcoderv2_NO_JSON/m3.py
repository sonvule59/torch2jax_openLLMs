import jax
from jax import numpy as np
import jax.nn as nn
import jax.example_libraries.optimizers as optim
import jaxlib.pytorch_model_zoo as models
import jaxlib.trainer as trainer
import jaxlib.data as data

# Data transformation and loading (similar to PyTorch's setup)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define the training and evaluation function (JAX version)
def train_step(model, optimizer, image, label):
    def loss_fn(params):
        pred = model.apply({"params": params}, image)
        return -np.sum(pred * jax.nn.log_softmax(pred))
    
    grads = jax.grad(loss_fn)(model.params)
    updates, new_optimizer = optimizer.update(grads, model.params, model.opt_state)
    new_model = model.apply({"params": updates}, image)
    return new_model, new_optimizer

def eval_step(model, image, label):
    pred = model.apply({"params": model.params}, image)
    pred_vals = np.argmax(pred, axis=1)
    accuracy = np.mean((pred_vals == label).astype(np.float32))
    return accuracy

def train_test_loop(model, optimizer, epochs=10):
    for epoch in range(epochs):
        for batch in train_loader:
            image, label = batch
            model, optimizer = train_step(model, optimizer, image, label)
        
        accuracy = eval_step(model, next(iter(test_loader))[0])
        print(f"Epoch {epoch}: Test Accuracy = {accuracy}")
    
    return model

# Define the CNN model (JAX version)
class VanillaCNNModel:
    def __init__(self):
        self.conv1 = nn.Conv(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.max_pool(window_shape=(2, 2), strides=(2, 2))
        self.fc1 = nn.Dense(64 * 16 * 16, 128)
        self.fc2 = nn.Dense(128, 10)
        self.relu = nn.relu
    
    def __call__(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape((x.shape[0], -1))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the optimizer (similar to PyTorch's Adam)
optimizer = optim.adam(learning_rate=0.001)
model = VanillaCNNModel()
model = train_test_loop(model, optimizer, epochs=10)