import jax
from jax import numpy as jnp
import flax.linen as nn
import optax
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple neural network model using Flax
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.relu(nn.Dense(features=128)(x))
        x = nn.Dense(features=10)(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
optimizer = optax.sgd(learning_rate=0.01)
state = optimizer.init(model.params)

# Define a function to compute gradients and update parameters
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        outputs = model.apply({'params': params}, images)
        return optax.softmax_cross_entropy_with_integer_labels(outputs, labels).mean()
    
    grads = jax.grad(loss_fn)(state.params)
    updates, new_state = optimizer.update(grads, state, model.params)
    new_params = optax.apply_updates(model.params, updates)
    return new_params, new_state

# Training loop with benchmarking
epochs = 5
for epoch in range(epochs):
    start_time = time.time()  # Start time for training
    for batch in train_loader:
        images, labels = batch
        model.params, state = train_step(state, images, labels)
    
    end_time = time.time()  # End time for training
    training_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/{epochs}], Time: {training_time:.4f}s")

# Evaluate the model on the test set and benchmark the accuracy
correct = 0
total = 0
start_time = time.time()  # Start time for testing
for batch in test_loader:
    images, labels = batch
    outputs = model.apply({'params': model.params}, images)
    predicted = jnp.argmax(outputs, axis=1)
    total += labels.shape[0]
    correct += (predicted == labels).sum()

end_time = time.time()  # End time for testing
testing_time = end_time - start_time
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")