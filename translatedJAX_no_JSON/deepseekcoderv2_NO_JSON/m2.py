import jax
import jaxlib.xla_client as xc
from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten
import flax.linen as nn
import optax
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import requests

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN Model using Flax
class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
params = model.init(random.PRNGKey(0), jnp.zeros([1, 32, 32, 3]))['params']
loss_fn = lambda params, images, labels: jnp.mean(nn.softmax_cross_entropy({'logits': model.apply(params, images)}, labels))
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in train_loader:
        images, labels = batch
        images = jnp.array(images)
        labels = jnp.array(labels)
        
        # Compute gradients and update parameters
        grads = grad(loss_fn)(params, images, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_fn(params, images, labels):.4f}")

# Evaluate on the test set
correct = 0
total = 0
for batch in test_loader:
    images, labels = batch
    images = jnp.array(images)
    labels = jnp.array(labels)
    
    predictions = model.apply(params, images)
    correct += (jnp.argmax(predictions, axis=1) == labels).sum()
    total += labels.shape[0]

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")