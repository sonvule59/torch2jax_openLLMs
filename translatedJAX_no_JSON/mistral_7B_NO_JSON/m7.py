import numpy as jnp
from jax import random, grad, vmap, jit, lax, nn, no_grad, convex_optimal_update, sin, cos, exp
from flax import linen as nn
from flax.training import train_state
from flax.optim import SGD
from jax.experimental import optimizers
import time

# Load MNIST dataset
import mnist_data

train_ds = mnist_data.fetch_mnist(kind='train', normalize=True)
test_ds = mnist_data.fetch_mnist(kind='t10k', normalize=True)
batch_size = 64
train_batches = train_ds.flat_map(lambda x: x.split(batch_size))
test_batches = test_ds.flat_map(lambda x: x.split(batch_size))

# Define a simple neural network model
class SimpleNN(nn.Module):
    def setup(self):
        self.fc1 = nn.Dense(units=128, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.fc2 = nn.Dense(units=10, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

    def __call__(self, images):
        x = vmap(lambda x: jnp.reshape(x, (-1, 784)))(images)
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
rng_key = random.PRNGKey(0)
model = SimpleNN()
params = model.init(rng_key, jnp.ones((1, 784)), name='params')
optimizer = optimizers.sgd(learning_rate=0.01)
loss_fn = nn.LogSoftmax() + nn.CrossEntropyLoss()

# Training loop with benchmarking
epochs = 5
for epoch in range(epochs):
    start_time = time.time()  # Start time for training
    for images, labels in train_batches:
        with no_grad():
            params = optimizer.apply_gradient(loss_fn, params, images=images, labels=labels)

    end_time = time.time()  # End time for training
    training_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_fn.apply(params, images=images, labels=labels).mean():.4f}, Time: {training_time:.4f}s")

# Evaluate the model on the test set and benchmark the accuracy
rng_key = random.PRNGKey(0)
correct = 0
total = 0
start_time = time.time()  # Start time for testing
for images, labels in test_batches:
    with no_grad():
        logits = model.apply(params, images=images).squeeze(-1)
        _, predicted = jnp.argmax(logits, axis=-1)
        total += labels.size
        correct += jnp.sum(predicted == labels)

end_time = time.time()  # End time for testing
testing_time = end_time - start_time
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")