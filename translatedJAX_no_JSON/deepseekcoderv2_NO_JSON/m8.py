import jax
import flax.linen as nn
from flax.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load MNIST dataset
def transform(dataset):
    def normalize_img(image, label):
        return tfds.as_numpy(tfds.features.normalize('center', (0.5,)), image), label
    return dataset.map(normalize_img)

train_dataset = tfds.load('mnist', split='train').cache().apply(transform)
test_dataset = tfds.load('mnist', split='test').cache().apply(transform)

# Prepare data loaders
def create_loader(dataset, batch_size):
    def map_fn(x):
        return jax.numpy.array(x['image']), jax.numpy.array(x['label'])
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(jax.local_device_count() * 4)
    return iter(dataset)

train_loader = create_loader(train_dataset, batch_size=64)
test_loader = create_loader(test_dataset, batch_size=64)

# Define an Autoencoder model using Flax
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # Downsample to 14x14
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # Downsample to 7x7
        
        # Decoder
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', output_padding=(1, 1))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2, 2), padding='SAME', output_padding=(1, 1))(x)
        x = nn.sigmoid(x)  # To keep pixel values between 0 and 1
        
        return x

# Initialize the model, loss function, and optimizer
model = Autoencoder()
params = model.init(jax.random.PRNGKey(0), jnp.zeros([1, 28, 28, 1]))['params']
optimizer = Adam().create(params)

# Define the loss function
def compute_loss(params, images):
    reconstructed = model.apply({'params': params}, images)
    return jnp.mean((reconstructed - images) ** 2)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in train_loader:
        images, _ = batch
        loss, grads = jax.value_and_grad(compute_loss)(optimizer.target, images)
        optimizer = optimizer.apply_gradient(grads)
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")