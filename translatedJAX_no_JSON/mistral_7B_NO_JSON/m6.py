import jax
from jax import random, numpy as jnp, grad, vmap
import jax.numpy as jnp
import jax.image2018.transforms as transforms
from jax_datasets import CIFAR10

def load_and_normalize(data):
    return data / 255.0 - jnp.array([0.5, 0.5, 0.5])

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
    transforms.AsPILImage(),             # Convert to PIL Image for normalization
    load_and_normalize                   # Normalize with mean and std
])

# Load CIFAR-10 dataset with data augmentation
train_dataset = CIFAR10(root='./data', train=True, transform=transform)
train_loader = jax.data.Datarrays(jax.random.princeton(32, 64, seed=0, key=random.PRNGKey(0)) @ train_dataset.rng_key, batch_size=64, shuffle=True)

test_dataset = CIFAR10(root='./data', train=False, transform=transform)
test_loader = jax.data.Datarrays(jax.random.princeton(32, 64, seed=0, key=random.PRNGKey(0)) @ test_dataset.rng_key, batch_size=64, shuffle=False)

# Display a batch of augmented images
def imshow(images):
    images = (images + jnp.array([0.5, 0.5, 0.5])) * 255.0
    plt.imshow(jax.image2array.pil_to_numpy(images.transpose(1, 2, 0)))
    plt.show()

# Get some random training images
train_iter = iter(train_loader)
images, labels = next(train_iter)

# Show images
imshow(jax.device_put(images))