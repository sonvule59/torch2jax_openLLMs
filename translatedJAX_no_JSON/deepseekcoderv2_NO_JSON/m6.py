import jax
import jaxlib
import flax.datasets as datasets
import flax.transforms.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import optax

# Load CIFAR-10 dataset with data augmentation
def transform(image):
    image = preprocessing.random_flip_left_right(image)
    image = preprocessing.random_crop(image, (32, 32), padding=4)
    image = np.array(image).astype('float32') / 255.0
    return (image - 0.5) / 0.5  # Normalize with mean and std

train_dataset = datasets.cifar10('data', train=True, transform=transform)
test_dataset = datasets.cifar10('data', train=False, transform=transform)

# Create DataLoader-like structure for JAX using random number generation for shuffling
def data_loader(dataset, batch_size):
    rng = jax.random.PRNGKey(0)
    num_samples = len(dataset)
    
    def next_batch():
        nonlocal rng
        rng, subrng = jax.random.split(rng)
        perm = jax.random.permutation(subrng, num_samples)
        for i in range(0, num_samples, batch_size):
            batch_perm = perm[i:i+batch_size]
            yield [dataset[j][None, ...] for j in batch_perm]
    
    return next_batch()

train_loader = data_loader(train_dataset, 64)
test_loader = data_loader(test_dataset, 64)

# Display a batch of augmented images
def imshow(images):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    images = (images * std + mean) * 255.0
    images = images.astype(np.uint8)
    plt.imshow(np.transpose(images[0], (1, 2, 0)))
    plt.show()

# Get some random training images
data_iter = iter(train_loader())
images, _ = next(data_iter)

# Show images
imshow(images)