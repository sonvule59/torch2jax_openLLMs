import jax
from jax import grad, jit, random, vmap
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

import numpy as np

def normalize(images):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    return (images - mean) / std

def cifar10_loader(batch_size, data_dir, is_train=True):
    from dataclasses import dataclass
    import os

    IMAGE_SIZE = 32
    NUM_CLASSES = 10

    @dataclass
    class CIFAR10:
        images: jnp.ndarray
        labels: jnp.ndarray

    def _load_cifar10():
        label_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        def read_image(path):
            with open(path, 'rb') as f:
                image = jnp.frombuffer(f.read(), dtype=jnp.uint8)
            image = image.reshape((3, IMAGE_SIZE, IMAGE_SIZE))
            image = normalize(image)
            return image, int(path[-15:][:-4])

        train_images, train_labels = list(map(read_image, sorted(os.listdir(train_dir))))
        test_images, test_labels = list(map(read_image, sorted(os.listdir(test_dir))))

        train_data = CIFAR10(jnp.stack(train_images), jnp.array(train_labels))
        test_data = CIFAR10(jnp.stack(test_images), jnp.array(test_labels))
        return train_data, test_data

    if is_train:
        data = _load_cifar10()
        def batchify(batch_size):
            def fn(data):
                images, labels = jax.tree_leaves(data)
                return images.reshape((len(images), -1, 3, 32, 32)), labels.reshape((-1,))
            return vmap(fn)(jax.split(jnp.arange(len(train_images)), num_or_size_splits=batch_size))
        return batchify(batch_size)
    else:
        data = _load_cifar10(False)
        def batchify(batch_size):
            def fn(data):
                images, labels = jax.tree_leaves(data)
                return images.reshape((len(images), -1, 3, 32, 32)), labels.reshape((-1,))
            return vmap(fn)(jax.split(jnp.arange(len(test_images)), num_or_size_splits=batch_size))
        return batchify(batch_size)

def cifar10_train(model, train_loader, test_loader, epochs=10):
    optimizer = jax.optimizers.Adam(learning_rate=0.001)
    train_losses = []
    test_accuracies = []

    for _ in range(epochs):
        state = optimizer.init(params=model.init_state())
        loss, grads = jit(grad(model.loss))(lambda params: (model.apply(params), train_loader(batch_size)))
        for images, labels in train_loader(batch_size):
            grads, state = jax.value_and_grad(optimizer.update)(grads, params=state, loss=loss)(*images, *labels)
            state = optimizer.apply_gradient(state, grads)
        train_losses.append(loss(state).item())

        if _ % 5 == 0:
            model.apply_init(state)
            test_acc = jit(lambda params: jnp.mean((jax.nn.onehot(params, num_classes=10)(jnp.argmax(model.apply(params)(*test_images[0]), axis=-1)) == jnp.argmax(*test_labels))) )(state)
            test_accuracies.append(test_acc)
            print("Epoch:", _, "Training Loss:", np.mean(train_losses[-5:]), "Test Accuracy:", np.mean(test_accuracies[-5:]))

class VanillaCNNModel(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 16 * 16)
        self.fc2 = nn.Dense(10)
        self.relu = nn.ReLU()

    def __call__(self, images):
        x = self.relu(self.conv1(images))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def init_state(self):
        return self.initialize({"params": self.apply_init})

    def apply_init(self, params):
        def kaiming_init(m):
            m.kernel = nn.initializers.xavier_uniform(rng)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.base.fill(0.)
        rng = random.PRNGKey(0)
        for name, m in self.apply_shapes(params):
            kaiming_init(m)
        return params

    def loss(self, params):
        images, labels = jax.tree_leaves((images, labels))
        logits = self.apply(params)(*images)
        loss = jnp.mean(jnp.sum(jnp.maximum(0., 1 - logits[..., labels] + jnp.log(jnp.exp(-logits[..., labels]) - 1)), axis=-1))
        return loss

batch_size = 32
data_dir = "./cifar-10-batches-bin"
train_loader = cifar10_loader(batch_size, data_dir, is_train=True)
test_loader = cifar10_loader(batch_size, data_dir, is_train=False)

for name, model in [("Vanilla", VanillaCNNModel())]:
    print(f"_________{name}_______________________")
    cifar10_train(model, train_loader, test_loader)