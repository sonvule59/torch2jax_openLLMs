import jax
from jax import random, jnn, nn
from PIL import Image
from torchvision.transforms import Resize
import numpy as np
import matplotlib.pyplot as plt

class SomeLayer(jax.Module):
    def __init__(self, features, key):
        super().__init__(name='SomeLayer')
        self.dense = nn.Dense(features)

    def init(self, params, rngs=None):
        self.key = rngs['rng']
        return params

    def __call__(self, x):
        subkey, self.key = jax.random.split(self.key)
        return jnn.relu(self.dense(x))

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = random.split(key)
    return random.normal(subkey, shape, dtype=dtype)

def main():
    key = random.PRNGKey(0)
    input_tensor_shape = (10, 10)
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer(features=5, key=key)
    params = layer.init({'params': key, 'rng': key}, input_tensor)
    output = layer.apply(params, input_tensor, rngs={'rng': key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)

    output_np = np.array(output)
    output_img = Image.fromarray(output_np.astype(np.uint8))
    heatmap = Resize(image.size)(output_img)

    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()