import jax
from jax import random, nn, jnn
from PIL import Image
import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

class SomeLayer(nn.Module):
    def __init__(self, features, key=None):
        super().__init__()
        self.dense = nn.Dense(features)

    def init(self, rngs, input_shape):
        subkey, key = random.split(rngs['rng'])
        params = self.dense.init(subkey, input_shape)
        return {'params': params}

    def apply(self, params, x, rngs=None):
        subkey, key = random.split(rngs['rng'] if rngs else key)
        return jnn.relu(self.dense.apply(params, x))

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    return jnp.random.normal(key, shape, dtype)

def main():
    rng = random.PRNGKey(0)
    input_tensor_shape = (10, 10)
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=rng)

    layer = SomeLayer(features=5, key=rng)
    params = layer.init({'params': rng, 'rng': rng}, input_tensor)
    output = layer.apply(params, input_tensor, rngs={'rng': rng})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)
    output_img = Image.fromarray(output_np.astype(np.uint8))
    heatmap = Resize(size=image.size)(output_img)

    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()