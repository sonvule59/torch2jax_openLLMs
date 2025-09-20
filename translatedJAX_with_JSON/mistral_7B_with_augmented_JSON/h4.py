import jax
from jax import random, jnn, nn, numpy as jnp
import torchvision.transforms as transforms
import PIL.Image as Image
import matplotlib.pyplot as plt

class SomeLayer(jnn.Module):
    def __init__(self, features, key=None):
        super().__init__()
        self.dense = nn.Dense(features)

    def init(self, rngs, input_shape=None):
        if input_shape is None:
            input_shape = (None,) + (1,) * self.dense.out_features
        key, subkey = random.split(rngs["rng"])
        params = self.dense.init(subkey, jnp.zeros(self.dense.rng_key_shape, dtype=jnp.float32))
        return {"params": params}

    def apply(self, params, x):
        return jnn.relu(self.dense.apply(params, x))

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
    params = layer.init({"params": key, "rng": key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)  # Convert JAX array to NumPy
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert to PIL Image
    heatmap = transforms.Resize(image.size)(output_img)

    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")  # Example title
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()