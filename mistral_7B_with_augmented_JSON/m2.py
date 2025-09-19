import jax
import jax.numpy as jnp
from flax import linen as jnn
from PIL import Image, transforms
import matplotlib.pyplot as plt
import numpy as np

class SomeLayer(jnn.Module):
    features: int

    def setup(self):
        self.dense = nn.Dense(self.features)

    def __call__(self, x):
        return jnn.relu(self.dense(x))  # Use a dense layer with relu activation

    def init(self, rngs, input_shape=None):
        key = rngs['rng']  # Extract the PRNGKey for randomness
        subkey, self.key = jax.random.split(key)  # Split the key for a new operation
        return {'params': self.dense.init(subkey, input_shape)}

    def apply(self, params, x, rngs=None):
        key = rngs['rng'] if rngs is not None else jax.random.PRNGKey(0)  # Use a default PRNG key if none provided
        subkey, self.key = jax.random.split(self.key)  # Split the key for a new operation
        return self.dense.apply(params['params'], x, rngs={'rng': subkey})

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")  # Error handling for missing key
    return jax.random.normal(key, shape, dtype=dtype)  # Generate a tensor with specified dtype

def main():
    key = jax.random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)  # Use the modified function

    layer = SomeLayer(features=5, key=key)  # Pass the PRNGKey explicitly
    params = layer.init({"params": key, "rng": key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    # generate a synthetic image in memory
    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)

    # Convert JAX array to NumPy, then to PIL Image, and apply Resize
    output_np = np.array(output)
    output_img = Image.fromarray(output_np.astype(np.uint8))
    heatmap = transforms.Resize(image.size)(output_img)

    # Display the image with the Grad-CAM heatmap
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")  # Example title
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()