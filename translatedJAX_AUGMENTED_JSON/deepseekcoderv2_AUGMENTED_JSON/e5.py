import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from PIL import Image
from torchvision import transforms

class SomeLayer(nn.Module):
    features: int
    key: jax.random.PRNGKey

    def setup(self):
        self.key = self.make_rng('rng')  # Create a separate RNG for operations

    def __call__(self, x):
        subkey, _ = jax.random.split(self.key)  # Split the key for randomness
        return nn.relu(nn.Dense(self.features)(x))  # Use a dense layer with relu activation

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, _ = jax.random.split(key)  # Split the main RNG state for randomness
    return jax.random.normal(subkey, shape, dtype=dtype)  # Generate a tensor with specified dtype

def main():
    key = jax.random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)  # Generate random input tensor

    layer = SomeLayer(features=5, key=key)  # Pass the PRNGKey explicitly
    params = layer.init({'params': key, 'rng': key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={'rng': key})  # Process the input tensor with initialized parameters

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # Generate synthetic image data
    image = Image.fromarray(dummy_image_data)  # Convert to PIL Image
    output_np = np.array(output)  # Convert JAX array to NumPy array
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert NumPy array to PIL Image for heatmap overlay
    heatmap = transforms.Resize(image.size)(output_img)  # Resize the heatmap to match image size

    plt.imshow(image)  # Display the original image
    plt.imshow(heatmap, alpha=0.5, cmap='jet')  # Overlay the Grad-CAM heatmap
    plt.title("Predicted Class: Example Class")  # Set title for visualization
    plt.axis('off')  # Turn off axis display
    plt.show()  # Display the final image with heatmap overlay

if __name__ == "__main__":
    main()