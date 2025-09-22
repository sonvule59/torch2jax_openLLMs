import jax
import jax.numpy as jnp
import flax.linen as nn
from PIL import Image
from torchvision import transforms
import numpy as np

class SomeLayer(nn.Module):
    features: int
    key: jax.random.PRNGKey

    def setup(self):
        self.key = self.make_rng('rng')  # Use the make_rng method to create a new RNG stream

    def __call__(self, x):
        subkey, self.key = jax.random.split(self.key)  # Split the key for a new operation
        return nn.relu(nn.Dense(self.features)(x))  # Use a dense layer with relu activation

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")  # Error handling for missing key
    subkey, key = jax.random.split(key)  # Split key for randomness
    return jax.random.normal(subkey, shape, dtype=dtype)  # Generate a tensor with specified dtype

def main():
    key = jax.random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)  # Generate random tensor

    layer = SomeLayer(features=5, key=key)  # Pass the PRNGKey explicitly
    params = layer.init({"params": key, "rng": key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={"rng": key})  # Process the input tensor with initialized parameters

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # Generate synthetic image data
    image = Image.fromarray(dummy_image_data)  # Create PIL Image from numpy array
    output_np = np.array(output)  # Convert JAX array to NumPy array
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert NumPy array to PIL Image
    heatmap = transforms.Resize(image.size)(output_img)  # Resize the image using torchvision.transforms.Resize

    plt.imshow(image)  # Display the original image
    plt.imshow(heatmap, alpha=0.5, cmap='jet')  # Overlay the heatmap on the image
    plt.title("Predicted Class: Example Class")  # Set title for the plot
    plt.axis('off')  # Turn off axis display
    plt.show()  # Display the final image with heatmap overlay

if __name__ == "__main__":
    main()