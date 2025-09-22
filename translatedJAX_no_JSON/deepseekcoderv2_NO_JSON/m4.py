import jax
from jax import numpy as jnp
import jaxlib
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from torchvision import models
import optax  # Importing Optax for optimizer creation

# Generate synthetic CT-scan data (batches, slices, RGB) and associated segmentation masks
jax.random.PRNGKey(42)
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

ct_images = jax.random.normal(jax.random.PRNGKey(0), (batch, num_slices, channels, width, height))
segmentation_masks = (jax.random.uniform(jax.random.PRNGKey(1), (batch, num_slices, 1, width, height)) > 0.5).astype(jnp.float32)

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

# Define the MedCNN class and its forward method
class MedCNN(nn.Module):
    backbone: nn.Module  # Assuming backbone is a ResNet model
    out_channel: int = 1

    @nn.compact
    def __call__(self, x):
        b, d, c, w, h = x.shape  # Input size: [B, D, C, W, H]
        print(f"Input shape [B, D, C, W, H]: {b, d, c, w, h}")
        
        x = x.reshape((b * d, c, w, h))  # Input to Resent 2DConv layers [B*D, C, W, H]
        features = self.backbone(x)
        print(f"ResNet output shape[B*D, C, W, H]: {features.shape}")
        
        new_c, new_w, new_h = features.shape[-3:]
        x = features.reshape((b, d, new_c, new_w, new_h))  # [B, D, C, W, H]
        x = jnp.transpose(x, (0, 2, 1, 3, 4))  # rearrange for 3DConv layers [B, C, D, W, H]
        print(f"Reshape Resnet output for 3DConv #1 [B, C, D, W, H]: {x.shape}")
        
        # Downsampling
        x = nn.relu(self.conv1(x))
        print(f"Output shape 3D Conv #1: {x.shape}")
        x = nn.relu(self.conv2(x))
        print(f"Output shape 3D Conv #2: {x.shape}")
        
        # Upsampling
        x = nn.relu(self.conv_transpose1(x))
        print(f"Output shape 3D Transposed Conv #1: {x.shape}")
        x = nn.relu(self.conv_transpose2(x))
        print(f"Output shape 3D Transposed Conv #2: {x.shape}")

        # Final segmentation
        x = jax.nn.sigmoid(self.final_conv(x))
        print(f"Final shape: {x.shape}")
        
        return x

def compute_dice_loss(pred, labels, eps=1e-8):
    '''
    Args
    pred: [B, D, 1, W, H]
    labels: [B, D, 1, W, H]
    
    Returns
    dice_loss: [B, D, 1, W, H]
    '''
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator

resnet_model = models.resnet18(pretrained=True)
resnet_model = nn.Sequential([*list(resnet_model.children())[:-2]])

model = MedCNN(backbone=resnet_model)

optimizer = optax.adam(learning_rate=0.01)
state = train_state.TrainState.create(apply_fn=model.apply, params=model.init(jax.random.PRNGKey(2), ct_images), tx=optimizer)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        pred = model.apply({'params': params}, batch['ct_images'])
        l = compute_dice_loss(pred, batch['segmentation_masks'])
        return l, pred
    
    (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, pred

epochs = 5
for epoch in range(epochs):
    for batch in train_batches:  # Assuming train_batches is defined elsewhere and contains the data
        state, loss, _ = train_step(state, batch)
        print(f"Loss at epoch {epoch}: {loss}")