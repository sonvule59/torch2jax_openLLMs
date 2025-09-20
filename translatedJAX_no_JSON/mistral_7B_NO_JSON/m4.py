import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim
import numpy as np

np.random.seed(42)
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

ct_images = jnp.random.randn((batch, num_slices, channels, width, height))
segmentation_masks = jnp.greater(jnp.random.randn((batch, num_slices, 1, width, height)), 0).astype(jnp.float32)

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

class MedCNN(nn.Module):
    def setup(self):
        self.backbone = Backbone()
        self.conv1 = nn.Conv3D(512, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3D(64, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv_transpose1 = nn.ConvTranspose3D(64, 32, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.conv_transpose2 = nn.ConvTranspose3D(32, 16, kernel_size=(1, 8, 8), stride=(1, 8, 8))
        self.final_conv = nn.Conv3D(16, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def __call__(self, x):
        b, d, c, w, h = x.shape
        x = x.reshape((b*d, c, w, h))
        features = self.backbone(x)
        x = features.reshape((b, d, features.shape[-3], features.shape[-2], features.shape[-1]))
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv_transpose1(x))
        x = self.relu(self.conv_transpose2(x))
        x = jax.nn.sigmoid(self.final_conv(x))
        return x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18()
        self.avgpool = nn.AvgPool3D((2, 2, 2))
        self.flatten = nn.Flatten()

    def __call__(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

def compute_dice_loss(pred, labels, eps=1e-8):
    numerator = jnp.sum(pred * labels, axis=(1, 2, 3, 4)) + eps
    denominator = jnp.sum(pred, axis=(1, 2, 3, 4)) + jnp.sum(labels, axis=(1, 2, 3, 4)) + eps
    return numerator / denominator

resnet_model = resnet18()
backbone = Backbone()
model = MedCNN(backbone=backbone)
optimizer = optim.Adam(model.parameters(), learning_rate=0.01)

epochs = 5
for epoch in range(epochs):
    grad_fn = jax.grad(compute_dice_loss)(segmentation_masks)
    pred = model(ct_images)
    loss = grad_fn(pred)
    optimizer.apply_gradient(loss)
    print(f"Loss at epoch {epoch}: {loss}")