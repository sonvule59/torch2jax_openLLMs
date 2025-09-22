import jax
from flax import linen as nn
import optax
import numpy as np

# Define a simple Language Model (e.g., an LSTM-based model) using Flax
class LanguageModel(nn.Module):
    vocab_size: int
    embed_size: int
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)(x)
        lstm = nn.LSTM(hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False, kernel_init=nn.initializers.lecun_uniform())
        lstm_out, _ = lstm(embedding)
        output = nn.Dense(features=self.vocab_size, kernel_init=nn.initializers.lecun_uniform())(lstm_out[-1])
        logits = nn.softmax(output)
        return logits

# Create synthetic training data (JAX equivalent of PyTorch's random integers)
np.random.seed(42)
vocab_size = 50
seq_length = 10
batch_size = 32
X_train = np.random.randint(0, vocab_size, (batch_size, seq_length))
y_train = np.random.randint(0, vocab_size, (batch_size,))

# Initialize the model, loss function, and optimizer
embed_size = 64
hidden_size = 128
num_layers = 2
model = LanguageModel(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)

# Define loss function (JAX equivalent of PyTorch's CrossEntropyLoss)
def cross_entropy_loss(logits, labels):
    one_hot_labels = np.eye(vocab_size)[labels]
    return -np.sum(one_hot_labels * np.log(logits + 1e-9)) / batch_size

# Define optimizer (JAX equivalent of PyTorch's Adam)
optimizer = optax.adam(learning_rate=0.001)
state = optimizer.init(model.params)

# Training loop
epochs = 5
for epoch in range(epochs):
    model = model.apply(state, x=X_train)
    logits = model['logits']
    loss = cross_entropy_loss(logits, y_train)
    
    # Update the model parameters (in practice, you would use a different approach for optimization in JAX)
    updates, state = optimizer.update(model.params, model.grads, state)
    new_params = optax.apply_updates(model.params, updates)
    
    # Log progress every epoch
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")