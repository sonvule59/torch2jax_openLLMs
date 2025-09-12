import jax
from jax import numpy as jnp
from jax import grad, vmap, jit, random
from jax.scipy.optimize import minimize_scalar
import optimized_loss as opt_loss  # Assuming you have an optimized loss function defined in a separate file

# Generate synthetic sequential data
jnp.random.seed(42)
sequence_length = 10
num_samples = 100

X = jnp.linspace(0, 4 * 3.14159, num=num_samples).reshape((-1, 1))
y = jnp.sin(X)

def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

class CustomLSTMModel:
    def __init__(self, input_dim, hidden_units):
        self.input_dim = input_dim
        self.hidden_units = hidden_units

    def init(self, rng_key):
        wxi = jax.random.normal(rng_key, (self.input_dim, self.hidden_units))
        whi = jax.random.normal(rng_key, (self.hidden_units, self.hidden_units))
        bi = jax.random.normal(rng_key, (self.hidden_units,))
        wxf = jax.random.normal(rng_key, (self.input_dim, self.hidden_units))
        whf = jax.random.normal(rng_key, (self.hidden_units, self.hidden_units))
        bf = jax.random.normal(rng_key, (self.hidden_units,))
        wxo = jax.random.normal(rng_key, (self.input_dim, self.hidden_units))
        who = jax.random.normal(rng_key, (self.hidden_units, self.hidden_units))
        bo = jax.random.normal(rng_key, (self.hidden_units,))
        wxc = jax.random.normal(rng_key, (self.input_dim, self.hidden_units))
        whc = jax.random.normal(rng_key, (self.hidden_units, self.hidden_units))
        bc = jax.random.normal(rng_key, (self.hidden_units,))
        fc = jax.nn.Dense(1)

        return dict(wxi=wxi, whi=whi, bi=bi, wxf=wxf, whf=whf, bf=bf, wxo=wxo, who=who, bo=bo, wxc=wxc, whc=whc, bc=bc, fc=fc)

    @jit
    def forward(self, params, inputs, H_C=None):
        def step(t, H_C, *args):
            X_t = args[0][t]
            I_t = jax.nn.sigmoid(jax.jacobian(lambda h: jnp.matmul(X_t, params['wxi']) + jnp.matmul(h, params['whi']) + params['bi'][None, :])(H_C))
            F_t = jax.nn.sigmoid(jax.jacobian(lambda h: jnp.matmul(X_t, params['wxf']) + jnp.matmul(h, params['whf']) + params['bf'][None, :])(H_C))
            O_t = jax.nn.sigmoid(jax.jacobian(lambda h: jnp.matmul(X_t, params['wxo']) + jnp.matmul(h, params['who']) + params['bo'][None, :])(H_C))
            C_tilde = jax.nn.tanh(jax.jacobian(lambda h: jnp.matmul(X_t, params['wxc']) + jnp.matmul(h, params['whc']) + params['bc'][None, :])(H_C))
            C = F_t * C_C[0] + I_t * C_tilde
            H = O_t * jax.nn.tanh(C)
            return H, C, X_t

        batch_size, seq_len, _ = inputs.shape
        if not H_C:
            H = jnp.zeros((batch_size, self.hidden_units))
            C = jnp.zeros((batch_size, self.hidden_units))
        else:
            H, C = H_C

        def body(carry, t):
            H_, C_, X_t = carry
            H_, C_ = step(t, (H_, C_), inputs)
            return H_, C_, (X_t,)

        def init_carry():
            return H, C, (X_seq[0],)

        _, final_h, _ = jax.lax.fori_loop(0, seq_len, body, init_carry=init_carry)
        outputs = final_h.reshape((-1, self.hidden_units))
        pred = params['fc'].apply(outputs).squeeze()
        return pred, (final_h, C)

# Define the LSTM Model
class LSTMModel:
    def __init__(self):
        self.lstm = jax.nn.LSTMCell(1, 50)
        self.fc = jax.nn.Dense(1)

    @jit
    def forward(self, inputs):
        out, _ = jax.lax.fori_loop(0, len(inputs), lambda i: self.lstm(inputs[i]), init=self.lstm(jax.random.normal((1, 50))))
        out = self.fc(out[-1])
        return out

# Initialize the model, loss function, and optimizer
params_custom = CustomLSTMModel(1, 50).init(random.PRNGKey(42))
params_inbuilt = LSTMModel().init(random.PRNGKey(42))
loss = opt_loss.MSELoss
optimizer = jax.optimizers.Adam

# Training loop for the custom model
num_steps = 500
for i in range(num_steps):
    # Forward pass
    state = (None, None) if i == 0 else (params_custom[1], params_custom[2])
    pred, state = CustomLSTMModel(**params_custom).forward(X_seq, state=state)
    loss_value = loss(pred[:, -1, :], y_seq[-1]) # Use the last output of the LSTM

    # Backward pass and optimization
    grads = jax.grad(loss)(CustomLSTMModel, (params_custom,), (X_seq, state))[0]
    params_custom = optimizer(step_size=0.01).update(params_custom, grads)

    # Log progress every 50 steps
    if i % 50 == 0:
        print(f"Step [{i + 1}/{num_steps}], Loss: {loss_value}")

# Training loop for the inbuilt model
num_steps = 500
for i in range(num_steps):
    # Forward pass
    pred = LSTMModel().forward(X_seq)
    loss_value = loss(pred, y_seq[-1])

    # Backward pass and optimization
    grads = jax.grad(loss)(LSTMModel, (None,), (X_seq))[0]
    params_inbuilt = optimizer(step_size=0.01).update(params_inbuilt, grads)

    # Log progress every 50 steps
    if i % 50 == 0:
        print(f"Step [{i + 1}/{num_steps}], Loss: {loss_value}")

# Testing on new data
test_steps = 100  # Ensure this is greater than sequence_length
X_test = jnp.linspace(0, 5 * 3.14159, steps=test_steps).reshape((-1, 1))
y_test = jnp.sin(X_test)

# Create test input sequences
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

with jax.random.PRNGKey(42):
    state_custom = (None, None) if i == 0 else CustomLSTMModel(**params_custom).forward(X_seq, state=state)[1]
    pred_custom, _ = CustomLSTMModel(**params_custom).forward(X_test_seq, state=state_custom)
    pred_inbuilt = LSTMModel().forward(X_test_seq)
pred_custom = pred_custom[:, -1]
pred_inbuilt = pred_inbuilt.squeeze()
print(f"Predictions with Custom Model for new sequence: {pred_custom}")
print(f"Predictions with In-Built Model: {pred_inbuilt}")

#Plot the predictions
import jaxplot as jp
jp.line(y_test, 'Ground Truth', marker='.', markersize=2)
jp.line(pred_custom, 'custom model', marker='.', markersize=2)
jp.line(pred_inbuilt, 'inbuilt model', marker='.', markersize=2)
jp.show()