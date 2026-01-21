import tensorflow as tf
from tensorflow.keras import layers, models, Input, losses, optimizers
import numpy as np

# --- Configuration Class (Good OOP practice to separate config) ---
class GANConfig:
    def __init__(self):
        self.seq_len = 30
        self.z_dim = 64
        self.hidden_dim = 128
        self.num_features = 5
        self.num_regimes = 3
        self.batch_size = 64
        self.lr = 0.0002
        self.epochs = 100

# --- The Main GAN Class ---
class MarketGAN(tf.keras.Model):
    def __init__(self, config):
        super(MarketGAN, self).__init__()
        self.config = config
        
        # Build components internally upon initialization
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Trackers for metrics (to visualize during training)
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(MarketGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def _build_generator(self):
        """Internal method to construct the Generator architecture"""
        # Inputs
        noise_input = Input(shape=(self.config.seq_len, self.config.z_dim))
        label_input = Input(shape=(1,), dtype='int32')
        
        # Label processing
        label_emb = layers.Embedding(self.config.num_regimes, self.config.z_dim)(label_input)
        label_emb = layers.Reshape((1, self.config.z_dim))(label_emb)
        label_tiled = tf.tile(label_emb, [1, self.config.seq_len, 1])
        
        # Combine
        merged = layers.Concatenate()([noise_input, label_tiled])
        
        # Network
        x = layers.LSTM(self.config.hidden_dim, return_sequences=True)(merged)
        x = layers.LSTM(self.config.hidden_dim, return_sequences=True)(x)
        output = layers.TimeDistributed(layers.Dense(self.config.num_features))(x)
        
        return models.Model([noise_input, label_input], output, name="Generator")

    def _build_discriminator(self):
        """Internal method to construct the Discriminator architecture"""
        # Inputs
        seq_input = Input(shape=(self.config.seq_len, self.config.num_features))
        label_input = Input(shape=(1,), dtype='int32')
        
        # Label processing
        label_emb = layers.Embedding(self.config.num_regimes, self.config.num_features)(label_input)
        label_emb = layers.Reshape((1, self.config.num_features))(label_emb)
        label_tiled = tf.tile(label_emb, [1, self.config.seq_len, 1])
        
        # Combine
        merged = layers.Concatenate()([seq_input, label_tiled])
        
        # Network
        x = layers.LSTM(self.config.hidden_dim, return_sequences=False)(merged)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model([seq_input, label_input], output, name="Discriminator")

    def train_step(self, data):
        """
        The custom training logic. This is called automatically by model.fit().
        'data' contains (real_sequences, labels).
        """
        real_sequences, labels = data
        batch_size = tf.shape(real_sequences)[0]

        # 1. Prepare Inputs
        # Create random noise for the generator
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.config.seq_len, self.config.z_dim)
        )

        # 2. Train Discriminator
        with tf.GradientTape() as tape:
            # Generate fake data
            fake_sequences = self.generator([random_latent_vectors, labels], training=True)
            
            # Get predictions
            real_predictions = self.discriminator([real_sequences, labels], training=True)
            fake_predictions = self.discriminator([fake_sequences, labels], training=True)
            
            # Calculate D Loss
            d_loss_real = self.loss_fn(tf.ones_like(real_predictions), real_predictions)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
            d_loss = d_loss_real + d_loss_fake

        # Update D weights
        grads_d = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_weights))

        # 3. Train Generator
        with tf.GradientTape() as tape:
            # Re-sample noise (optional, but standard practice)
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.config.seq_len, self.config.z_dim)
            )
            
            fake_sequences = self.generator([random_latent_vectors, labels], training=True)
            fake_predictions_for_g = self.discriminator([fake_sequences, labels], training=True)
            
            # Calculate G Loss (Generator wants D to predict '1' for fakes)
            g_loss = self.loss_fn(tf.ones_like(fake_predictions_for_g), fake_predictions_for_g)

        # Update G weights
        grads_g = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_weights))

        # 4. Update Metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

    def generate(self, regime_label, num_samples):
        """
        Public method to generate synthetic scenarios after training.
        """
        regime_label = int(regime_label)
        noise = tf.random.normal(shape=(num_samples, self.config.seq_len, self.config.z_dim))
        labels = tf.constant([regime_label] * num_samples, dtype=tf.int32)
        
        generated_data = self.generator([noise, labels], training=False)
        return generated_data.numpy()

# --- Mock Data Helper ---
def get_dataset(config):
    # Generates dummy sine waves
    X_train = []
    y_train = []
    for _ in range(1000):
        regime = np.random.randint(0, config.num_regimes)
        freq = [1, 2, 4][regime]
        t = np.linspace(0, 4*np.pi, config.seq_len)
        sample = np.zeros((config.seq_len, config.num_features))
        for f in range(config.num_features):
             sample[:, f] = np.sin(t * freq + f) + np.random.normal(0, 0.1, config.seq_len)
        X_train.append(sample)
        y_train.append(regime)
    
    return np.array(X_train, dtype='float32'), np.array(y_train, dtype='int32')

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Setup
    config = GANConfig()
    gan = MarketGAN(config)
    
    # 2. Compile
    gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=config.lr),
        g_optimizer=optimizers.Adam(learning_rate=config.lr),
        loss_fn=losses.BinaryCrossentropy(from_logits=False)
    )
    
    # 3. Data
    X, y = get_dataset(config)
    
    # 4. Train (The OOP benefit: standard .fit call)
    print("Starting Training...")
    gan.fit(X, y, batch_size=config.batch_size, epochs=config.epochs)
    
    # 5. Usage
    print("\nGenerating Bear Market Scenarios (Regime 0)...")
    synthetic_bear = gan.generate(regime_label=0, num_samples=5)
    print("Synthetic Data Shape:", synthetic_bear.shape)
    
    # Example: Access the internal generator directly if needed to save weights
    # gan.generator.save("my_generator.h5")