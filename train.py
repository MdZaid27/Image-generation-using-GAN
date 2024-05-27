import tensorflow as tf
import time
import os

os.environ["TF_DISABLE_GPU_WARNINGS"] = "1"
from data.load_data import load_mnist_data, create_dataset
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.losses import discriminator_loss, generator_loss
from utils.visualization import generate_and_save_images

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Hyperparameters
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
latent_dim = 100
num_examples_to_generate = 16

# Load data
train_images = load_mnist_data()
train_dataset = create_dataset(train_images, BUFFER_SIZE, BATCH_SIZE)

# Build models
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Seed for consistent image generation
seed = tf.random.normal([num_examples_to_generate, latent_dim])


# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed)

        print(f"Time for epoch {epoch + 1} is {time.time() - start:.2f} sec")

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


# Train the GAN
train(train_dataset, EPOCHS)
